// ──────────────────────────────────────────────
// Gemini provider — chat runner
// ──────────────────────────────────────────────
//
// backend = subscription → agy print mode on the Google sign-in only. When
//                          agy's settings.json is set up for API-key billing
//                          (modelProvider, or a GOOGLE_GEMINI_BASE_URL) the
//                          request is refused, and agy always runs with every
//                          Gemini key / base-URL / gateway variable removed.
// backend = api          → direct OpenAI-compatible HTTP with a Gemini key to
//                          Google's OpenAI-compatible endpoint, or to the base
//                          URL that comes from the same place as the key.
// backend = auto         → agy when installed, following agy's own
//                          configuration (including its API-key mode); else
//                          the api backend when a key exists.
//
// agy turns run the isolated chat agent (see agy.js). The CLI takes one text
// prompt on STDIN, which agy caps at 48,000 tokens (192,000 bytes) before
// tail-truncating the end of the conversation. Short chats are folded directly
// into STDIN; longer chats spill <system_instructions> and older history turns
// into a per-run custom agent definition (sent via systemInstruction, which is
// exempt from the per-step STDIN cap) so the recent history and current user
// turn at the end of STDIN are never truncated. Images on the current turn are
// refused on agy (use the api backend); older ones become "[image omitted]".

import { isForeignKey } from '../common/keys.js';
import {
    agyLaunchSpec, runAgy, readAntigravitySettings, agyBilling,
    AGY_MAX_STDIN_BYTES, AGY_MAX_TOTAL_BYTES, CHAT_AGENT_MD, CHAT_AGENT_NAME, buildChatAgentMd,
} from './agy.js';
import { resolveGeminiModel, listGeminiModels, geminiCatalog, agyModelId } from './models.js';
import { runOpenAiCompat } from '../common/openai-compat.js';
import {
     foldConversation, splitConversation, formatHistoryTurnForFold, formatCurrentTurnForFold,
} from '../common/messages.js';
import { toOpenAiUsage } from '../common/sse.js';
import { abortError } from '../common/abort.js';
import { bearerFromRequest } from '../claude/env.js';

const TAG = '[subscriptions/gemini]';
const GOOGLE_OPENAI_COMPAT_BASE = 'https://generativelanguage.googleapis.com/v1beta/openai';
const GOOGLE_HOST = 'generativelanguage.googleapis.com';
const utf8Bytes = (s) => Buffer.byteLength(String(s ?? ''), 'utf8');

function splitUtf8Tail(text, maxTailBytes) {
    const str = String(text ?? '');
    if (maxTailBytes <= 0) return { head: str, tail: '' };
    if (utf8Bytes(str) <= maxTailBytes) return { head: '', tail: str };
    const buf = Buffer.from(str, 'utf8');
    let start = buf.length - maxTailBytes;
    while (start < buf.length && (buf[start] & 0xc0) === 0x80) start++;
    const nl = buf.indexOf(0x0a, start);
    if (nl !== -1 && nl - start <= 2048 && nl + 1 < buf.length) start = nl + 1;
    return {
        head: buf.subarray(0, start).toString('utf8').trimEnd(),
        tail: buf.subarray(start).toString('utf8'),
    };
}

function sliceUtf8Head(text, maxHeadBytes) {
    const str = String(text ?? '');
    if (maxHeadBytes <= 0) return '';
    if (utf8Bytes(str) <= maxHeadBytes) return str;
    const buf = Buffer.from(str, 'utf8');
    let end = maxHeadBytes;
    while (end > 0 && (buf[end] & 0xc0) === 0x80) end--;
    return buf.subarray(0, end).toString('utf8');
}

/**
 * Fold and partition an OpenAI messages[] array for agy so that:
 *  1. STDIN (`prompt`) never exceeds `maxStdinBytes` (180 KB, below agy's
 *     192,000-byte MaxTokensPerUserInputStep tail-truncation limit).
 *  2. Combined `agent.md` + `prompt` never exceeds `maxTotalBytes` (720 KB,
 *     below agy's ~765,000-byte TruncationThresholdTokens ceiling).
 *  3. When the full folded conversation fits within `maxStdinBytes`, all of it
 *     stays in `prompt` with no extra `agentSections`.
 *  4. When it exceeds `maxStdinBytes`, `<system_instructions>` and any older
 *     history turns that do not fit alongside the newest history turns +
 *     current turn in `prompt` are spilled into `agentSections` in strict
 *     chronological order. If total size still exceeds `maxTotalBytes`, the
 *     oldest history turns are dropped first.
 *
 * @param {object[]} messages
 * @param {{ userLabel?: string, assistantLabel?: string, maxStdinBytes?: number, maxTotalBytes?: number }} [opts]
 * @returns {{ prompt: string, agentSections: Array<{ heading: string, content: string }>, spilledToAgent: boolean, droppedHistoryTurns: number }}
 */
export function foldConversationForAgy(messages, opts = {}) {
    const {
        userLabel = 'User',
        assistantLabel = 'Assistant',
        maxStdinBytes = AGY_MAX_STDIN_BYTES,
        maxTotalBytes = AGY_MAX_TOTAL_BYTES,
    } = opts;
    const fullFolded = foldConversation(messages, { includeSystem: true, userLabel, assistantLabel });
    if (utf8Bytes(fullFolded) <= maxStdinBytes && utf8Bytes(CHAT_AGENT_MD) + utf8Bytes(fullFolded) <= maxTotalBytes) {
        return { prompt: fullFolded, agentSections: [], spilledToAgent: false, droppedHistoryTurns: 0 };
    }

    const { system, history, current, prefill, currentMessage } = splitConversation(messages);
    const lines = [];
    for (const m of history) {
        const formatted = formatHistoryTurnForFold(m, { userLabel, assistantLabel });
        if (formatted) lines.push(formatted);
    }

    const currentParts = formatCurrentTurnForFold({ current, prefill, currentMessage, userLabel, assistantLabel });
    const currentBlock = currentParts.join('\n\n');

    let sysText = system ? String(system) : '';
    let currentOverflow = '';
    let stdinCurrent = currentBlock;
    let earlierLines = [];
    let recentLines = [];

    if (utf8Bytes(currentBlock) > maxStdinBytes) {
        const split = splitUtf8Tail(currentBlock, maxStdinBytes);
        currentOverflow = split.head;
        stdinCurrent = split.tail;
        earlierLines = [...lines];
    } else {
        const wrapperBytes = utf8Bytes('<conversation_history>\n\n</conversation_history>\n\n');
        const historyBudget = maxStdinBytes - utf8Bytes(stdinCurrent) - wrapperBytes;
        let recentBytes = 0;
        let splitIdx = lines.length;
        if (historyBudget > 0) {
            for (let i = lines.length - 1; i >= 0; i--) {
                const lineBytes = utf8Bytes(lines[i]);
                const sepBytes = recentBytes > 0 ? 2 : 0;
                if (recentBytes + sepBytes + lineBytes <= historyBudget) {
                    recentBytes += sepBytes + lineBytes;
                    splitIdx = i;
                } else {
                    break;
                }
            }
        }
        earlierLines = lines.slice(0, splitIdx);
        recentLines = lines.slice(splitIdx);
    }

    const buildPrompt = (recent) => (recent.length
        ? `<conversation_history>\n${recent.join('\n\n')}\n</conversation_history>\n\n${stdinCurrent}`
        : stdinCurrent);

    const buildSections = (sys, earlier, curOver) => {
        const out = [];
        if (sys && sys.trim()) out.push({ heading: 'System_Instructions', content: sys });
        if (earlier.length) out.push({ heading: 'Earlier_Conversation_History', content: earlier.join('\n\n') });
        if (curOver && curOver.trim()) out.push({ heading: 'Current_Turn_Context', content: curOver });
        return out;
    };

    const sampleName = `${CHAT_AGENT_NAME}-000000-0000000000000-0000`;
    const totalSize = (sys, earlier, recent, curOver) => {
        const sections = buildSections(sys, earlier, curOver);
        const agentMd = buildChatAgentMd({ name: sampleName, sections });
        return utf8Bytes(agentMd) + utf8Bytes(buildPrompt(recent));
    };

    let droppedHistoryTurns = 0;
    while (earlierLines.length > 0 && totalSize(sysText, earlierLines, recentLines, currentOverflow) > maxTotalBytes) {
        earlierLines.shift();
        droppedHistoryTurns++;
    }
    while (recentLines.length > 0 && totalSize(sysText, earlierLines, recentLines, currentOverflow) > maxTotalBytes) {
        recentLines.shift();
        droppedHistoryTurns++;
    }
    while (currentOverflow && totalSize(sysText, earlierLines, recentLines, currentOverflow) > maxTotalBytes) {
        const excess = totalSize(sysText, earlierLines, recentLines, currentOverflow) - maxTotalBytes;
        const target = Math.max(0, utf8Bytes(currentOverflow) - excess - 32);
        currentOverflow = target > 0 ? splitUtf8Tail(currentOverflow, target).tail : '';
    }
    while (sysText && totalSize(sysText, earlierLines, recentLines, currentOverflow) > maxTotalBytes) {
        const excess = totalSize(sysText, earlierLines, recentLines, currentOverflow) - maxTotalBytes;
        const target = Math.max(0, utf8Bytes(sysText) - excess - 32);
        sysText = target > 0 ? sliceUtf8Head(sysText, target) : '';
    }

    const prompt = buildPrompt(recentLines);
    const agentSections = buildSections(sysText, earlierLines, currentOverflow);
    return {
        prompt,
        agentSections,
        spilledToAgent: agentSections.length > 0,
        droppedHistoryTurns,
    };
}

/** Keys that clearly belong to another vendor (SillyTavern's Custom API key field is shared). */
export function isOtherVendorKey(key) {
    return isForeignKey('gemini', key);
}

/**
 * Key + base URL for the direct API backend. The base URL is paired with the
 * key's source, so a key never travels to an endpoint configured for another
 * one:
 *   ST_SUBSCRIPTIONS_GEMINI_BASE_URL          explicit override, any key
 *   GEMINI_API_KEY / GOOGLE_API_KEY (env)     GOOGLE_GEMINI_BASE_URL (env)
 *   agy settings.json env block               GOOGLE_GEMINI_BASE_URL from the same block
 *   Custom API key field / plugin key         Google's endpoint (set the override for anything else)
 * @param {import('express').Request|null} req
 * @param {{ settings?: object|null, env?: Record<string,string|undefined> }} [opts] test seams
 */
export function discoverGeminiApiCredentials(req, { settings = readAntigravitySettings(), env = process.env } = {}) {
    const pick = (v) => (typeof v === 'string' && v.trim().length > 0 ? v.trim() : null);
    const header = pick(bearerFromRequest(req));
    const candidates = [
        [header && !isOtherVendorKey(header) ? header : null, 'api-key-field'],
        [pick(env.ST_SUBSCRIPTIONS_GEMINI_API_KEY), 'ST_SUBSCRIPTIONS_GEMINI_API_KEY'],
        [pick(env.GEMINI_API_KEY), 'GEMINI_API_KEY'],
        [pick(env.GOOGLE_API_KEY), 'GOOGLE_API_KEY'],
        [pick(settings?.env?.GEMINI_API_KEY), 'antigravity-settings'],
    ];
    const hit = candidates.find(([key]) => key);
    if (!hit) return null;
    const [key, source] = hit;
    let baseUrl = pick(env.ST_SUBSCRIPTIONS_GEMINI_BASE_URL);
    if (!baseUrl) {
        if (source === 'antigravity-settings') baseUrl = pick(settings?.env?.GOOGLE_GEMINI_BASE_URL);
        else if (source === 'GEMINI_API_KEY' || source === 'GOOGLE_API_KEY') baseUrl = pick(env.GOOGLE_GEMINI_BASE_URL);
    }
    return { key, baseUrl: baseUrl || GOOGLE_OPENAI_COMPAT_BASE, source };
}

function mapAgyUsage(u) {
    if (!u) return null;
    return toOpenAiUsage({
        input: Math.max(0, (u.input_tokens ?? 0) - (u.cache_read_tokens ?? 0)),
        output: u.output_tokens ?? 0,
        cacheRead: u.cache_read_tokens ?? 0,
        reasoning: u.thinking_tokens ?? 0,
    });
}

const hasImage = (m) => Array.isArray(m?.content) && m.content.some((p) => p?.type === 'image_url');

function errorType(err) {
    if (err?.httpStatus === 429) return 'rate_limit_error';
    return err?.httpStatus && err.httpStatus < 500 ? 'invalid_request_error' : 'server_error';
}

async function runViaAgy({ messages, model, settings, writer, signal, allowApiKey }) {
    const { currentMessage } = splitConversation(messages);
    if (hasImage(currentMessage)) {
        throw Object.assign(new Error('Images are not supported through the Antigravity CLI (agy) backend. Switch the Gemini backend to "API" to send images, or remove the image.'), { httpStatus: 400 });
    }
    const omitted = (messages ?? []).filter(hasImage).length;
    if (omitted) console.warn(`${TAG} agy takes text only: ${omitted} earlier message(s) with images sent as "[image omitted]"`);

    await listGeminiModels({ live: false });
    const resolved = resolveGeminiModel(model, settings.gemini.effort, geminiCatalog());
    const { prompt, agentSections, spilledToAgent, droppedHistoryTurns } = foldConversationForAgy(messages);
    if (droppedHistoryTurns > 0) {
        console.warn(`${TAG} conversation exceeded agy's context ceiling; dropped ${droppedHistoryTurns} oldest history turn(s) so recent history and the current turn stay intact`);
    }
    const effortNote = resolved.agyEffort ? ` --effort ${resolved.agyEffort}` : (resolved.effort ? ` (effort ${resolved.effort})` : '');
    const spillNote = spilledToAgent ? ' (long context partitioned into agent systemInstruction + STDIN)' : '';
    console.log(`${TAG} ${resolved.agyModel}${effortNote} via agy${allowApiKey ? ' (agy API-key mode)' : ''}${spillNote}`);

    const result = await runAgy({
        prompt,
        agentSections,
        model: resolved.agyModel,
        effort: resolved.agyEffort,
        allowApiKey,
        stream: writer.stream !== false,
        signal,
        onText: (delta) => writer.pushText(delta),
        onReasoning: settings.showReasoning ? (delta) => writer.pushReasoning(delta) : undefined,
    });
    if (signal?.aborted) throw abortError();
    writer.flushTail();
    writer.finish({ usage: mapAgyUsage(result?.usage) });
}

async function runViaDirectApi({ messages, model, settings, writer, signal, creds }) {
    const resolved = resolveGeminiModel(model, settings.gemini.effort, undefined, { forAgy: false });
    const extraBody = {};
    if (resolved.effort) extraBody.reasoning_effort = resolved.effort;
    if (settings.maxTokens) extraBody.max_tokens = settings.maxTokens;
    if (settings.temperature !== undefined) extraBody.temperature = settings.temperature;
    if (settings.topP !== undefined) extraBody.top_p = settings.topP;
    if (settings.stops.length) extraBody.stop = settings.stops.slice(0, 4);
    let host = creds.baseUrl;
    try { host = new URL(creds.baseUrl).host; } catch { /* not a URL */ }
    // Google's endpoint knows bare ids only; other endpoints get the id as routed (google/… kept).
    const apiModel = host === GOOGLE_HOST ? agyModelId(resolved.apiModel) : resolved.apiModel;
    console.log(`${TAG} ${apiModel} via direct API (${creds.source} → ${host}${resolved.effort ? `, effort ${resolved.effort}` : ''})`);
    await runOpenAiCompat({ baseUrl: creds.baseUrl, apiKey: creds.key, model: apiModel, messages, extraBody, writer, signal, tag: TAG });
}

/**
 * @param {object} args
 * @param {import('express').Request} args.req
 * @param {object[]} args.messages
 * @param {string} args.model requested model id
 * @param {object} args.settings from extractRequestSettings()
 * @param {import('../common/completion.js').CompletionWriter} args.writer
 * @param {AbortSignal} [args.signal] aborted when the client disconnects or the plugin stops
 */
export async function runGeminiChat({ req, messages, model, settings, writer, signal }) {
    const backend = settings.gemini.backend;
    const cliAvailable = !!agyLaunchSpec();
    try {
        if (backend === 'api') {
            const creds = discoverGeminiApiCredentials(req);
            if (!creds) throw Object.assign(new Error('Gemini backend "API" selected but no key found: set GEMINI_API_KEY (plus GOOGLE_GEMINI_BASE_URL for a compatible endpoint), or paste the key into SillyTavern\'s Custom API key field (plus ST_SUBSCRIPTIONS_GEMINI_BASE_URL for a non-Google endpoint).'), { httpStatus: 400 });
            return await runViaDirectApi({ messages, model, settings, writer, signal, creds });
        }
        if (backend === 'subscription') {
            if (!cliAvailable) throw Object.assign(new Error('Antigravity CLI (agy) not found: install it from https://antigravity.google/cli and sign in by running `agy` once, or switch the Gemini backend to "API".'), { httpStatus: 400 });
            const billing = agyBilling();
            if (billing.subscriptionBlocked) throw Object.assign(new Error(billing.reason), { httpStatus: 400 });
            return await runViaAgy({ messages, model, settings, writer, signal, allowApiKey: false });
        }
        // auto: agy as it is configured; the API backend only without agy.
        if (cliAvailable) return await runViaAgy({ messages, model, settings, writer, signal, allowApiKey: agyBilling().apiKeyMode });
        const creds = discoverGeminiApiCredentials(req);
        if (creds) return await runViaDirectApi({ messages, model, settings, writer, signal, creds });
        throw Object.assign(new Error('Neither the Antigravity CLI (agy) nor a Gemini API key is available. Install agy and sign in, or set GEMINI_API_KEY.'), { httpStatus: 400 });
    } catch (err) {
        // Client gone or plugin stopping: nothing to send; the listener ends the response.
        if (signal?.aborted) return;
        console.error(`${TAG} request failed:`, err instanceof Error ? err.message : err);
        writer.fail(err, { status: err?.httpStatus ?? 500, type: errorType(err) });
    }
}
