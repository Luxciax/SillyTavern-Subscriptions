// ──────────────────────────────────────────────
// Google Antigravity CLI (`agy`) driver
// ──────────────────────────────────────────────
//
// Print mode with `--output-format stream-json` gives NDJSON events:
//   {"event":"init","init":{"model":…,"agent":…,"tools":[…]}}
//   {"event":"step_update","step_update":{"step_type":"agent_response","state":"ACTIVE","text_delta":"…"}}
//   {"event":"step_update","step_update":{"step_type":"tool","tool_name":"…","state":"ACTIVE"|"DONE"|"ERROR"}}
//   {"event":"step_update","step_update":{"step_type":"error_message","state":"DONE"}}
//   {"event":"result","result":{"status":"SUCCESS"|"ERROR","response":"…","error":"…","usage":{…}}}
// A failed turn also prints `AGY_ERROR: {"status","error_code","code_kind",…}`
// on stderr and exits 3. The prompt is piped on STDIN with
// `--input-format text` (no -p), which sidesteps the command-line length
// limit that would truncate long chats. agy never streams its thinking in
// this mode, so reasoning display is unavailable for Gemini through the CLI.
//
// Isolation. A plain agy turn is the full Antigravity coding agent: the
// user's saved tool permissions (often "always proceed"), global MCP servers,
// hooks, skills, plugins and GEMINI.md / AGENTS.md rules all apply, and the
// model can run commands and read files. Roleplay turns therefore run a
// plugin-owned custom agent — `.agents/agents/<CHAT_AGENT_NAME>/agent.md` in
// the plugin's scratch workspace — that inherits none of it:
//   inheritCustomizations: false    no user/workspace rules, skills, plugins, subagents, hooks
//   inheritMcp: false               no MCP servers
//   excludeDefaultComponents: true  no coding-agent prompt sections, no built-in tools
//   tools: []                       nothing added back
// Verified with agy 1.2.14 against a local mock Gemini endpoint: the model
// request then carries only this agent's system prompt and three inert tools
// (list_resources / read_resource, which answer "server … is not allowed in
// this context", and manage_task, which only sees the turn's own background
// tasks — there are none); user rules, skills, MCP tools and hooks are gone
// and run_command is rejected as an unknown tool. Checks that fail closed:
//   • agy < 1.2.11 ignores project agents in headless runs → refused;
//   • an --agent that does not resolve silently falls back to the default
//     agent, so every run writes a private --log-file that is read when the
//     `init` event arrives; a fallback kills the turn at once;
//   • any tool step other than the inert ones ends the turn (502).
// `init.tools` is no signal at all: agy lists the build's whole tool set
// there even for a tool-less agent.
//
// Billing. agy bills a Gemini API key (optionally through a custom base URL)
// whenever settings.json has `modelProvider`; the key and base URL come from
// agy's own process environment. The subscription backend refuses that
// configuration (see chat.js) and always runs agy with every key, base-URL
// and gateway variable removed, so it can only use the Google sign-in.
//
// agy also sends each prompt once more to a small model to title the
// conversation; there is no switch for that.

import { spawn, execFile } from 'node:child_process';
import { createInterface } from 'node:readline';
import { promisify } from 'node:util';
import { readFileSync, writeFileSync, existsSync, mkdirSync, rmSync, renameSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

import { resolveExecutable, commonBinDirs } from '../common/exec.js';
import { IS_WINDOWS, IS_TERMUX, termuxPrefix, envInt, envFlag } from '../common/platform.js';
import { runtimeDir } from '../common/runtime.js';
import { onAbort, abortError } from '../common/abort.js';

const TAG = '[subscriptions/gemini]';
const execFileAsync = promisify(execFile);

/** First agy release whose headless runs honour project agents (.agents/agents). */
export const MIN_ISOLATION_VERSION = '1.2.11';
export const CHAT_AGENT_NAME = 'st-subscriptions-chat';

/**
 * agy's TrajectoryChatConverter caps each USER_INPUT step at
 * MaxTokensPerUserInputStep = 48,000 tokens (192,000 UTF-8 bytes at agy's
 * 4-bytes/token estimator) and tail-truncates anything beyond byte 192,000 —
 * erasing the newest history turns and the current user message. Keep STDIN
 * safely under that cap and spill leading sections into the custom agent's
 * systemInstruction (which is exempt from MaxTokensPerUserInputStep).
 */
export const AGY_MAX_STDIN_BYTES = 180_000;

/**
 * agy's CascadePlannerConfig sets TruncationThresholdTokens = 192,000 tokens
 * (~768,000 UTF-8 bytes minus ~2,400 bytes for the inert tool declarations)
 * across systemInstruction + USER_INPUT combined. Keep the combined payload
 * safely below that ceiling so maxMessageTokens is always positive.
 */
export const AGY_MAX_TOTAL_BYTES = 720_000;

// Tools the isolated agent still gets and that cannot reach anything (verified, see header).
const INERT_TOOLS = new Set(['list_resources', 'read_resource', 'manage_task', 'finish']);
const KNOWN_STEP_TYPES = new Set(['user_input', 'agent_response', 'error_message', 'tool', 'unknown']);
const seenStepTypes = new Set();

/**
 * ParseAgentConfigFromURI in agy splits agent.md into top-level XML sections
 * on any line whose first non-blank token is a single `#` (and drops sections
 * with an empty body). Demote single-`#` markdown headings to `##` so embedded
 * prompts and transcripts stay inside their enclosing section verbatim.
 */
export function escapeAgentMarkdownSection(text) {
    return String(text ?? '').replace(/^([ \t]*)#(?=\s|$)/gm, '$1##');
}

/**
 * Build the markdown definition for the isolated roleplay agent, optionally
 * appending extra top-level `# <heading>` sections (e.g. spilled system
 * instructions and earlier conversation history).
 * @param {{ name?: string, sections?: Array<{ heading: string, content: string }> }} [opts]
 */
export function buildChatAgentMd({ name = CHAT_AGENT_NAME, sections = [] } = {}) {
    const lines = [
        '---',
        `name: ${name}`,
        'description: "SillyTavern Subscriptions: one text-only chat reply (no tools, no customizations)"',
        'mainAgent: true',
        'subagent: false',
        'hidden: true',
        'inheritCustomizations: false',
        'inheritMcp: false',
        'excludeDefaultComponents: true',
        'tools: []',
        '---',
        '# Role',
        'You are a conversational partner inside a chat application. Write the reply the conversation asks for, as plain prose in the requested style and voice.',
        '',
        '# Rules',
        'You have no tools. Do not try to run commands, read or write files, browse, or call any function; there is nothing to execute. Just write the reply.',
        '',
    ];
    for (const s of sections ?? []) {
        if (!s || typeof s.content !== 'string' || !s.content.trim()) continue;
        const heading = String(s.heading ?? 'Context').trim().replace(/^#+\s*/, '') || 'Context';
        lines.push(`# ${heading}`, escapeAgentMarkdownSection(s.content), '');
    }
    return lines.join('\n');
}

export const CHAT_AGENT_MD = buildChatAgentMd();

export function agyLaunchSpec() {
    const h = homedir();
    const extra = [];
    if (IS_WINDOWS) extra.push(join(process.env.LOCALAPPDATA || join(h, 'AppData', 'Local'), 'agy', 'bin'));
    extra.push(join(h, '.agy', 'bin'), join(h, '.local', 'bin'), ...commonBinDirs());
    if (IS_TERMUX) extra.unshift(join(termuxPrefix(), 'bin'));
    return resolveExecutable({
        envVars: ['ST_SUBSCRIPTIONS_AGY_PATH', 'AGY_PATH'],
        names: ['agy', 'agy.bin'],
        extraDirs: extra,
        npm: { pkg: 'agy', files: ['bin/agy.js', 'bin/agy'] },
    });
}

export function antigravitySettingsPath() {
    return join(homedir(), '.gemini', 'antigravity-cli', 'settings.json');
}

export function readAntigravitySettings() {
    try {
        const p = antigravitySettingsPath();
        if (!existsSync(p)) return null;
        return JSON.parse(readFileSync(p, 'utf8'));
    } catch {
        return null;
    }
}

const nonEmpty = (v) => (typeof v === 'string' && v.trim() ? v.trim() : null);

/**
 * How agy bills when the plugin runs it as configured.
 *   google-sign-in               no modelProvider — the Google account agy is signed in to
 *   gemini-api-key               modelProvider set — GEMINI_API_KEY, pay per token
 *   api-key via custom base URL  the same, sent to GOOGLE_GEMINI_BASE_URL
 * agy reads the key and base URL from its process environment (a settings.json
 * `env` block does not reach its own model calls — verified with 1.2.14).
 * @param {object|null} [settings] parsed settings.json
 * @param {Record<string,string|undefined>} [env]
 */
export function agyBilling(settings = readAntigravitySettings(), env = process.env) {
    const modelProvider = nonEmpty(settings?.modelProvider);
    const settingsBaseUrl = nonEmpty(settings?.env?.GOOGLE_GEMINI_BASE_URL);
    const envBaseUrl = nonEmpty(env.GOOGLE_GEMINI_BASE_URL);
    const apiKeyMode = !!modelProvider;
    // Only the process env reaches agy's model calls (the settings.json env
    // block does not), so a base URL that lives only in settings.json is not
    // where the key goes.
    const baseUrl = apiKeyMode ? envBaseUrl : null;
    const billing = !apiKeyMode ? 'google-sign-in' : (baseUrl ? 'api-key via custom base URL' : 'gemini-api-key');
    // A base URL in settings.json means agy was set up for key billing even
    // if modelProvider was dropped — the subscription backend stays out.
    const subscriptionBlocked = apiKeyMode || !!settingsBaseUrl;
    let reason = null;
    if (subscriptionBlocked) {
        const what = apiKeyMode
            ? `modelProvider "${modelProvider}"${baseUrl ? ' with a custom base URL' : ''}`
            : 'GOOGLE_GEMINI_BASE_URL in its env block';
        reason = `agy is configured to bill an API key (settings.json has ${what}), so the Gemini backend "Subscription only" will not run it. ` +
            `Switch the Gemini backend to "Auto" or "API" to allow pay-per-token billing, or remove modelProvider/GOOGLE_GEMINI_BASE_URL from ${antigravitySettingsPath()} to use your Google sign-in.`;
    }
    return { billing, apiKeyMode, modelProvider, baseUrl, settingsBaseUrl, subscriptionBlocked, reason };
}

// ── Subprocess environment ──

// Variables that switch agy (or the Google GenAI SDK inside it) to key,
// Cloud-project or gateway billing. Passed through only when the request
// allows key billing and agy's own config asks for it.
const GEMINI_BILLING_VARS = new Set([
    'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_GEMINI_BASE_URL',
    'GOOGLE_GENAI_USE_VERTEXAI', 'GOOGLE_GENAI_USE_ENTERPRISE', 'GOOGLE_APPLICATION_CREDENTIALS', 'AGY_ADC_AUTH',
]);
const GATEWAY_PREFIX = /^AGY_LLM_GATEWAY_/;
// Other providers' credentials and anything that looks like a secret.
const SECRET_PREFIX = /^(ANTHROPIC_|OPENAI_|CLAUDE|CODEX_|ST_SUBSCRIPTIONS_|AZURE_OPENAI)/;
const SECRET_NAME = /(^|_)(API_?KEY|ACCESS_?KEY|SECRET(_?ACCESS)?(_?KEY)?|PASSWORD|PASSWD|PASSPHRASE|TOKEN|PRIVATE_?KEY|CREDENTIALS?)$/;

/**
 * Environment for an agy subprocess: SillyTavern's env minus other
 * providers' keys and secrets; Gemini key/base-URL/gateway variables only
 * when `allowApiKey` (agy in API-key mode under an Auto/API backend).
 * Everything agy needs for its own sign-in, keyring and network (PATH, HOME,
 * USERPROFILE, APPDATA, TEMP, proxies, certificates, D-Bus, XDG_*) is kept.
 * @param {{ allowApiKey?: boolean, base?: Record<string,string|undefined> }} [opts]
 */
export function buildAgyEnv({ allowApiKey = false, base = process.env } = {}) {
    const env = {};
    for (const [k, v] of Object.entries(base)) {
        if (v === undefined) continue;
        const name = k.toUpperCase();
        if (GEMINI_BILLING_VARS.has(name) || GATEWAY_PREFIX.test(name)) {
            if (allowApiKey) env[k] = v;
            continue;
        }
        if (SECRET_PREFIX.test(name) || SECRET_NAME.test(name)) continue;
        env[k] = v;
    }
    // agy documents GEMINI_API_KEY; the SDK would silently prefer GOOGLE_API_KEY.
    if (allowApiKey) {
        const keys = Object.keys(env);
        const gemini = keys.find((k) => k.toUpperCase() === 'GEMINI_API_KEY');
        const google = keys.find((k) => k.toUpperCase() === 'GOOGLE_API_KEY');
        if (gemini && google) delete env[google];
    }
    env.NO_COLOR = '1';
    return env;
}

// ── Version ──

const versionCache = new Map();
export async function agyVersion(launch = agyLaunchSpec()) {
    if (!launch) return null;
    const hit = versionCache.get(launch.path);
    if (hit && Date.now() - hit.at < 5 * 60 * 1000) return hit.value;
    let value = null;
    try {
        const { stdout } = await execFileAsync(launch.command, [...launch.args, '--version'], { timeout: 8000, windowsHide: true, env: buildAgyEnv() });
        value = stdout.trim().split(/\r?\n/).pop() || null;
    } catch {
        value = null;
    }
    versionCache.set(launch.path, { at: Date.now(), value });
    return value;
}

/** Numeric compare of dotted versions ("1.2.14" vs "1.2.11"); null when unparsable. */
export function compareVersions(a, b) {
    const pa = String(a ?? '').match(/(\d+)\.(\d+)\.(\d+)/);
    const pb = String(b ?? '').match(/(\d+)\.(\d+)\.(\d+)/);
    if (!pa || !pb) return null;
    for (let i = 1; i <= 3; i++) {
        const d = Number(pa[i]) - Number(pb[i]);
        if (d) return d < 0 ? -1 : 1;
    }
    return 0;
}

/** Can this agy run the isolated chat agent? ok: true | false | null (version unknown). */
export function isolationSupport(version) {
    const cmp = compareVersions(version, MIN_ISOLATION_VERSION);
    if (cmp === null) return { ok: null, message: 'agy version unknown; each turn verifies the isolated chat agent loaded' };
    if (cmp < 0) {
        return {
            ok: false,
            message: `Antigravity CLI ${version} is too old for isolated roleplay turns (needs ${MIN_ISOLATION_VERSION}+; older builds silently run the full coding agent with your tools, MCP servers and rules). Run \`agy update\`.`,
        };
    }
    return { ok: true };
}

// ── Isolated workspace ──

export function chatWorkspace() {
    return runtimeDir('gemini-cwd');
}

/** Write (or repair) the plugin-owned chat agent definition; returns its path. */
export function ensureChatAgent(cwd = chatWorkspace()) {
    const dir = join(cwd, '.agents', 'agents', CHAT_AGENT_NAME);
    const file = join(dir, 'agent.md');
    const current = () => { try { return readFileSync(file, 'utf8'); } catch { return null; } };
    if (current() === CHAT_AGENT_MD) return file;
    try {
        mkdirSync(dir, { recursive: true });
        writeFileSync(file, CHAT_AGENT_MD);
    } catch (err) {
        // A concurrent request may be writing the same content.
        if (current() !== CHAT_AGENT_MD) throw new Error(`cannot write the isolated agy chat agent (${file}): ${err.message}`);
    }
    return file;
}

const STALE_TEMP_AGENT_MS = 30 * 60 * 1000;
function sweepStaleTempAgents(agentsRoot) {
    try {
        if (!existsSync(agentsRoot)) return;
        const prefix = `${CHAT_AGENT_NAME}-`;
        const cutoff = Date.now() - STALE_TEMP_AGENT_MS;
        for (const name of readdirSync(agentsRoot)) {
            if (!name.startsWith(prefix)) continue;
            const full = join(agentsRoot, name);
            try {
                if (statSync(full).mtimeMs < cutoff) rmSync(full, { recursive: true, force: true });
            } catch { /* already gone */ }
        }
    } catch { /* best-effort cleanup */ }
}

// ── Models ──

/** `agy models` → [{ id, name }] (network call; cached by the caller). */
export async function agyListModels(launch = agyLaunchSpec()) {
    if (!launch) return null;
    // Listing never bills; follow agy's own configuration so the list matches what chat will see.
    const env = buildAgyEnv({ allowApiKey: agyBilling().apiKeyMode });
    const { stdout } = await execFileAsync(launch.command, [...launch.args, 'models'], { timeout: 20000, windowsHide: true, env, cwd: chatWorkspace() });
    const out = [];
    for (const line of stdout.split(/\r?\n/)) {
        const m = line.match(/^\s*([a-z0-9][a-z0-9.\-_]*)\s+(.+?)\s*$/i);
        if (!m || /^fetching/i.test(m[1])) continue;
        out.push({ id: m[1], name: m[2] });
    }
    return out.length ? out : null;
}

// ── Errors ──

const GRPC_HTTP = {
    INVALID_ARGUMENT: 400, FAILED_PRECONDITION: 400, OUT_OF_RANGE: 400, UNAUTHENTICATED: 401,
    PERMISSION_DENIED: 403, NOT_FOUND: 404, RESOURCE_EXHAUSTED: 429, UNAVAILABLE: 503, DEADLINE_EXCEEDED: 504,
};

/**
 * The `AGY_ERROR: {...}` line agy prints on stderr when a turn fails, or null.
 * Verified shape: { short_error, status, error_code, code_kind: 'http'|'grpc', retryable, error_id }.
 */
export function parseAgyError(stderr) {
    const m = String(stderr ?? '').match(/AGY_ERROR:\s*(\{.*\})\s*$/m);
    if (!m) return null;
    try {
        const e = JSON.parse(m[1]);
        return e && typeof e === 'object' ? e : null;
    } catch {
        return null;
    }
}

/** Map agy's error report to an HTTP status (undefined when unknown). */
export function agyErrorStatus(agyErr, text = '') {
    if (/blocked by content safety|safety filter|prohibited content/i.test(String(text))) return 422;
    if (/invalid model selection|not recognized as a known model|has no "[^"]+" effort|conflicts with --effort/i.test(String(text))) return 400;
    // agy in API-key mode without GEMINI_API_KEY in its environment (exits 1, no AGY_ERROR).
    if (/modelProvider is set to .* environment variable is not set/i.test(String(text))) return 400;
    if (!agyErr) return undefined;
    const code = Number(agyErr.error_code ?? agyErr.http_code ?? agyErr.httpStatus);
    if (String(agyErr.code_kind ?? '').toLowerCase() === 'http' && code >= 400 && code <= 599) return code;
    return GRPC_HTTP[String(agyErr.status ?? '').toUpperCase()];
}

function agyFailure(message, { httpStatus, refusal = false, noRetry = false } = {}) {
    const err = new Error(message);
    if (httpStatus) err.httpStatus = httpStatus;
    if (refusal) { err.refusal = true; err.noRetry = true; }
    if (noRetry) err.noRetry = true;
    return err;
}

const tail = (s, n = 400) => String(s ?? '').trim().slice(-n);

/**
 * The last failed model call agy logged while it retries on its own, or null.
 * agy's stream only says `error_message`; the run log has the detail:
 *   run.go:395] Run: attempt 2 failed (Error 429, Message: …, Status: RESOURCE_EXHAUSTED, Details: []), retrying in 7.2s
 * @returns {{ attempt: number, code: number, message: string, status: string } | null}
 */
export function lastUpstreamFailure(logText) {
    const re = /attempt (\d+) failed \(Error (\d{3}), Message: (.*?), Status: ([A-Z_]+)/g;
    let last = null;
    for (const m of String(logText ?? '').matchAll(re)) last = m;
    if (!last) return null;
    return { attempt: Number(last[1]), code: Number(last[2]), message: last[3].slice(0, 200), status: last[4] };
}

/** How many failed attempts with 429 before the turn is ended as rate limited (agy's own backoff keeps doubling). */
const RATE_LIMIT_ATTEMPTS = 3;

// ── Process control ──

/** Kill the agy process we started and everything it spawned (language server, MCP servers). */
function killTree(proc) {
    if (!proc || proc.exitCode !== null || proc.signalCode !== null) return;
    if (IS_WINDOWS && proc.pid) {
        const taskkill = join(process.env.SystemRoot || process.env.windir || 'C:\\Windows', 'System32', 'taskkill.exe');
        execFile(taskkill, ['/PID', String(proc.pid), '/T', '/F'], { windowsHide: true }, (err) => {
            if (err) { try { proc.kill(); } catch { /* already gone */ } }
        });
        return;
    }
    try { proc.kill('SIGTERM'); } catch { /* already gone */ }
    const hard = setTimeout(() => {
        if (proc.exitCode === null && proc.signalCode === null) { try { proc.kill('SIGKILL'); } catch { /* gone */ } }
    }, 3000);
    hard.unref?.();
}

let runCounter = 0;
function runLogPath() {
    return join(runtimeDir('gemini-logs'), `agy-${process.pid}-${Date.now()}-${++runCounter}.log`);
}

/** Did the run log show the isolated agent loading? 'loaded' | 'fallback' | 'unknown'. */
export function agentLoadState(logText, agentName = CHAT_AGENT_NAME) {
    const text = String(logText ?? '');
    const escaped = agentName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (new RegExp(`Agent "${escaped}" not found`).test(text) || /Starting new conversation \(agent=false\)/.test(text)) return 'fallback';
    if (/Starting new conversation \(agent=true\)/.test(text)) return 'loaded';
    return 'unknown';
}

const warnedOnce = new Set();
function warnOnce(key, message) {
    if (warnedOnce.has(key)) return;
    warnedOnce.add(key);
    console.warn(message);
}

/**
 * Run one isolated print-mode turn.
 * @param {object} args
 * @param {string} args.prompt
 * @param {Array<{ heading: string, content: string }>} [args.agentSections] extra sections written into a per-run isolated agent definition (sent via systemInstruction)
 * @param {string} args.model agy model id (already validated)
 * @param {string} [args.effort] passed as --effort (only with an effort-less model id)
 * @param {boolean} [args.allowApiKey] keep Gemini key/base-URL variables (Auto/API backend with agy in API-key mode)
 * @param {(delta: string) => boolean} args.onText returns true to stop (stop string matched)
 * @param {(delta: string) => void} [args.onReasoning]
 * @param {AbortSignal} [args.signal] client gone / plugin stopping
 * @param {number} [args.timeoutMs] whole-turn deadline
 * @param {number} [args.firstOutputMs] no reply text yet and no progress for this long → fail
 * @param {number} [args.idleMs] reply text started, then silence for this long → fail
 * @returns {Promise<{ usage: object|null, response: string, status: string|null }>}
 */
export async function runAgy({
    prompt, agentSections, model, effort, allowApiKey = false, onText, onReasoning, signal,
    timeoutMs = envInt('ST_SUBSCRIPTIONS_AGY_TIMEOUT_MS', 10 * 60 * 1000),
    firstOutputMs = envInt('ST_SUBSCRIPTIONS_AGY_FIRST_OUTPUT_MS', 240 * 1000),
    idleMs = envInt('ST_SUBSCRIPTIONS_AGY_IDLE_MS', 120 * 1000),
}) {
    const launch = agyLaunchSpec();
    if (!launch) throw agyFailure('Antigravity CLI (agy) not found.', { httpStatus: 400 });
    if (signal?.aborted) throw abortError();

    const version = await agyVersion(launch);
    const support = isolationSupport(version);
    if (support.ok === false) throw agyFailure(support.message, { httpStatus: 400 });
    if (support.ok === null) warnOnce('agy-version', `${TAG} ${support.message}`);
    if (signal?.aborted) throw abortError();

    const cwd = chatWorkspace();
    ensureChatAgent(cwd);
    const hasExtraSections = Array.isArray(agentSections) && agentSections.some((s) => s && typeof s.content === 'string' && s.content.trim());
    let agentName = CHAT_AGENT_NAME;
    let agentFile = join(cwd, '.agents', 'agents', CHAT_AGENT_NAME, 'agent.md');
    let tempAgentDir = null;
    if (hasExtraSections) {
        const agentsRoot = join(cwd, '.agents', 'agents');
        sweepStaleTempAgents(agentsRoot);
        agentName = `${CHAT_AGENT_NAME}-${process.pid}-${Date.now()}-${++runCounter}`;
        tempAgentDir = join(agentsRoot, agentName);
        agentFile = join(tempAgentDir, 'agent.md');
        try {
            mkdirSync(tempAgentDir, { recursive: true });
            writeFileSync(agentFile, buildChatAgentMd({ name: agentName, sections: agentSections }));
        } catch (err) {
            throw agyFailure(`cannot write the isolated agy chat agent (${agentFile}): ${err.message}`, { httpStatus: 500 });
        }
    }
    const logFile = runLogPath();

    const args = [...launch.args, '--output-format', 'stream-json', '--input-format', 'text', '--disable-slash-commands', '--agent', agentName, '--log-file', logFile];
    if (envFlag('ST_SUBSCRIPTIONS_AGY_SANDBOX', true)) args.push('--sandbox');
    if (model) args.push('--model', model);
    if (effort) args.push('--effort', effort);

    return new Promise((resolve, reject) => {
        let proc;
        try {
            proc = spawn(launch.command, args, { stdio: ['pipe', 'pipe', 'pipe'], cwd, windowsHide: true, shell: false, env: buildAgyEnv({ allowApiKey }) });
        } catch (err) {
            if (tempAgentDir) { try { rmSync(tempAgentDir, { recursive: true, force: true }); } catch { /* ignore */ } }
            reject(agyFailure(`failed to start agy (${launch.path}): ${err.message}`, { httpStatus: 500 }));
            return;
        }
        let usage = null;
        let response = '';
        let status = null;
        let errorText = null;
        let stderr = '';
        let terminated = false;
        let settled = false;
        let failed = false;
        let textStarted = false;
        let textStep = null;
        let textStepDone = false;
        let postTextError = false;
        let upstreamErrors = 0;
        let upstream = null;      // last failed model call from the run log
        let emptyReply = false;   // an agent_response step finished without text
        let agentState = null;    // null (not checked yet) | 'loaded' | 'unknown'
        const stepText = new Map();
        const readLog = () => { try { return readFileSync(logFile, 'utf8'); } catch { return null; } };
        const upstreamNote = () => (upstream ? `; last upstream error: HTTP ${upstream.code} ${upstream.status}${upstream.message ? ` (${upstream.message})` : ''}` : '');

        let watchdog = null;
        const arm = () => {
            if (settled) return;
            if (watchdog) clearTimeout(watchdog);
            const ms = textStarted ? idleMs : firstOutputMs;
            watchdog = setTimeout(() => {
                terminate();
                const what = textStarted
                    ? `agy stopped streaming for ${Math.round(ms / 1000)}s mid-reply`
                    : `agy produced no reply within ${Math.round(ms / 1000)}s${upstreamErrors ? ` (${upstreamErrors} upstream error${upstreamErrors > 1 ? 's' : ''} while retrying${upstreamNote()})` : ''}`;
                const agyErr = parseAgyError(stderr);
                const upstreamStatus = !textStarted && upstream && upstream.code >= 400 ? upstream.code : undefined;
                fail(agyFailure(`${what}${stderr.trim() ? `: ${tail(stderr, 300)}` : ''}`, { httpStatus: agyErrorStatus(agyErr, stderr) ?? upstreamStatus ?? 504 }));
            }, ms);
            watchdog.unref?.();
        };
        const deadline = setTimeout(() => {
            terminate();
            fail(agyFailure(`agy did not finish within ${Math.round(timeoutMs / 1000)}s`, { httpStatus: 504 }));
        }, timeoutMs);
        deadline.unref?.();

        const finish = (err) => {
            if (settled) return;
            settled = true;
            if (watchdog) clearTimeout(watchdog);
            clearTimeout(deadline);
            offAbort();
            if (err) reject(err); else resolve({ usage, response, status });
        };
        const fail = (err) => { failed = true; finish(err); };
        const terminate = () => {
            if (terminated) return;
            terminated = true;
            killTree(proc);
        };
        const offAbort = onAbort(signal, () => { terminate(); finish(abortError()); });

        // Called at `init` and again before the first reply text is passed on
        // when the log could not confirm it yet (agy may flush it late).
        const checkAgentLoaded = ({ final = false } = {}) => {
            if (agentState === 'loaded') return true;
            const text = readLog();
            const state = text === null ? 'unknown' : agentLoadState(text, agentName);
            agentState = state === 'loaded' ? 'loaded' : 'unknown';
            if (state === 'fallback') {
                terminate();
                fail(agyFailure(
                    `agy did not load the plugin's isolated chat agent (${agentName}) and fell back to its full coding agent; the turn was stopped before it could use your tools, MCP servers or rules. ` +
                    `Check that ${agentFile} is readable and that agy is ${MIN_ISOLATION_VERSION}+ (\`agy update\`).`,
                    { httpStatus: 500, noRetry: true },
                ));
                return false;
            }
            if (state === 'unknown' && final) warnOnce('agy-log-format', `${TAG} could not confirm from agy's log that the isolated chat agent loaded (log format changed?); the tool-step guard stays active`);
            return true;
        };

        proc.on('error', (err) => fail(agyFailure(`failed to start agy (${launch.path}): ${err.message}`, { httpStatus: 500 })));
        proc.stderr.on('data', (d) => {
            const s = String(d);
            stderr = (stderr + s).slice(-4000);
            // Retry noise must not keep a stalled turn alive; anything else is progress.
            if (!/error|AGY_ERROR/i.test(s)) arm();
        });

        createInterface({ input: proc.stdout }).on('line', (line) => {
            if (settled) return;
            const trimmed = line.trim();
            if (!trimmed.startsWith('{')) return;
            let ev;
            try { ev = JSON.parse(trimmed); } catch { return; }

            if (ev.event === 'init') {
                arm();
                checkAgentLoaded();
                return;
            }

            if (ev.event === 'step_update' && ev.step_update) {
                const step = ev.step_update;
                const type = String(step.step_type ?? '');
                if (type && !KNOWN_STEP_TYPES.has(type) && !seenStepTypes.has(type)) {
                    seenStepTypes.add(type);
                    console.log(`${TAG} agy step type "${type}" seen (state ${step.state ?? '?'})`);
                }

                // Tools are disabled for roleplay: anything beyond the inert ones ends the turn.
                const toolName = step.tool_name ?? step.tool_info?.name ?? step.tool_call?.name;
                if (type === 'tool' || toolName || step.tool_info || step.tool_call) {
                    if (toolName && INERT_TOOLS.has(String(toolName))) {
                        warnOnce(`inert-${toolName}`, `${TAG} model called agy's inert ${toolName} tool (no effect in the isolated agent)`);
                        arm();
                        return;
                    }
                    terminate();
                    fail(agyFailure(`Gemini tried to use a tool (${toolName ?? type}) in a roleplay chat; the reply was stopped. Tools are disabled for roleplay.`, { httpStatus: 502, noRetry: true }));
                    return;
                }

                if (type === 'error_message') {
                    upstreamErrors++;
                    upstream = lastUpstreamFailure(readLog()) ?? upstream;
                    if (textStarted) {
                        // An error event alone does not identify which operation failed.
                        // Wait for the final result, but reject any restarted response
                        // below before its text can be appended to the first answer.
                        if (!postTextError) console.warn(`${TAG} error_message after response step ${textStep} (done=${textStepDone}); waiting for the final result`);
                        postTextError = true;
                        return;
                    }
                    if (emptyReply) {
                        // An empty model turn followed by an error is a blocked
                        // or refused reply; agy would re-ask several times.
                        terminate();
                        fail(agyFailure('Gemini returned no reply (blocked by a safety filter or refused). Refusals are not retried or sent to another model — reword, regenerate, or go back a message.', { httpStatus: 422, refusal: true }));
                        return;
                    }
                    // Failed model call; agy backs off and retries on its own
                    // (non-retryable ones end the run with AGY_ERROR). Persistent
                    // rate limiting is reported instead of waiting out the watchdog.
                    if (upstream?.code === 429 && upstream.attempt >= RATE_LIMIT_ATTEMPTS) {
                        terminate();
                        fail(agyFailure(`Gemini is rate limiting this account (HTTP 429 ${upstream.status}, ${upstream.attempt} attempts); try again later.`, { httpStatus: 429 }));
                    }
                    return; // otherwise bounded by the first-output watchdog
                }

                if (type === 'agent_response') {
                    const idx = step.step_index ?? -1;
                    if (postTextError && ((typeof step.text_delta === 'string' && step.text_delta) || (idx !== textStep && step.state === 'ACTIVE'))) {
                        // Treat resumed text as ambiguous even if agy reuses the index.
                        // A DONE status-only update for the original step is harmless.
                        terminate();
                        fail(agyFailure(`Gemini's reply was interrupted: agy started another response after an error; stopped before appending restarted text${upstreamNote()}.`, { httpStatus: 502, noRetry: true }));
                        return;
                    }
                    if (typeof step.text_delta === 'string' && step.text_delta) {
                        // Nothing reaches the client until the isolated agent is confirmed (or at least not refuted).
                        if (!textStarted && agentState !== 'loaded' && !checkAgentLoaded({ final: true })) return;
                        textStarted = true;
                        textStep = idx;
                        textStepDone = false;
                        emptyReply = false;
                        stepText.set(idx, true);
                        arm();
                        response += step.text_delta;
                        if (onText(step.text_delta)) { terminate(); finish(null); return; }
                    } else {
                        arm();
                    }
                    if (idx === textStep && step.state === 'DONE' && stepText.get(idx)) textStepDone = true;
                    if (step.state === 'DONE' && !stepText.get(idx) && !textStarted) emptyReply = true;
                } else if (type) {
                    arm();
                }
                if (onReasoning && typeof step.reasoning_delta === 'string' && step.reasoning_delta) onReasoning(step.reasoning_delta);
                if (onReasoning && typeof step.thinking_delta === 'string' && step.thinking_delta) onReasoning(step.thinking_delta);
                if (step.usage) usage = step.usage;
                return;
            }

            if (ev.event === 'result' && ev.result) {
                status = ev.result.status ?? null;
                if (ev.result.usage) usage = ev.result.usage;
                if (status && status !== 'SUCCESS') errorText = ev.result.error || `agy result status ${status}`;
                if (!response && typeof ev.result.response === 'string' && ev.result.response && status === 'SUCCESS') {
                    // Nothing streamed (should not happen): emit the final text.
                    if (agentState !== 'loaded' && !checkAgentLoaded({ final: true })) return;
                    response = ev.result.response;
                    textStarted = true;
                    onText(ev.result.response);
                }
            }
        });

        proc.stdin.on('error', () => { /* EPIPE when agy exits early; reported via close */ });
        proc.on('close', (code, sig) => {
            if (!settled) {
                const agyErr = parseAgyError(stderr);
                const detail = errorText ?? agyErr?.short_error ?? null;
                const httpStatus = agyErrorStatus(agyErr, `${detail ?? ''}\n${stderr}`);
                if (errorText || (postTextError && agyErr)) {
                    fail(agyFailure(`agy: ${detail ?? 'reported an error after response text'}`, { httpStatus: httpStatus ?? 502, refusal: httpStatus === 422 }));
                } else if (code !== 0) {
                    fail(agyFailure(`agy exited with ${code ?? sig}${response ? ' after partial output' : ''}${stderr.trim() ? `: ${tail(stderr)}` : ''}`, { httpStatus: httpStatus ?? 502, refusal: httpStatus === 422 }));
                } else if (/print timeout after .* returning partial output/i.test(stderr)) {
                    fail(agyFailure('agy hit its print timeout; the reply was truncated', { httpStatus: 504 }));
                } else if (postTextError && (status !== 'SUCCESS' || !textStepDone)) {
                    fail(agyFailure(`agy reported an error after text without both a completed response step and final SUCCESS${upstreamNote()}`, { httpStatus: httpStatus ?? upstream?.code ?? 502, refusal: httpStatus === 422 }));
                } else if (!response && status !== 'SUCCESS') {
                    fail(agyFailure(`agy ended without a reply${stderr.trim() ? `: ${tail(stderr)}` : ''}`, { httpStatus: 502 }));
                } else {
                    if (postTextError) console.warn(`${TAG} accepting completed response step ${textStep}: final SUCCESS after error_message`);
                    finish(null);
                }
            }
            if (tempAgentDir) {
                try { rmSync(tempAgentDir, { recursive: true, force: true }); } catch { /* still locked or already gone */ }
            }
            // The per-run log: kept (as agy-last-failure.log) only when the turn failed.
            try {
                if (failed && existsSync(logFile)) renameSync(logFile, join(runtimeDir('gemini-logs'), 'agy-last-failure.log'));
                else rmSync(logFile, { force: true });
            } catch { /* still locked or already gone */ }
        });

        arm();
        proc.stdin.write(prompt);
        proc.stdin.end();
    });
}

export { TAG as AGY_TAG };
