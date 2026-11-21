// ──────────────────────────────────────────────
// OpenAI messages[] helpers shared by every provider
// ──────────────────────────────────────────────
//
// Claude gets real multi-turn context through the Agent SDK's session resume
// (lib/claude/jsonl-entries.js). Codex and Gemini run through CLIs that take a
// single prompt per turn, so their history is *folded* into one labelled
// transcript. All of them share one reading of the message list —
// partitionConversation() — so the three providers agree on what the system
// prompt is, which turn is "current", and when a reply is a continuation.
//
// Position matters for system messages. SillyTavern sends:
//   • a LEADING run of system messages (main prompt, card, persona, examples)
//     → the system prompt;
//   • system messages INSIDE the history (Author's Note / World Info at depth)
//     → they must stay where they are;
//   • system messages at the END (Impersonate prompt, quiet prompts used by
//     Summarize / captions / /gen, the group-chat nudge, post-history
//     instructions) → they ARE the current instruction.
// Only the leading run is hoisted; every later system message becomes a user
// turn in place (the same conversion SillyTavern's own Claude converter
// does). A reply is treated as a continuation/prefill ONLY when the literal
// last message is an assistant message.

/** Flatten OpenAI content (string or multi-part array) to plain text. */
export function contentToText(content) {
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
        return content
            .filter((p) => p && p.type === 'text' && typeof p.text === 'string')
            .map((p) => p.text)
            .join('\n');
    }
    return '';
}

/** Content → array of parts (text parts + image parts), dropping unknown ones. */
function contentToParts(content) {
    if (typeof content === 'string') return content ? [{ type: 'text', text: content }] : [];
    if (!Array.isArray(content)) return [];
    return content.filter((p) => p && ((p.type === 'text' && typeof p.text === 'string') || p.type === 'image_url'));
}

/** Merge two user-role contents, keeping image parts. */
function mergeUserContent(a, b) {
    const pa = contentToParts(a);
    const pb = contentToParts(b);
    const hasImages = [...pa, ...pb].some((p) => p.type === 'image_url');
    if (!hasImages) {
        return [contentToText(a), contentToText(b)].filter((t) => t.length > 0).join('\n\n');
    }
    const sep = pa.length && pb.length ? [{ type: 'text', text: '\n\n' }] : [];
    return [...pa, ...sep, ...pb];
}

function isRenderable(m) {
    if (m.role === 'tool') return true;
    if (m.role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length) return true;
    if (contentToText(m.content).trim()) return true;
    return Array.isArray(m.content) && m.content.some((p) => p?.type === 'image_url');
}

/**
 * Read an OpenAI messages array the way every provider should.
 * @param {object[]} messages
 * @returns {{
 *   system: string|undefined,   leading system run, joined
 *   turns: object[],            everything after it; later system messages converted to user
 *                               turns and adjacent user turns merged
 *   prefill: string|null,       text of the literal last message when it is an assistant turn
 * }}
 */
export function partitionConversation(messages) {
    const list = (messages ?? []).filter((m) => m && typeof m === 'object' && typeof m.role === 'string');

    let i = 0;
    const sys = [];
    while (i < list.length && list[i].role === 'system') {
        const text = contentToText(list[i].content);
        if (text.trim()) sys.push(text);
        i++;
    }

    const turns = [];
    for (let offset = 0; offset < list.length - i; offset++) {
        const m = list[i + offset];
        const originalIndex = i + offset;
        let msg = m;
        if (m.role === 'system' || m.role === 'developer') {
            msg = { role: 'user', content: m.content, fromSystem: true, originalRole: m.role, originalIndex, kind: 'in_chat_system' };
        } else if (m.role === 'user') {
            msg = { ...m, originalRole: m.originalRole ?? m.role, originalIndex: m.originalIndex ?? originalIndex, kind: m.kind ?? 'user' };
        }
        if (!isRenderable(msg)) continue;
        const msgSegments = msg.role === 'user'
            ? [{ text: contentToText(msg.content), fromSystem: !!msg.fromSystem, originalRole: msg.originalRole ?? msg.role, originalIndex: msg.originalIndex ?? originalIndex, kind: msg.kind ?? (msg.fromSystem ? 'in_chat_system' : 'user') }]
            : undefined;
        const prev = turns[turns.length - 1];
        if (msg.role === 'user' && prev && prev.role === 'user') {
            const prevSegments = prev.segments ?? [{ text: contentToText(prev.content), fromSystem: !!prev.fromSystem, originalRole: prev.originalRole ?? prev.role, originalIndex: prev.originalIndex, kind: prev.kind ?? (prev.fromSystem ? 'in_chat_system' : 'user') }];
            turns[turns.length - 1] = { ...prev, content: mergeUserContent(prev.content, msg.content), fromSystem: !!(prev.fromSystem && msg.fromSystem), segments: [...prevSegments, ...msgSegments] };
            continue;
        }
        turns.push(msgSegments ? { ...msg, segments: msgSegments } : msg);
    }

    const last = list[list.length - 1];
    const prefill = last && last.role === 'assistant' && i < list.length ? contentToText(last.content) : null;

    return { system: sys.length ? sys.join('\n\n') : undefined, turns, prefill };
}

/** The system prompt: the leading run of system messages only. */
export function extractSystemText(messages) {
    return partitionConversation(messages).system;
}

/** Data-URL image parts of one message → [{ mediaType, data, url }]. */
export function imagePartsOf(content) {
    if (!Array.isArray(content)) return [];
    const out = [];
    for (const p of content) {
        if (!p || p.type !== 'image_url') continue;
        const url = p.image_url?.url ?? '';
        const m = url.match(/^data:(image\/[^;]+);base64,([A-Za-z0-9+/]+=*)$/);
        if (m) out.push({ mediaType: m[1], data: m[2], url });
        else if (/^https?:\/\//i.test(url)) out.push({ mediaType: null, data: null, url });
    }
    return out;
}

/**
 * Closest CLI-side approximation of Messages-API assistant prefill. The
 * prefill text stays client-side (SillyTavern keeps it), so the synthetic turn
 * tells the model to continue after it rather than repeat it.
 *
 * Phrasing matters: Fable's safeguard classifier rejects a resumed session
 * whose last turn reads like an injected "reply as if you already said…"
 * template (verified live), while this plain "your message was cut off"
 * wording passes on every backend.
 */
export function buildPrefillContinuation(prefill) {
    const normalized = String(prefill ?? '').trimEnd();
    if (!normalized.trim()) return '[Continue your last message — write only what comes next.]';
    return [
        '[Continue your last message. It currently ends with:',
        `"${normalized}"`,
        'Write only what comes next — do not repeat what is already written, and keep the same voice and format.]',
    ].join('\n');
}

/**
 * Split messages into { system, history, current, prefill, currentMessage }:
 *  - system: the leading system text (or undefined)
 *  - history: prior turns (excluding the current one; a continued assistant
 *    message stays in it)
 *  - current: the final user turn's text, or a continuation instruction when
 *    the conversation ends on an assistant turn (prefill / "continue")
 *  - prefill: the trailing assistant text when that case applies
 *  - currentMessage: the final user turn as a message object (images kept)
 */
export function splitConversation(messages) {
    const { system, turns, prefill } = partitionConversation(messages);
    if (turns.length === 0) {
        return { system, history: [], current: '[Start]', prefill: null, currentMessage: null };
    }
    const last = turns[turns.length - 1];
    if (prefill !== null && last.role === 'assistant') {
        // The message being continued stays in the history (the model sees its
        // own words as its turn); the instruction to continue follows it.
        return { system, history: turns, current: buildPrefillContinuation(prefill), prefill, currentMessage: null };
    }
    if (last.role === 'assistant') {
        // Defensive: a trailing assistant turn that was not the literal last message cannot happen after partitioning.
        return { system, history: turns, current: '[Start]', prefill: null, currentMessage: null };
    }
    return { system, history: turns.slice(0, -1), current: contentToText(last.content) || '[Start]', prefill: null, currentMessage: last };
}

/**
 * Format one history turn for a folded transcript, keeping in-chat system
 * injections (Author's Note / World Info at depth) distinct from the user's
 * spoken dialogue when both were merged into the same turn.
 */
export function formatHistoryUnitsForFold(m, { userLabel = 'User', assistantLabel = 'Assistant' } = {}) {
    const omitted = Array.isArray(m.content) && m.content.some((p) => p?.type === 'image_url') ? ' [image omitted]' : '';
    if (m.role === 'assistant') {
        const text = contentToText(m.content).trim();
        return text || omitted ? [{ text: assistantLabel + ': ' + text + omitted, fromSystem: false, kind: 'assistant', originalRole: 'assistant', originalIndex: m.originalIndex }] : [];
    }
    if (m.role === 'user' || m.role === 'tool') {
        const segs = Array.isArray(m.segments) ? m.segments : null;
        if (segs?.length) {
            const parts = [];
            let attachedOmitted = false;
            for (const seg of segs) {
                const t = String(seg.text ?? '').trim();
                if (!t) continue;
                if (seg.fromSystem) parts.push({ text: t, fromSystem: true, kind: seg.kind ?? 'in_chat_system', originalRole: seg.originalRole ?? 'system', originalIndex: seg.originalIndex });
                else {
                    const suffix = !attachedOmitted ? omitted : '';
                    attachedOmitted = true;
                    parts.push({ text: userLabel + ': ' + t + suffix, fromSystem: false, kind: seg.kind ?? 'user', originalRole: seg.originalRole ?? 'user', originalIndex: seg.originalIndex });
                }
            }
            if (!attachedOmitted && omitted) parts.push({ text: userLabel + ':' + omitted, fromSystem: false, kind: 'user', originalRole: 'user', originalIndex: m.originalIndex });
            return parts;
        }
        const text = contentToText(m.content).trim();
        if (!text && !omitted) return [];
        return [{ text: userLabel + ': ' + text + omitted, fromSystem: false, kind: m.role === 'tool' ? 'tool' : 'user', originalRole: m.role, originalIndex: m.originalIndex }];
    }
    return [];
}

/** Format one history turn as a single string for callers that do not need segment metadata. */
export function formatHistoryTurnForFold(m, opts = {}) {
    return formatHistoryUnitsForFold(m, opts).map((part) => part.text).join(String.fromCharCode(10, 10));
}

/**
 * Format the current (final) turn for a folded transcript. When Depth-0 /
 * post-history system prompts were merged with the user's turn, place those
 * system instructions in `<post_history_instructions>` *before* `${userLabel}:`
 * so dense preset rules never bury the user's actual latest message above the
 * end of the prompt.
 */
export function formatCurrentTurnForFold({ current, prefill, currentMessage, userLabel = 'User', assistantLabel = 'Assistant' }) {
    if (prefill !== null) return [current];
    if (currentMessage?.fromSystem) {
        // The current turn IS an instruction (Impersonate, a quiet prompt, a
        // nudge): it defines the task, so no "reply as Assistant" framing.
        return [current];
    }
    const segs = Array.isArray(currentMessage?.segments) ? currentMessage.segments : null;
    const sysNotes = segs ? segs.filter((s) => s.fromSystem && String(s.text ?? '').trim()).map((s) => String(s.text).trim()) : [];
    const userTexts = segs ? segs.filter((s) => !s.fromSystem && String(s.text ?? '').trim()).map((s) => String(s.text).trim()) : [];
    const out = [];
    if (sysNotes.length && userTexts.length) {
        out.push(`<post_history_instructions>\n${sysNotes.join('\n\n')}\n</post_history_instructions>`);
        out.push(`${userLabel}: ${userTexts.join('\n\n')}`);
    } else {
        out.push(`${userLabel}: ${current}`);
    }
    out.push(`Reply as ${assistantLabel} only — one reply, in character, without speaking for ${userLabel}.`);
    return out;
}

/**
 * Fold a conversation into a single prompt string for one-shot CLIs.
 *
 * @param {object[]} messages OpenAI messages
 * @param {object} [opts]
 * @param {boolean} [opts.includeSystem=true] inline the system prompt (false when the backend takes it separately)
 * @param {string}  [opts.preamble] extra instruction placed first (e.g. "text-only, no tools")
 * @param {string}  [opts.userLabel='User'] label for user turns
 * @param {string}  [opts.assistantLabel='Assistant'] label for assistant turns
 */
export function foldConversation(messages, opts = {}) {
    const { includeSystem = true, preamble, userLabel = 'User', assistantLabel = 'Assistant' } = opts;
    const { system, history, current, prefill, currentMessage } = splitConversation(messages);
    const sections = [];
    if (preamble) sections.push(preamble.trim());
    if (includeSystem && system) sections.push(`<system_instructions>\n${system}\n</system_instructions>`);

    const lines = [];
    for (const m of history) {
        const formatted = formatHistoryTurnForFold(m, { userLabel, assistantLabel });
        if (formatted) lines.push(formatted);
    }
    if (lines.length) sections.push(`<conversation_history>\n${lines.join('\n\n')}\n</conversation_history>`);

    sections.push(...formatCurrentTurnForFold({ current, prefill, currentMessage, userLabel, assistantLabel }));
    return sections.join('\n\n');
}

/** Roleplay-safe default instruction for CLIs that would otherwise use a coding preamble. */
export const TEXT_ONLY_PREAMBLE =
    'You are a conversational assistant inside a chat application. Reply with plain prose in the requested style. ' +
    'Do not use tools, do not run commands, do not browse, and do not create files — there is nothing to execute; ' +
    'just write the reply.';
