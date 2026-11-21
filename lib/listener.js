// ──────────────────────────────────────────────
// Standalone HTTP listener (separate from SillyTavern's Express app)
// ──────────────────────────────────────────────
//
// SillyTavern wraps its entire Express app in CSRF protection. When its
// chat-completions backend issues a server-side fetch to the "Custom
// Endpoint" URL, that loopback request carries no CSRF token and is refused
// — so the OpenAI-compatible surface lives on its own port, outside ST's
// middleware stack.
//
// Routes (all also available under /claude, /codex and /gemini prefixes,
// which filter the model list to one provider):
//   GET  /status                 aggregate health (?deep=1 → account reads)
//   GET  /v1/models              unified model list
//   GET  /v1/usage/quota         Claude windows + Codex rate limits
//   POST /v1/chat/completions    chat (SSE + JSON) — routed by model id
//   POST /v1/embeddings          always 501
//
// Access control. The listener spends the user's subscriptions, so:
//   • Bound to loopback (the default), requests must carry a loopback Host
//     header (localhost / 127.0.0.1 / [::1], plus ST_SUBSCRIPTIONS_ALLOWED_HOSTS)
//     — this defeats DNS-rebinding pages that resolve their own hostname to
//     127.0.0.1.
//   • Bound to anything else, ST_SUBSCRIPTIONS_TOKEN is mandatory and every
//     request must send it as the `X-Subscriptions-Token` header (add it under
//     "Include Headers" in SillyTavern's Custom connection). Without a token
//     the listener refuses to start.
//
// Cancellation: one AbortController per chat request, aborted when the
// client disconnects (the response's 'close' before it finished) or the
// plugin shuts down. Runners get its signal — see lib/common/abort.js.
//
// GET routes carry loopback-only CORS so the companion UI extension can read
// them directly when SillyTavern is browsed on the same machine.

import express from 'express';
import { timingSafeEqual } from 'node:crypto';

import { extractRequestSettings } from './settings.js';
import { resolveProvider, PROVIDERS, PROVIDER_LABELS } from './router.js';
import { CompletionWriter } from './common/completion.js';
import { partitionConversation } from './common/messages.js';
import { unifiedModelList } from './models.js';
import { aggregateStatus, setListenerInfo } from './status.js';
import { runClaudeChat } from './claude/chat.js';
import { runCodexChat } from './codex/chat.js';
import { runGeminiChat } from './gemini/chat.js';
import { claudeQuota } from './claude/oauth.js';
import { codexRateLimits } from './codex/status.js';
import { claudeStatus } from './claude/status.js';
import { codexStatus } from './codex/status.js';
import { geminiStatus } from './gemini/status.js';
import { isAbortError } from './common/abort.js';

const TAG = '[subscriptions]';
const SHUTDOWN_GRACE_MS = 2000;
const MODEL_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:/@[\]-]{0,127}$/;
const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '::1']);

let serverInstance = null;
const inflight = new Set();

function allowCorsGet(req, res, next) {
    const origin = req.headers.origin;
    if (origin && /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/i.test(origin)) {
        res.setHeader('Access-Control-Allow-Origin', origin);
        res.setHeader('Vary', 'Origin');
    }
    res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    next();
}

/** Wrap an async route so a rejection becomes a JSON 500 instead of an unhandled rejection (Express 4). */
function safe(fn) {
    return (req, res, next) => {
        Promise.resolve(fn(req, res, next)).catch((err) => {
            console.error(`${TAG} route ${req.method} ${req.path} failed:`, err);
            if (res.headersSent) { try { res.end(); } catch { /* socket gone */ } return; }
            res.status(500).json({ error: { message: err instanceof Error ? err.message : String(err), type: 'server_error' } });
        });
    };
}

const isDeep = (req) => /^(1|true|yes)$/i.test(String(req.query?.deep ?? ''));

const RUNNERS = { claude: runClaudeChat, codex: runCodexChat, gemini: runGeminiChat };
const STATUS = { claude: claudeStatus, codex: (o) => codexStatus(o), gemini: geminiStatus };

// ── Access control ──

export function isLoopbackHost(host) {
    const h = String(host ?? '').trim().toLowerCase().replace(/^\[|\]$/g, '');
    return LOOPBACK_HOSTS.has(h) || /^127(?:\.\d{1,3}){3}$/.test(h);
}

/** Hostname part of a Host header (handles [v6]:port). */
function hostHeaderName(value) {
    const v = String(value ?? '').trim().toLowerCase();
    if (!v) return '';
    if (v.startsWith('[')) return v.slice(1, v.indexOf(']') > 0 ? v.indexOf(']') : undefined);
    const colon = v.lastIndexOf(':');
    return colon > -1 && v.indexOf(':') === colon ? v.slice(0, colon) : v;
}

function allowedExtraHosts() {
    return String(process.env.ST_SUBSCRIPTIONS_ALLOWED_HOSTS ?? '')
        .split(',').map((s) => s.trim().toLowerCase()).filter(Boolean);
}

function tokenMatches(sent, expected) {
    const a = Buffer.from(String(sent ?? ''));
    const b = Buffer.from(String(expected ?? ''));
    return a.length === b.length && a.length > 0 && timingSafeEqual(a, b);
}

/**
 * @param {{ bindHost: string, token?: string }} opts
 */
export function accessGuard({ bindHost, token }) {
    const loopbackBind = isLoopbackHost(bindHost);
    const extra = allowedExtraHosts();
    return (req, res, next) => {
        if (req.method === 'OPTIONS') return next();
        if (loopbackBind) {
            const name = hostHeaderName(req.headers.host);
            if (name && !isLoopbackHost(name) && !extra.includes(name)) {
                return res.status(403).json({ error: { message: `Host "${name}" is not allowed (loopback only; add it to ST_SUBSCRIPTIONS_ALLOWED_HOSTS if intended).`, type: 'forbidden' } });
            }
            return next();
        }
        if (!tokenMatches(req.headers['x-subscriptions-token'], token)) {
            return res.status(401).json({ error: { message: 'Missing or wrong X-Subscriptions-Token header.', type: 'unauthorized' } });
        }
        return next();
    };
}

// ── Chat ──

/** Shared chat handler; `pathProvider` comes from a /claude|codex|gemini prefix. */
export async function handleChatCompletions(req, res, pathProvider = null) {
    const body = req.body || {};
    const messages = body.messages;
    const model = typeof body.model === 'string' ? body.model.trim() : '';
    if (!Array.isArray(messages) || messages.length === 0 || !model) {
        return res.status(400).json({ error: { message: 'messages[] (non-empty) and model are required', type: 'invalid_request_error' } });
    }
    if (!MODEL_ID_RE.test(model)) {
        return res.status(400).json({ error: { message: `Invalid model id ${JSON.stringify(model.slice(0, 80))}`, type: 'invalid_request_error' } });
    }

    const settings = extractRequestSettings(body);
    const { provider, reason } = resolveProvider({ model, pathProvider, settingsProvider: settings.provider });
    if (!provider) {
        return res.status(400).json({
            error: {
                message: `Cannot tell which subscription serves model "${model}". Use a model id from /v1/models ` +
                    '(claude-*, gpt-*/o*, gemini-*), connect through a provider prefix (/claude/v1, /codex/v1, /gemini/v1), ' +
                    'or send subscriptions.provider in the request body.',
                type: 'invalid_request_error',
            },
        });
    }

    // One cancellation signal for the whole request: client gone or plugin stopping.
    const ac = new AbortController();
    const onResClose = () => { if (!res.writableFinished) ac.abort(); };
    res.on('close', onResClose);
    if (res.destroyed || req.socket?.destroyed) ac.abort();
    inflight.add(ac);

    const writer = new CompletionWriter({
        res,
        stream: body.stream === true,
        model,
        stops: settings.stops,
        showReasoning: settings.showReasoning,
        idPrefix: `chatcmpl-${provider}`,
    });
    const { prefill } = partitionConversation(messages);
    if (prefill) writer.setPrefill(prefill);
    if (writer.stream && provider !== 'gemini' && typeof writer.startHeartbeat === 'function') writer.startHeartbeat();
    if (reason !== 'model') console.log(`${TAG} provider ${provider} chosen by ${reason} for model ${model}`);

    try {
        await RUNNERS[provider]({ req, res, messages, model, settings, writer, signal: ac.signal });
    } catch (err) {
        if (ac.signal.aborted && isAbortError(err)) {
            // Client left or plugin stopping — nothing useful to send.
        } else {
            console.error(`${TAG} unhandled ${provider} error:`, err);
        }
        try { writer.fail(err instanceof Error ? err : new Error(String(err)), { status: err?.httpStatus ?? 500 }); } catch { /* socket gone */ }
    } finally {
        res.off('close', onResClose);
        inflight.delete(ac);
        // A runner that returned without ending the response (client gone) must not leak the socket.
        if (!res.writableEnded && ac.signal.aborted) { try { res.end(); } catch { /* ignore */ } }
    }
}

export function rejectEmbeddings(_req, res) {
    return res.status(501).json({
        error: {
            message: 'The Subscriptions proxy does not support embeddings. Configure a separate embedding source (OpenAI, Google, or local).',
            type: 'not_supported',
        },
    });
}

export async function handleStatus(req, res) {
    res.json(await aggregateStatus({ deep: isDeep(req) }));
}

export async function handleQuota(req, res) {
    const [claude, codex] = await Promise.all([
        claudeQuota().catch((e) => ({ ok: false, message: String(e?.message ?? e) })),
        codexRateLimits({ startServer: isDeep(req) }).catch(() => null),
    ]);
    res.json({ ok: true, claude, codex, gemini: null, fetchedAt: Date.now() });
}

export async function handleModels(req, res, only = null) {
    try {
        const list = await unifiedModelList({ only });
        res.json(list);
    } catch (err) {
        res.status(500).json({ error: { message: err instanceof Error ? err.message : String(err), type: 'server_error' } });
    }
}

/** Final error handler: OpenAI-shaped JSON, never Express's HTML page. */
function jsonErrors(err, req, res, _next) {
    if (res.headersSent) { try { res.end(); } catch { /* ignore */ } return; }
    let status = err?.status || err?.statusCode || 500;
    let message = err instanceof Error ? err.message : String(err);
    if (err?.type === 'entity.too.large') { status = 413; message = 'Request body too large.'; }
    else if (err?.type === 'entity.parse.failed') { status = 400; message = 'Request body is not valid JSON.'; }
    if (status >= 500) console.error(`${TAG} ${req.method} ${req.path} failed:`, err);
    res.status(status).json({ error: { message, type: status >= 500 ? 'server_error' : 'invalid_request_error' } });
}

/**
 * @param {{ bindHost?: string, token?: string }} [opts]
 */
export function buildApp({ bindHost = '127.0.0.1', token } = {}) {
    const app = express();
    app.disable('x-powered-by');
    app.use(accessGuard({ bindHost, token }));
    app.use(express.json({ limit: '100mb' }));

    app.get('/', allowCorsGet, (_req, res) => res.json({
        plugin: 'subscriptions',
        providers: PROVIDERS.map((p) => ({ id: p, label: PROVIDER_LABELS[p], endpoint: `/${p}/v1` })),
        endpoints: ['/status', '/v1/models', '/v1/usage/quota', '/v1/chat/completions'],
    }));
    app.get('/status', allowCorsGet, safe(handleStatus));
    app.get('/v1/models', allowCorsGet, safe((req, res) => handleModels(req, res, null)));
    app.get('/v1/usage/quota', allowCorsGet, safe(handleQuota));
    app.options(['/status', '/v1/models', '/v1/usage/quota', '/:provider(claude|codex|gemini)/status', '/:provider(claude|codex|gemini)/v1/models'], allowCorsGet, (_req, res) => res.sendStatus(204));
    app.post('/v1/chat/completions', safe((req, res) => handleChatCompletions(req, res, null)));
    app.post('/v1/embeddings', rejectEmbeddings);

    // Per-provider prefixes
    app.get('/:provider(claude|codex|gemini)/status', allowCorsGet, safe(async (req, res) => {
        res.json(await STATUS[req.params.provider]({ deep: isDeep(req) }));
    }));
    app.get('/:provider(claude|codex|gemini)/v1/models', allowCorsGet, safe((req, res) => handleModels(req, res, req.params.provider)));
    app.post('/:provider(claude|codex|gemini)/v1/chat/completions', safe((req, res) => handleChatCompletions(req, res, req.params.provider)));
    app.post('/:provider(claude|codex|gemini)/v1/embeddings', rejectEmbeddings);

    app.use(jsonErrors);
    return app;
}

export function startStandaloneListener({ port, host }) {
    if (serverInstance) return Promise.resolve(serverInstance);
    const token = String(process.env.ST_SUBSCRIPTIONS_TOKEN ?? '').trim();
    if (!isLoopbackHost(host) && !token) {
        const err = new Error(
            `refusing to listen on ${host}: a non-loopback listener would let anyone who can reach it spend your subscriptions. ` +
            'Set ST_SUBSCRIPTIONS_TOKEN (and send it as the X-Subscriptions-Token header via "Include Headers"), or keep ST_SUBSCRIPTIONS_HOST at 127.0.0.1.',
        );
        console.error(`${TAG} ${err.message}`);
        return Promise.reject(err);
    }
    const app = buildApp({ bindHost: host, token });
    return new Promise((resolve, reject) => {
        const server = app.listen(port, host, () => {
            serverInstance = server;
            setListenerInfo({ host, port, running: true });
            console.log(`${TAG} standalone listener: http://${host}:${port}/v1 (per-provider: /claude/v1, /codex/v1, /gemini/v1)`);
            resolve(server);
        });
        server.on('error', (err) => {
            if (err && err.code === 'EADDRINUSE') {
                console.error(
                    `${TAG} port ${port} is already in use. If the old SillyTavern-ClaudeSubscription plugin is still installed in plugins/, ` +
                    'remove it (this plugin replaces it). Otherwise set ST_SUBSCRIPTIONS_PORT to a free port and update the Endpoint in the Subscriptions panel.',
                );
            } else {
                console.error(`${TAG} listener error:`, err);
            }
            reject(err);
        });
    });
}

/**
 * Stop accepting connections, cancel every in-flight generation (so CLI turns
 * stop spending quota), and resolve within a few seconds even when a client
 * keeps its socket open — SillyTavern's shutdown awaits this.
 */
export function stopStandaloneListener() {
    if (!serverInstance) return Promise.resolve();
    const server = serverInstance;
    serverInstance = null;
    for (const ac of inflight) ac.abort();
    return new Promise((resolve) => {
        let done = false;
        const finish = () => {
            if (done) return;
            done = true;
            setListenerInfo({ running: false });
            resolve();
        };
        server.close(finish);
        server.closeIdleConnections?.();
        const force = setTimeout(() => server.closeAllConnections?.(), SHUTDOWN_GRACE_MS);
        const hard = setTimeout(finish, SHUTDOWN_GRACE_MS + 1000);
        force.unref?.();
        hard.unref?.();
    });
}
