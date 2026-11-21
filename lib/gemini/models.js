// ──────────────────────────────────────────────
// Gemini / Antigravity model catalog
// ──────────────────────────────────────────────
//
// Live from `agy models` (the same list the CLI's own picker shows), cached
// for an hour, with a static snapshot as fallback. A failed refresh keeps the
// previous list and backs off for five minutes; concurrent refreshes share
// one `agy models` process, and a caller that does not allow live fetches
// never replaces a live list with the snapshot.
//
// Antigravity encodes the reasoning effort in the model id
// (gemini-3.8-flash-high). The panel's effort setting overrides that suffix:
// the matching id when the live catalog lists it, otherwise the base id plus
// agy's own `--effort` flag, so agy validates the level itself.

import { agyLaunchSpec, agyListModels } from './agy.js';

const TAG = '[subscriptions/gemini]';
const CACHE_TTL_MS = 60 * 60 * 1000;
const FAILURE_BACKOFF_MS = 5 * 60 * 1000;
// agy model ids after the google/ and models/ prefixes are stripped.
const AGY_MODEL_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;

export const STATIC_GEMINI_MODELS = [
    { id: 'gemini-3.8-flash-high', name: 'Gemini 3.8 Flash (High)' },
    { id: 'gemini-3.8-flash-medium', name: 'Gemini 3.8 Flash (Medium)' },
    { id: 'gemini-3.8-flash-low', name: 'Gemini 3.8 Flash (Low)' },
    { id: 'gemini-3.7-flash-high', name: 'Gemini 3.7 Flash (High)' },
    { id: 'gemini-3.7-flash-medium', name: 'Gemini 3.7 Flash (Medium)' },
    { id: 'gemini-3.7-flash-low', name: 'Gemini 3.7 Flash (Low)' },
    { id: 'gemini-3.6-flash-high', name: 'Gemini 3.6 Flash (High)' },
    { id: 'gemini-3.6-flash-medium', name: 'Gemini 3.6 Flash (Medium)' },
    { id: 'gemini-3.6-flash-low', name: 'Gemini 3.6 Flash (Low)' },
    { id: 'gemini-3.1-pro-high', name: 'Gemini 3.1 Pro (High)' },
    { id: 'gemini-3.1-pro-low', name: 'Gemini 3.1 Pro (Low)' },
];

// at: last successful live fetch; failedAt: last failed attempt.
let cache = { at: 0, failedAt: 0, source: 'static', models: STATIC_GEMINI_MODELS };
let inflight = null;

async function refreshCatalog() {
    try {
        const models = await agyListModels();
        if (models?.length) {
            cache = { at: Date.now(), failedAt: 0, source: 'agy', models };
            return models;
        }
        console.warn(`${TAG} agy models returned nothing; keeping the ${cache.source} list`);
    } catch (err) {
        console.warn(`${TAG} live model list unavailable (keeping the ${cache.source} list): ${err instanceof Error ? err.message : err}`);
    }
    cache = { ...cache, failedAt: Date.now() };
    return cache.models;
}

/**
 * @param {object} [opts]
 * @param {boolean} [opts.live=true] allow an `agy models` refresh when the cache is stale
 * @param {boolean} [opts.force=false] refresh even when fresh (still deduplicated)
 */
export async function listGeminiModels({ live = true, force = false } = {}) {
    const now = Date.now();
    const fresh = (cache.source === 'agy' && now - cache.at < CACHE_TTL_MS) || (cache.failedAt && now - cache.failedAt < FAILURE_BACKOFF_MS);
    if (!force && fresh) return cache.models;
    if (!live || !agyLaunchSpec()) return cache.models; // never downgrade a live list
    inflight ??= refreshCatalog().finally(() => { inflight = null; });
    return inflight;
}

export function geminiCatalogSource() {
    return cache.source;
}

/** Snapshot of the catalog for model resolution: { source: 'agy'|'static', models }. */
export function geminiCatalog() {
    return { source: cache.source, models: cache.models };
}

/** Effort encoded in the id (gemini-3.8-flash-high → high). */
export function effortFromModelId(id) {
    const m = String(id ?? '').toLowerCase().match(/-(low|medium|high)$/);
    return m ? m[1] : null;
}

/** Strip the effort suffix (gemini-3.8-flash-high → gemini-3.8-flash). */
export function baseGeminiModel(id) {
    return String(id ?? '').replace(/-(low|medium|high)$/i, '');
}

/** The id agy understands: router-style google/ and models/ prefixes removed. */
export function agyModelId(id) {
    return String(id ?? '').trim().replace(/^(?:google\/|models\/)+/i, '');
}

function badModel(message) {
    return Object.assign(new Error(message), { httpStatus: 400 });
}

/**
 * Resolve a request's model + effort.
 *   agyModel / agyEffort  what runAgy passes as --model / --effort
 *   apiModel              the id for the direct API (prefixes kept, effort suffix removed)
 *   effort                the effective effort (for the API body and logs)
 * @param {string} requested model id from the request
 * @param {string} [effortOverride] low | medium | high from the panel / reasoning_effort
 * @param {{ source: string, models: { id: string }[] }} [catalog] geminiCatalog(); only a live ('agy') one is trusted for validation
 * @param {{ forAgy?: boolean }} [opts] forAgy:false (the direct API) skips the agy argv check — relay ids may contain '/'
 */
export function resolveGeminiModel(requested, effortOverride, catalog, { forAgy = true } = {}) {
    const sent = String(requested ?? '').trim();
    const raw = agyModelId(sent);
    const inherent = effortFromModelId(raw);
    const effort = inherent ?? effortOverride ?? undefined;
    const base = baseGeminiModel(raw);

    let agyModel = raw;
    let agyEffort;
    if (!inherent && effortOverride) {
        const wanted = `${base}-${effortOverride}`;
        const live = catalog?.source === 'agy' && Array.isArray(catalog.models) ? catalog.models.map((m) => String(m.id)) : null;
        const variants = live ? live.filter((id) => baseGeminiModel(id) === base && effortFromModelId(id)) : [];
        if (live?.includes(wanted)) {
            agyModel = wanted;
        } else if (variants.length) {
            throw badModel(`${base} has no "${effortOverride}" effort variant (available: ${variants.map(effortFromModelId).join(', ')}). Pick one of those in the Gemini reasoning-effort setting.`);
        } else {
            // Not in a live catalog: let agy resolve (and validate) the level.
            agyModel = base;
            agyEffort = effortOverride;
        }
    }
    if (forAgy && !AGY_MODEL_ID_RE.test(agyModel)) throw badModel(`Invalid Gemini model id ${JSON.stringify(sent.slice(0, 80))}.`);

    const apiModel = baseGeminiModel(sent) || sent;
    return { requested: sent, agyModel, agyEffort, apiModel, effort };
}

export async function geminiModelEntries(opts) {
    const models = await listGeminiModels(opts);
    return models.map((m) => ({
        id: m.id,
        object: 'model',
        created: 0,
        owned_by: 'google',
        display_name: m.name,
        context_window: /pro/i.test(m.id) ? 2000000 : 1000000,
        provider: 'gemini',
    }));
}
