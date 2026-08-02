// Client-side "recently opened" tracker for the sidebar Recents section.
//
// Per-type lists in localStorage (no backend — per-browser). Each entry is the
// minimum the sidebar needs to render + navigate: { id, name, path }. Newest
// first, de-duped by id, capped. Recording fires a `gavel:recents` event so an
// open Sidebar refreshes immediately (localStorage 'storage' events don't fire
// in the same tab).

const TYPES = ['guardrail', 'rule', 'ce'];
const CAP = 3;   // sidebar shows the last 3 of each type
const key = (type) => `gavel_recents_${type}`;

export function getRecents(type) {
    try {
        const raw = localStorage.getItem(key(type));
        const list = raw ? JSON.parse(raw) : [];
        return Array.isArray(list) ? list : [];
    } catch {
        return [];
    }
}

export function recordRecent(type, item) {
    if (!TYPES.includes(type) || !item || item.id == null || !item.name) return;
    try {
        const entry = { id: item.id, name: String(item.name), path: item.path };
        const next = [entry, ...getRecents(type).filter(x => String(x.id) !== String(entry.id))].slice(0, CAP);
        localStorage.setItem(key(type), JSON.stringify(next));
        window.dispatchEvent(new CustomEvent('gavel:recents'));
    } catch {
        // ignore quota / serialization errors — recents is best-effort
    }
}

// Drop an entry from a type's list (e.g. after the underlying item is deleted),
// so the sidebar never links to something that 404s.
export function forgetRecent(type, id) {
    try {
        const next = getRecents(type).filter(x => String(x.id) !== String(id));
        localStorage.setItem(key(type), JSON.stringify(next));
        window.dispatchEvent(new CustomEvent('gavel:recents'));
    } catch {
        // ignore
    }
}
