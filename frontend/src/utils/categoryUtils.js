/**
 * Normalizes a category value to a clean string, or returns null.
 * Handles strings, bracket-wrapped strings, and {name, label, value} objects.
 */
export const normalizeCategoryValue = (value) => {
    if (!value) return null;
    if (typeof value === 'string') {
        const trimmed = value.trim();
        const cleaned = trimmed.replace(/^[{[]+/, '').replace(/[\]}]+$/, '').trim();
        return cleaned.length > 0 ? cleaned : null;
    }
    if (typeof value === 'object') {
        const candidate = value.name || value.label || value.value;
        if (typeof candidate === 'string') {
            const trimmed = candidate.trim();
            return trimmed.length > 0 ? trimmed : null;
        }
    }
    return null;
};
