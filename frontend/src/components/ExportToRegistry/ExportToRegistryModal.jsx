import React from 'react';
import { FiAlertTriangle, FiDownload, FiExternalLink, FiFile } from 'react-icons/fi';
import GlassModal from '../GlassModal/GlassModal';
import {
    getRegistryExportPreflight,
    downloadRegistryExportFile,
    getExportSettings,
    saveExportSettings,
} from '../../api';

// Contributing to the public library means opening a pull request against the
// gavel-rules repo — the studio is read-only against it. This modal turns a
// local rule / CE / rule set into the exact files that repo expects, tells the
// user up front what would make its CI reject them, and then hands over the
// files plus a link to the right place on GitHub.
//
// A rule set is one small YAML file, so GitHub's new-file editor can be
// pre-filled with its contents and the whole contribution is a single click.
// Rules and CEs ship alongside dataset JSONs that are far too large for a URL,
// so those get downloaded and the link points at the upload page instead.
const REPO = 'Offensive-AI-Lab/gavel-rules';
const REPO_BRANCH = 'main';

// GitHub rejects over-long URLs; well under the ~8KB practical ceiling.
const PREFILL_LIMIT = 6000;

const LABELS = {
    rule: 'rule',
    ce: 'cognitive element',
    ruleset: 'rule set',
};

const fieldStyle = {
    width: '100%', padding: '8px 10px', borderRadius: 8,
    border: '1px solid rgba(148, 163, 184, 0.28)',
    background: 'rgba(15, 23, 42, 0.55)', color: '#e2e8f0', fontSize: '0.88rem',
};
const labelStyle = { display: 'block', fontSize: '0.78rem', color: '#94a3b8', marginBottom: 4 };
const rowStyle = { marginBottom: 14 };

export default function ExportToRegistryModal({ open, onClose, kind, entityId, defaultTitle = '' }) {
    const [title, setTitle] = React.useState(defaultTitle);
    const [description, setDescription] = React.useState('');
    const [role, setRole] = React.useState('');
    const [author, setAuthor] = React.useState('');
    const [authorLoaded, setAuthorLoaded] = React.useState(false);
    const [preflight, setPreflight] = React.useState(null);
    const [error, setError] = React.useState('');
    const [busy, setBusy] = React.useState(false);
    const [done, setDone] = React.useState(false);

    // The GitHub handle is per-operator identity, not a per-export choice, so it
    // is remembered server-side and only typed once.
    React.useEffect(() => {
        if (!open || authorLoaded) return;
        let alive = true;
        getExportSettings()
            .then((res) => { if (alive) setAuthor(res.data?.github_username || ''); })
            .catch(() => { /* falls back to an empty field */ })
            .finally(() => { if (alive) setAuthorLoaded(true); });
        return () => { alive = false; };
    }, [open, authorLoaded]);

    // Re-run preflight as the user edits: the warning list IS the feedback about
    // whether this export would survive the registry's CI.
    React.useEffect(() => {
        if (!open || !entityId || !authorLoaded) return;
        let alive = true;
        const params = { author };
        if (title) params.title = title;
        if (kind === 'ruleset') params.description = description;
        if (kind === 'ce' && role) params.role = role;

        const timer = setTimeout(() => {
            getRegistryExportPreflight(kind, entityId, params)
                .then((res) => {
                    if (!alive) return;
                    // A 200 that isn't a preflight payload means the request never
                    // reached the backend (dev-proxy gap → the SPA's index.html
                    // comes back instead). Say so rather than rendering an inert
                    // Download button with nothing behind it.
                    if (!Array.isArray(res.data?.files)) {
                        setPreflight(null);
                        setError('The export service did not respond as expected. '
                            + 'Check that the backend is running and reachable.');
                        return;
                    }
                    setPreflight(res.data);
                    setError('');
                    setRole((cur) => cur || res.data?.role || '');
                })
                .catch((e) => {
                    if (!alive) return;
                    setPreflight(null);
                    setError(e?.response?.data?.detail || 'Could not prepare this export.');
                });
        }, 250);
        return () => { alive = false; clearTimeout(timer); };
    }, [open, kind, entityId, author, authorLoaded, title, description, role]);

    // Memoized so the prefill-URL useMemo below isn't invalidated every render.
    const files = React.useMemo(() => preflight?.files || [], [preflight]);
    // errors block the download (the backend refuses them with 422 anyway);
    // warnings are advisory — e.g. "contribute the draft CE in the same PR".
    const blockers = preflight?.errors || [];
    const warnings = preflight?.warnings || [];

    // A pre-filled GitHub editor only works for a single small file.
    const prefillUrl = React.useMemo(() => {
        if (kind !== 'ruleset' || !preflight?.preview || files.length !== 1) return null;
        const encoded = encodeURIComponent(preflight.preview);
        if (encoded.length > PREFILL_LIMIT) return null;
        return `https://github.com/${REPO}/new/${REPO_BRANCH}`
            + `?filename=${encodeURIComponent(files[0].path)}&value=${encoded}`;
    }, [kind, preflight, files]);

    const uploadUrl = preflight
        ? `https://github.com/${REPO}/upload/${REPO_BRANCH}/${preflight.directory}`
        : null;

    const handleDownload = async () => {
        if (!files.length) return;
        setBusy(true);
        setError('');
        const params = { author };
        if (title) params.title = title;
        if (kind === 'ruleset') params.description = description;
        if (kind === 'ce' && role) params.role = role;
        try {
            // Sequential: browsers throttle (and prompt about) bursts of downloads.
            for (const file of files) {
                await downloadRegistryExportFile(kind, entityId, file.filename, params);
            }
            if (author) await saveExportSettings(author).catch(() => { /* best-effort */ });
            setDone(true);
        } catch (e) {
            setError(e?.response?.data?.detail || e?.message || 'Export failed.');
        } finally {
            setBusy(false);
        }
    };

    return (
        <GlassModal isOpen={open} onClose={onClose} title={`Export ${LABELS[kind] || 'artifact'}`}>
            <p style={{ margin: '0 0 16px', color: '#94a3b8', fontSize: '0.88rem' }}>
                Saves this {LABELS[kind]} in the format the public library uses. Nothing is
                published — you submit the files as a pull request to{' '}
                <code style={{ color: '#cbd5e1' }}>{REPO}</code>.
            </p>

            <div style={rowStyle}>
                <label style={labelStyle} htmlFor="export-title">Title</label>
                <input
                    id="export-title" style={fieldStyle} value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="A one-line display name"
                />
            </div>

            {kind === 'ruleset' && (
                <div style={rowStyle}>
                    <label style={labelStyle} htmlFor="export-description">Description</label>
                    <textarea
                        id="export-description" style={{ ...fieldStyle, minHeight: 60, resize: 'vertical' }}
                        value={description} onChange={(e) => setDescription(e.target.value)}
                        placeholder="What belongs in this set — the inclusion criterion"
                    />
                </div>
            )}

            {kind === 'ce' && (
                <div style={rowStyle}>
                    <label style={labelStyle} htmlFor="export-role">Role</label>
                    <select id="export-role" style={fieldStyle} value={role}
                        onChange={(e) => setRole(e.target.value)}>
                        <option value="">Choose a role…</option>
                        {(preflight?.roles || []).map((r) => (
                            <option key={r} value={r}>{r}</option>
                        ))}
                    </select>
                </div>
            )}

            <div style={rowStyle}>
                <label style={labelStyle} htmlFor="export-author">GitHub username</label>
                <input
                    id="export-author" style={fieldStyle} value={author}
                    onChange={(e) => setAuthor(e.target.value)}
                    placeholder="the account that will open the pull request"
                />
                <div style={{ fontSize: '0.74rem', color: '#64748b', marginTop: 4 }}>
                    Recorded as the artifact's author and remembered for next time.
                </div>
            </div>

            {blockers.length > 0 && (
                <div style={{
                    border: '1px solid rgba(248, 113, 113, 0.4)', background: 'rgba(248, 113, 113, 0.08)',
                    borderRadius: 10, padding: '10px 12px', marginBottom: 14,
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#f87171',
                        fontSize: '0.8rem', marginBottom: 6 }}>
                        <FiAlertTriangle size={14} /> The registry would reject this — fix before downloading
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 18, color: '#cbd5e1', fontSize: '0.8rem' }}>
                        {blockers.map((w) => <li key={w} style={{ marginBottom: 3 }}>{w}</li>)}
                    </ul>
                </div>
            )}

            {warnings.length > 0 && (
                <div style={{
                    border: '1px solid rgba(251, 191, 36, 0.35)', background: 'rgba(251, 191, 36, 0.08)',
                    borderRadius: 10, padding: '10px 12px', marginBottom: 14,
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#fbbf24',
                        fontSize: '0.8rem', marginBottom: 6 }}>
                        <FiAlertTriangle size={14} /> Needs attention before the pull request
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 18, color: '#cbd5e1', fontSize: '0.8rem' }}>
                        {warnings.map((w) => <li key={w} style={{ marginBottom: 3 }}>{w}</li>)}
                    </ul>
                </div>
            )}

            {files.length > 0 && (
                <div style={rowStyle}>
                    <div style={labelStyle}>Files ({files.length})</div>
                    <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
                        {files.map((f) => (
                            <li key={f.path} style={{ display: 'flex', alignItems: 'center', gap: 6,
                                color: '#cbd5e1', fontSize: '0.8rem', marginBottom: 3 }}>
                                <FiFile size={13} style={{ color: '#64748b' }} />
                                <code>{f.path}</code>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {error && (
                <div style={{ color: '#f87171', fontSize: '0.82rem', marginBottom: 12 }}>{error}</div>
            )}

            <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                <button
                    className="bookmark-btn" onClick={handleDownload}
                    disabled={busy || !files.length || blockers.length > 0}
                    aria-label={`Download ${LABELS[kind]} files`}
                >
                    <FiDownload /> {busy ? 'Preparing…' : `Download ${files.length || ''} file${files.length === 1 ? '' : 's'}`}
                </button>

                {blockers.length === 0 && (prefillUrl ? (
                    <a className="bookmark-btn" href={prefillUrl} target="_blank" rel="noreferrer"
                        style={{ textDecoration: 'none' }}>
                        <FiExternalLink /> Open a pull request on GitHub
                    </a>
                ) : uploadUrl && (
                    <a className="bookmark-btn" href={uploadUrl} target="_blank" rel="noreferrer"
                        style={{ textDecoration: 'none' }}>
                        <FiExternalLink /> Open GitHub to upload them
                    </a>
                ))}
            </div>

            {done && (
                <div style={{ marginTop: 12, color: '#4ade80', fontSize: '0.82rem' }}>
                    Saved. {kind === 'ruleset'
                        ? 'Use the GitHub link to submit it.'
                        : `On GitHub, upload them into ${preflight?.directory}.`}
                </div>
            )}
        </GlassModal>
    );
}
