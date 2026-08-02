// AddModelModal — register an LLM, either by Hugging Face repo/link or by
// uploading a local .zip. Extracted from the (removed) Models page so model
// management lives where it's needed: the guardrail's Choose-Model / train flow.
//
// Props:
//   isOpen   — controls visibility
//   onClose  — close the modal
//   userId   — owner of the new model
//   onAdded(model?) — called after a successful add (caller refetches models)
import { useState, useEffect } from 'react';
import ReactiveButton from '../ReactiveButton/ReactiveButton';
import GlassModal from '../GlassModal/GlassModal';
import { FiPlus, FiHardDrive, FiCloud, FiZap } from 'react-icons/fi';
import { showAlertDialog, showLoadingDialog } from '../ConfirmDialog/confirmDialog';
import api, { createModel } from '../../api';

// Built-in "demo" models — pick one to experiment with, no HF link needed.
// `layers` is the model's transformer-layer count (the layer-picker bound);
// `gated` ones require the user's HF token before they can be used.
const DEMO_MODELS = [
    { label: 'Llama3-8B', repo: 'meta-llama/Meta-Llama-3-8B-Instruct', gated: true, layers: 32 },
    { label: 'Mistral-7B', repo: 'mistralai/Mistral-7B-Instruct-v0.2', gated: false, layers: 32 },
    { label: 'Qwen3-8B', repo: 'Qwen/Qwen3-8B', gated: false, layers: 36 },
    { label: 'Gemma-4B', repo: 'google/gemma-3-4b-it', gated: true, layers: 34 },
];
// Default layer band — the middle of the network (e.g. [13, 27) of 32), where
// representations are most informative. The user can change it.
const defaultRange = (n) => [Math.max(0, Math.round(n * 0.4)), Math.min(n, Math.round(n * 0.84))];

const REPO_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/;

const parseHuggingFaceInput = (value) => {
    const input = (value || '').trim();
    if (!input) return { ok: false, error: 'Please enter a Hugging Face link or repo ID' };
    if (REPO_ID_PATTERN.test(input)) return { ok: true, repoId: input };
    try {
        const url = new URL(input);
        if (url.protocol !== 'https:') return { ok: false, error: 'Link must use https' };
        if (url.hostname.toLowerCase() !== 'huggingface.co') {
            return { ok: false, error: 'Link must be from https://huggingface.co' };
        }
        const segments = url.pathname.split('/').filter(Boolean);
        if (segments.length < 2) return { ok: false, error: 'Invalid Hugging Face model link' };
        let owner = segments[0], repo = segments[1];
        if (segments[0].toLowerCase() === 'models') {
            if (segments.length < 3) return { ok: false, error: 'Invalid Hugging Face model link' };
            owner = segments[1]; repo = segments[2];
        }
        const repoId = `${owner}/${repo}`;
        if (!REPO_ID_PATTERN.test(repoId)) return { ok: false, error: 'Invalid Hugging Face model link format' };
        return { ok: true, repoId };
    } catch {
        return { ok: false, error: 'Invalid link. Use a Hugging Face model URL' };
    }
};

const AddModelModal = ({ isOpen, onClose, userId, onAdded }) => {
    const [uploadMode, setUploadMode] = useState(null);
    const [modelName, setModelName] = useState('');
    const [hfPath, setHfPath] = useState('');
    const [hfToken, setHfToken] = useState('');
    const [selectedFile, setSelectedFile] = useState(null);
    // Demo-model selection. (Layers are configured later, in the rule setup.)
    const [selectedDemo, setSelectedDemo] = useState(null);

    // Reset to a clean state each time the modal opens.
    useEffect(() => {
        if (isOpen) {
            setUploadMode(null); setModelName(''); setHfPath(''); setHfToken(''); setSelectedFile(null);
            setSelectedDemo(null);
        }
    }, [isOpen]);

    const pickDemo = (m) => {
        setSelectedDemo(m);
        if (!m.gated) setHfToken('');
    };

    const submitDemo = async () => {
        const m = selectedDemo;
        if (!m) return;
        if (m.gated && !hfToken.trim()) {
            return showAlertDialog({ title: 'Token required', message: `${m.label} is a gated model — add your Hugging Face token to use it.`, variant: 'warning' });
        }
        // Seed a sensible default layer band; the user tunes it in the rule setup.
        const [start, end] = defaultRange(m.layers);
        onClose();
        const close = showLoadingDialog({ title: 'Adding model', message: 'Connecting to Hugging Face…' });
        try {
            await createModel(userId, m.label, m.repo, m.gated ? hfToken.trim() : null, m.layers, [start, end]);
            close();
            await showAlertDialog({ title: 'Success', message: 'Model added successfully!', variant: 'success' });
            onAdded?.();
        } catch (error) {
            close();
            await showAlertDialog({ title: 'Error', message: error?.response?.data?.detail || error.message || 'Operation failed', variant: 'error' });
        }
    };

    const handleSubmit = async () => {
        if (!modelName.trim()) return showAlertDialog({ title: 'Missing name', message: 'Please enter a model name', variant: 'warning' });
        if (uploadMode === 'local' && !selectedFile) {
            return showAlertDialog({ title: 'No file selected', message: 'Please select a file', variant: 'warning' });
        }

        let normalizedRepoId = '';
        if (uploadMode === 'hf') {
            const parsedHf = parseHuggingFaceInput(hfPath);
            if (!parsedHf.ok) return showAlertDialog({ title: 'Invalid Hugging Face URL', message: parsedHf.error, variant: 'warning' });
            normalizedRepoId = parsedHf.repoId;
        }

        onClose();
        const close = showLoadingDialog({
            title: 'Adding model',
            message: uploadMode === 'local'
                ? 'Uploading file… large models can take a few minutes.'
                : 'Connecting to Hugging Face...',
        });
        try {
            if (uploadMode === 'local') {
                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('user_id', userId);
                formData.append('name', modelName);
                await api.post('/models/upload', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    onUploadProgress: (e) => {
                        if (!e.total) return;
                        const pct = Math.round((e.loaded / e.total) * 100);
                        close.update?.(pct < 100
                            ? `Uploading… ${pct}% (large models can take a few minutes)`
                            : 'Upload complete — extracting & validating the model…');
                    },
                });
                window.dispatchEvent(new Event('gavel:libraryChanged'));
            } else {
                await createModel(userId, modelName, normalizedRepoId, hfToken.trim() || null);
            }
            close();
            await showAlertDialog({ title: 'Success', message: 'Model added successfully!', variant: 'success' });
            onAdded?.();
        } catch (error) {
            close();
            await showAlertDialog({
                title: 'Error',
                message: error?.response?.data?.detail || error.message || 'Operation failed',
                variant: 'error',
            });
        }
    };

    return (
        <GlassModal
            isOpen={isOpen}
            onClose={onClose}
            title={!uploadMode ? 'Add a model' : uploadMode === 'demo' ? 'Pick a demo model' : uploadMode === 'local' ? 'Upload Local File' : 'Connect Hugging Face'}
        >
            {!uploadMode ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                    <p style={{ color: '#94a3b8', marginBottom: '5px' }}>Choose how to add a model:</p>
                    <button onClick={() => setUploadMode('demo')} style={optionBtnStyle}>
                        <div style={{ ...iconBoxStyle, background: 'rgba(99, 102, 241, 0.18)', color: '#c7d2fe' }}><FiZap size={24} /></div>
                        <div style={{ textAlign: 'left' }}>
                            <span style={{ display: 'block', fontWeight: '600', color: '#f1f5f9' }}>Pick a demo model</span>
                            <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Ready-to-use LLMs — no link needed</span>
                        </div>
                    </button>
                    <button onClick={() => setUploadMode('local')} style={optionBtnStyle}>
                        <div style={{ ...iconBoxStyle, background: 'rgba(14, 165, 233, 0.18)', color: '#67e8f9' }}><FiHardDrive size={24} /></div>
                        <div style={{ textAlign: 'left' }}>
                            <span style={{ display: 'block', fontWeight: '600', color: '#f1f5f9' }}>Upload Local File</span>
                            <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>.zip of model directory</span>
                        </div>
                    </button>
                    <button onClick={() => setUploadMode('hf')} style={optionBtnStyle}>
                        <div style={{ ...iconBoxStyle, background: 'rgba(139, 92, 246, 0.18)', color: '#c4b5fd' }}><FiCloud size={24} /></div>
                        <div style={{ textAlign: 'left' }}>
                            <span style={{ display: 'block', fontWeight: '600', color: '#f1f5f9' }}>Hugging Face Link</span>
                            <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Connect via Repo ID</span>
                        </div>
                    </button>
                </div>
            ) : uploadMode === 'demo' ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                    <p style={{ color: '#94a3b8', margin: 0, fontSize: '0.88rem' }}>Pick a built-in model to experiment with — no link required.</p>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {DEMO_MODELS.map((m) => {
                            const sel = selectedDemo?.repo === m.repo;
                            return (
                                <button
                                    key={m.repo}
                                    onClick={() => pickDemo(m)}
                                    style={{ ...optionBtnStyle, gap: 10, border: sel ? '2px solid #818cf8' : '1px solid rgba(148, 163, 184, 0.18)', background: sel ? 'rgba(99, 102, 241, 0.18)' : 'rgba(15, 23, 42, 0.55)' }}
                                >
                                    <div style={{ textAlign: 'left', flex: 1 }}>
                                        <span style={{ display: 'block', fontWeight: 600, color: '#f1f5f9' }}>{m.label}</span>
                                        <span style={{ fontSize: '0.76rem', color: '#94a3b8' }}>{m.repo} · {m.layers} layers</span>
                                    </div>
                                    {m.gated && <span style={gatedBadgeStyle}>needs token</span>}
                                </button>
                            );
                        })}
                    </div>

                    {selectedDemo && (
                        <>
                            {selectedDemo.gated && (
                                <div>
                                    <label style={labelStyle}>Hugging Face Token <span style={{ opacity: 0.6, fontWeight: 400 }}>(required — {selectedDemo.label} is gated)</span></label>
                                    <input className="glass-input" type="password" placeholder="hf_..." value={hfToken} onChange={(e) => setHfToken(e.target.value)} maxLength={512} autoComplete="off" />
                                </div>
                            )}

                            <div style={{ display: 'flex', gap: '12px', marginTop: '4px' }}>
                                <button onClick={() => { setUploadMode(null); setSelectedDemo(null); }} style={modalBackBtnStyle}>Back</button>
                                <div style={{ flex: '1' }}>
                                    <ReactiveButton
                                        label={selectedDemo.gated && !hfToken.trim() ? 'Add a token first' : 'Add model'}
                                        onClick={submitDemo}
                                        Icon={FiPlus}
                                        disabled={selectedDemo.gated && !hfToken.trim()}
                                        style={{ width: '100%', justifyContent: 'center', ...(selectedDemo.gated && !hfToken.trim() ? { opacity: 0.6, cursor: 'not-allowed' } : {}) }}
                                    />
                                </div>
                            </div>
                        </>
                    )}
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                    <div>
                        <label style={labelStyle}>Model Name</label>
                        <input className="glass-input" placeholder="e.g. My Llama Model" value={modelName} onChange={(e) => setModelName(e.target.value)} autoFocus maxLength={120} />
                    </div>
                    {uploadMode === 'local' ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            <div style={{
                                background: 'rgba(14, 165, 233, 0.08)', border: '1px solid rgba(14, 165, 233, 0.25)',
                                borderRadius: '10px', padding: '14px 16px', fontSize: '0.82rem', lineHeight: '1.55', color: '#94a3b8',
                            }}>
                                <span style={{ color: '#67e8f9', fontWeight: 600 }}>Upload a .zip containing your full model directory.</span>
                                <br />It must include:
                                <ul style={{ margin: '6px 0 0 16px', padding: 0 }}>
                                    <li><b style={{ color: '#e2e8f0' }}>config.json</b> — model architecture definition</li>
                                    <li><b style={{ color: '#e2e8f0' }}>Model weights</b> — .safetensors, .bin, or .pth file(s)</li>
                                    <li><b style={{ color: '#e2e8f0' }}>Tokenizer files</b> — tokenizer.json, tokenizer_config.json</li>
                                </ul>
                                <span style={{ fontSize: '0.78rem', color: '#64748b' }}>
                                    Optional: vocab.json, merges.txt, generation_config.json, special_tokens_map.json.
                                    Zip all model files into a single archive — no extra files allowed.
                                </span>
                            </div>
                            <div>
                                <label style={labelStyle}>Select .zip File</label>
                                <input type="file" className="glass-input" accept=".zip" onChange={(e) => setSelectedFile(e.target.files[0])} style={{ padding: '10px' }} />
                            </div>
                        </div>
                    ) : (
                        <div>
                            <label style={labelStyle}>Hugging Face Link</label>
                            <input className="glass-input" placeholder="e.g. https://huggingface.co/meta-llama/Llama-2-7b-hf" value={hfPath} onChange={(e) => setHfPath(e.target.value)} maxLength={2048} />
                            <label style={{ ...labelStyle, marginTop: '12px' }}>Hugging Face Token <span style={{ opacity: 0.6, fontWeight: 400 }}>(optional — only for gated / private models)</span></label>
                            <input className="glass-input" type="password" placeholder="hf_..." value={hfToken} onChange={(e) => setHfToken(e.target.value)} maxLength={512} autoComplete="off" />
                        </div>
                    )}
                    <div style={{ display: 'flex', gap: '12px', marginTop: '10px' }}>
                        <button onClick={() => setUploadMode(null)} style={modalBackBtnStyle}>Back</button>
                        <div style={{ flex: '1' }}>
                            <ReactiveButton label="Confirm" onClick={handleSubmit} Icon={FiPlus} style={{ width: '100%', justifyContent: 'center' }} />
                        </div>
                    </div>
                </div>
            )}
        </GlassModal>
    );
};

const optionBtnStyle = { display: 'flex', alignItems: 'center', gap: '15px', width: '100%', padding: '12px', background: 'rgba(15, 23, 42, 0.55)', border: '1px solid rgba(148, 163, 184, 0.18)', borderRadius: '12px', cursor: 'pointer', transition: 'all 0.2s', color: '#e2e8f0' };
const iconBoxStyle = { padding: '10px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center' };
const labelStyle = { display: 'block', marginBottom: '8px', fontWeight: '600', fontSize: '0.9rem', color: '#cbd5e1' };
const modalBackBtnStyle = { flex: '1', padding: '12px', borderRadius: '12px', border: '1px solid rgba(148, 163, 184, 0.18)', background: 'rgba(15, 23, 42, 0.55)', color: '#cbd5e1', cursor: 'pointer', fontWeight: '600', fontSize: '1rem' };
const gatedBadgeStyle = { flexShrink: 0, fontSize: '0.68rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em', color: '#fcd34d', background: 'rgba(245, 158, 11, 0.16)', border: '1px solid rgba(251, 191, 36, 0.4)', borderRadius: 6, padding: '2px 8px' };

export default AddModelModal;
