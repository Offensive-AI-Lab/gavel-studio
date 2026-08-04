// OpenAiKeyModal — the one place the user sets their OpenAI key.
//
// Mounted once from App.jsx. It has no props: it opens when something calls
// promptForOpenAiKey() (the axios interceptor on the 503 contract error, or a
// surface that spots the marker in a 200 body / a polled status).
//
// Saving applies the key to the running backend, so whatever just failed can
// be retried straight away — promptForOpenAiKey({ onSaved }) gets that callback
// once the save succeeds.
import { useCallback, useEffect, useRef, useState } from 'react';
import { FiKey } from 'react-icons/fi';
import GlassModal from '../GlassModal/GlassModal';
import ReactiveButton from '../ReactiveButton/ReactiveButton';
import { notifyOpenAiKeySaved, subscribeOpenAiKeyPrompt } from './openAiKeyPrompt';
import { saveOpenAiKey } from '../../api';

const OpenAiKeyModal = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [apiKey, setApiKey] = useState('');
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState('');
    const openRef = useRef(false);
    const onSavedRef = useRef(null);

    const open = useCallback(({ onSaved = null } = {}) => {
        onSavedRef.current = onSaved;
        // A second prompt while the modal is already up must not wipe what the
        // user has typed.
        if (!openRef.current) {
            setApiKey('');
            setError('');
            setSaving(false);
        }
        openRef.current = true;
        setIsOpen(true);
    }, []);

    useEffect(() => subscribeOpenAiKeyPrompt(open), [open]);

    const close = useCallback(() => {
        openRef.current = false;
        setIsOpen(false);
    }, []);

    const handleSave = async () => {
        const value = apiKey.trim();
        if (!value) {
            setError('Paste your key to continue.');
            return;
        }
        setSaving(true);
        setError('');
        try {
            await saveOpenAiKey(value);
            const onSaved = onSavedRef.current;
            onSavedRef.current = null;
            close();
            // Everyone showing a "no key yet" note re-checks and clears it…
            notifyOpenAiKeySaved();
            // …and the one call that failed retries itself.
            onSaved?.();
        } catch (e) {
            setSaving(false);
            const detail = e?.response?.data?.detail;
            setError(typeof detail === 'string' && detail
                ? detail
                : 'That key was not accepted. Check it and try again.');
        }
    };

    return (
        <GlassModal isOpen={isOpen} onClose={saving ? () => {} : close} title="Add your OpenAI key">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                <p style={{ margin: 0, color: '#94a3b8', fontSize: '0.88rem', lineHeight: 1.55 }}>
                    Gavel uses OpenAI for this step and no key is set yet. Paste your key below
                    and it starts working right away.
                </p>
                <div>
                    <label style={labelStyle} htmlFor="openai-key-input">OpenAI API Key</label>
                    <input
                        id="openai-key-input"
                        className="glass-input"
                        type="password"
                        placeholder="sk-..."
                        value={apiKey}
                        onChange={(e) => { setApiKey(e.target.value); if (error) setError(''); }}
                        onKeyDown={(e) => { if (e.key === 'Enter' && !saving) handleSave(); }}
                        maxLength={512}
                        autoComplete="off"
                        autoFocus
                        disabled={saving}
                    />
                    <p style={{ margin: '8px 0 0', color: '#64748b', fontSize: '0.78rem' }}>
                        The key is stored on this machine and never shown again.
                    </p>
                </div>
                {error && <div style={errorStyle} role="alert">{error}</div>}
                <div style={{ display: 'flex', gap: '12px', marginTop: '4px' }}>
                    <button onClick={close} style={cancelBtnStyle} disabled={saving}>Cancel</button>
                    <div style={{ flex: '1' }}>
                        <ReactiveButton
                            label={saving ? 'Saving…' : 'Save key'}
                            onClick={handleSave}
                            Icon={FiKey}
                            disabled={saving}
                            style={{ width: '100%', justifyContent: 'center', ...(saving ? { opacity: 0.6, cursor: 'not-allowed' } : {}) }}
                        />
                    </div>
                </div>
            </div>
        </GlassModal>
    );
};

const labelStyle = { display: 'block', marginBottom: '8px', fontWeight: '600', fontSize: '0.9rem', color: '#cbd5e1' };
const cancelBtnStyle = { flex: '1', padding: '12px', borderRadius: '12px', border: '1px solid rgba(148, 163, 184, 0.18)', background: 'rgba(15, 23, 42, 0.55)', color: '#cbd5e1', cursor: 'pointer', fontWeight: '600', fontSize: '1rem' };
const errorStyle = { background: 'rgba(239, 68, 68, 0.10)', border: '1px solid rgba(248, 113, 113, 0.35)', borderRadius: '10px', padding: '10px 14px', color: '#fca5a5', fontSize: '0.85rem', lineHeight: 1.5 };

export default OpenAiKeyModal;
