// CreateChooserModal — the shared "what do you want to create?" modal.
//
// Lists the three authoring flows (CE / Rule / Build Rule); picking one opens
// that flow's modal. Used by the sidebar "Create" item and by the "Create a New
// Rule" card on the Rule Set Manager, so both entry points behave identically.
import { useState } from 'react';
import { FiCpu, FiZap, FiLayers } from 'react-icons/fi';
import GlassModal from '../GlassModal/GlassModal';
import RuleGenerationModal from '../../pages/RuleGenerationModal';
import BuildRuleFromCEsModal from '../../pages/BuildRuleFromCEsModal';
import CEGenerationModal from '../../pages/CEGenerationModal';

const CreateChooserModal = ({ isOpen, onClose }) => {
    const [ruleAI, setRuleAI] = useState(false);
    const [ruleBuild, setRuleBuild] = useState(false);
    const [ceAI, setCeAI] = useState(false);

    const pick = (setter) => { onClose(); setter(true); };

    const options = [
        { Icon: FiCpu, label: 'CE with AI', desc: 'Generate a cognitive element from a description.', onClick: () => pick(setCeAI) },
        { Icon: FiZap, label: 'Rule with AI', desc: 'Build a rule from a scenario with AI.', onClick: () => pick(setRuleAI) },
        { Icon: FiLayers, label: 'Build Rule from Bookmarked CEs', desc: 'Compose a rule out of CEs in your Library.', onClick: () => pick(setRuleBuild) },
    ];

    return (
        <>
            <GlassModal isOpen={isOpen} onClose={onClose} title="Create">
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    <p style={{ color: '#94a3b8', fontSize: '0.88rem', margin: 0 }}>What do you want to create?</p>
                    {options.map(o => (
                        <button key={o.label} onClick={o.onClick} style={optionRowStyle}>
                            <div style={iconBoxStyle}><o.Icon size={22} /></div>
                            <div style={{ textAlign: 'left' }}>
                                <span style={{ display: 'block', fontWeight: 600, color: '#f1f5f9' }}>{o.label}</span>
                                <span style={{ fontSize: '0.82rem', color: '#94a3b8' }}>{o.desc}</span>
                            </div>
                        </button>
                    ))}
                </div>
            </GlassModal>

            {/* Mount each flow modal only while open (they do real work at render). */}
            {ruleAI && <RuleGenerationModal open onClose={() => setRuleAI(false)} />}
            {ruleBuild && <BuildRuleFromCEsModal open onClose={() => setRuleBuild(false)} />}
            {ceAI && <CEGenerationModal open onClose={() => setCeAI(false)} />}
        </>
    );
};

const optionRowStyle = { display: 'flex', alignItems: 'center', gap: 14, width: '100%', padding: 12, background: 'rgba(15, 23, 42, 0.55)', border: '1px solid rgba(148, 163, 184, 0.18)', borderRadius: 12, cursor: 'pointer', color: '#e2e8f0', transition: 'all 0.15s' };
const iconBoxStyle = { padding: 10, borderRadius: 10, background: 'rgba(99, 102, 241, 0.18)', color: '#c7d2fe', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 };

export default CreateChooserModal;
