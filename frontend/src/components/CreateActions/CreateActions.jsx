import { useState } from 'react';
import { FiCpu, FiLayers } from 'react-icons/fi';
import ReactiveButton from '../ReactiveButton/ReactiveButton';
import RuleGenerationModal from '../../pages/RuleGenerationModal';
import BuildRuleFromCEsModal from '../../pages/BuildRuleFromCEsModal';
import CEGenerationModal from '../../pages/CEGenerationModal';

// Reusable cluster of "create" entry points, mirroring the buttons + modals
// that live on the Community Browse / BrowseCEs pages. Drop it into any page
// (My Drafts, Bookmarks, etc.) to surface the same create flows.
//
// The modals are self-contained: each starts a fresh pipeline run on open,
// builds in the background on approve, and broadcasts gavel:libraryChanged on
// success — so any page using useLibraryRefresh (or listening for that event)
// refreshes itself. No success callbacks are wired here, matching how the
// Browse pages mount them.
//
//   kind='rule' → "Create Rule with AI" (FiCpu) + "Build Rule from CEs" (FiLayers)
//   kind='ce'   → "Create New CE (AI)" (FiCpu)
const CreateActions = ({ kind, style = {} }) => {
    const [ruleModalOpen, setRuleModalOpen] = useState(false);
    const [buildFromCEsOpen, setBuildFromCEsOpen] = useState(false);
    const [ceModalOpen, setCeModalOpen] = useState(false);

    const wrapperStyle = { display: 'flex', gap: 8, flexWrap: 'wrap', ...style };

    if (kind === 'ce') {
        return (
            <div style={wrapperStyle}>
                <ReactiveButton
                    label="Create New CE (AI)"
                    onClick={() => setCeModalOpen(true)}
                    Icon={FiCpu}
                />
                <CEGenerationModal open={ceModalOpen} onClose={() => setCeModalOpen(false)} />
            </div>
        );
    }

    return (
        <div style={wrapperStyle}>
            <ReactiveButton
                label="Create Rule with AI"
                onClick={() => setRuleModalOpen(true)}
                Icon={FiCpu}
            />
            <ReactiveButton
                label="Build Rule from CEs"
                onClick={() => setBuildFromCEsOpen(true)}
                Icon={FiLayers}
            />
            <RuleGenerationModal open={ruleModalOpen} onClose={() => setRuleModalOpen(false)} />
            <BuildRuleFromCEsModal open={buildFromCEsOpen} onClose={() => setBuildFromCEsOpen(false)} />
        </div>
    );
};

export default CreateActions;
