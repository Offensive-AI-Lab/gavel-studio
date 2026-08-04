import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

// Components
import Layout from '../components/Layout/Layout';
import ReactiveButton from '../components/ReactiveButton/ReactiveButton';
import GlassModal from '../components/GlassModal/GlassModal';
import ResourceCard from '../components/ResourceCard/ResourceCard';
import GlassSelect from '../components/GlassSelect/GlassSelect';
import AddModelModal from '../components/AddModelModal/AddModelModal';
import Breadcrumb from '../components/Breadcrumb/Breadcrumb';

// Icons & Utils
import { FiPlus, FiInbox, FiHome, FiShield, FiCpu, FiLayers, FiZap, FiCheckCircle, FiAlertCircle, FiRefreshCw, FiUploadCloud, FiDownload, FiCopy, FiFolder, FiFolderPlus, FiEdit2, FiTrash2, FiMove, FiChevronDown, FiChevronRight, FiGitBranch } from 'react-icons/fi';
import ExportToRegistryModal from '../components/ExportToRegistry/ExportToRegistryModal';
import { FaRocket } from 'react-icons/fa';
import Swal from 'sweetalert2';
import { showAlertDialog, showConfirmDialog, showLoadingDialog } from '../components/ConfirmDialog/confirmDialog';
import {
    getUserGuardrails, createGuardrail, cloneClassifierToModel, getUserModels,
    deleteClassifier, getTrainingStatus, startImport, getBundleJob,
    getGuardrailFolders, createGuardrailFolder, renameGuardrailFolder,
    deleteGuardrailFolder, assignGuardrailFolder,
    getRuleSetBookmarks, forkPublicRuleSet, getPublicRuleSets,
} from '../api';
import { forgetRecent, getRecents } from '../utils/recents';
import { useTutorialContent } from '../contexts/TutorialContext';
import InlineHelp from '../components/InlineHelp/InlineHelp';
import { manualRuleConfig } from '../components/InlineHelp/instructorHelp';

// Status badge helper (shared visual language with the per-model rule sets view).
const StatusBadge = ({ status }) => {
    const styles = {
        untrained:         { bg: 'rgba(148, 163, 184, 0.18)', color: '#cbd5e1',  border: 'rgba(148, 163, 184, 0.30)', label: 'Untrained' },
        training:          { bg: 'rgba(245, 158, 11, 0.20)', color: '#fcd34d',  border: 'rgba(251, 191, 36, 0.40)', label: 'Training...' },
        active:            { bg: 'rgba(16, 185, 129, 0.20)', color: '#6ee7b7',  border: 'rgba(52, 211, 153, 0.40)', label: 'Up to Date' },
        needs_retraining:  { bg: 'rgba(249, 115, 22, 0.20)', color: '#fdba74',  border: 'rgba(251, 146, 60, 0.40)', label: 'Needs Retraining' },
        error:             { bg: 'rgba(239, 68, 68, 0.20)',  color: '#fca5a5',  border: 'rgba(248, 113, 113, 0.40)', label: 'Error' },
    };
    const s = styles[status] || styles.untrained;
    const Icon = status === 'active' ? FiCheckCircle : status === 'error' ? FiAlertCircle : status === 'training' ? FiRefreshCw : FiZap;
    return (
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', padding: '2px 10px', borderRadius: '99px', background: s.bg, color: s.color, border: `1px solid ${s.border}`, fontSize: '0.78rem', fontWeight: '600' }}>
            <Icon size={11} />
            {s.label}
        </span>
    );
};

const Guardrails = () => {
    const navigate = useNavigate();

    const [guardrails, setGuardrails] = useState([]);
    const [models, setModels] = useState([]);
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [trainingIds, setTrainingIds] = useState(new Set());

    // Create (name only — model is chosen later, at train time)
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [newName, setNewName] = useState('');

    // "Apply to another model" (clone)
    const [cloneOpen, setCloneOpen] = useState(false);
    const [cloneSource, setCloneSource] = useState(null); // { classifier_id, name, model_id }
    const [cloneTargetModelId, setCloneTargetModelId] = useState('');
    const [cloneBusy, setCloneBusy] = useState(false);
    const [addModelOpen, setAddModelOpen] = useState(false);

    // Fork from bookmarked Community rule sets into new private rule sets.
    // Multi-select, mirroring the "Add a Rule" picker inside a rule set.
    const [forkModalOpen, setForkModalOpen] = useState(false);
    const [bookmarkedSets, setBookmarkedSets] = useState([]);
    const [forkLoading, setForkLoading] = useState(false);
    const [forkBusy, setForkBusy] = useState(false);
    const [selectedForkIds, setSelectedForkIds] = useState(() => new Set());
    const [expandedForkIds, setExpandedForkIds] = useState(() => new Set());

    // Import a rule set bundle (.gavel.zip) — server-side background job we poll.
    const importInputRef = useRef(null);
    const importPollRef = useRef(null);
    const [importPhase, setImportPhase] = useState(null);
    const IMPORT_JOB_KEY = 'gavel_import_job';

    // Folders ("library" grouping — purely manual).
    const [folders, setFolders] = useState([]);
    const [folderModalOpen, setFolderModalOpen] = useState(false);
    const [newFolderName, setNewFolderName] = useState('');
    const [renameTarget, setRenameTarget] = useState(null); // folder being renamed (modal)
    const [renameValue, setRenameValue] = useState('');
    const [moveTarget, setMoveTarget] = useState(null);     // rule set being moved (modal)
    const [exportTarget, setExportTarget] = useState(null); // rule set being exported (modal)
    const [collapsed, setCollapsed] = useState(() => new Set()); // collapsed folder_ids
    const [draggingId, setDraggingId] = useState(null);     // rule set being dragged
    const [dragOverKey, setDragOverKey] = useState(null);   // drop target: folder_id or 'ungrouped'

    const trainedCount = guardrails.filter(c => c.status === 'active').length;
    const unattachedCount = guardrails.filter(c => !c.model_id).length;
    const pageHelp = {
        title: 'Rule Sets',
        summary: 'A rule set is a reusable collection of rules. Build its rules first, then pick the model it runs on when you train — each rule set trains and evaluates on exactly that one model. To run the same rules on another model, use "Apply to another model" to copy it.',
        sections: [
            {
                heading: 'Right now',
                bullets:
                    guardrails.length === 0
                        ? ['No rule sets yet. Click "Create Rule Set" to start one — just give it a name; you\'ll pick its model when you train.']
                        : [
                            `${guardrails.length} rule set${guardrails.length === 1 ? '' : 's'}; ${trainedCount} trained.`,
                            unattachedCount > 0 ? `${unattachedCount} not yet attached to a model — open one, build rules, then Train to pick its model.` : 'All rule sets are attached to a model.',
                        ],
            },
            {
                heading: 'Per-rule-set actions',
                bullets: [
                    'Click a rule set → build rules, train, evaluate, monitor.',
                    '"Apply to another model" → copy this rule set onto a second model (independent copy, retrained for that model).',
                    'Trash icon → remove the rule set and its trained model.',
                ],
            },
            {
                heading: 'Models',
                bullets: [
                    'A rule set with no model yet appears only here. Once you pick its model, it also shows under that model in the Models view.',
                    'Need to add an LLM? Go to Models.',
                ],
            },
        ],
    };
    useTutorialContent(pageHelp);

    useEffect(() => {
        const storedUser = JSON.parse(sessionStorage.getItem('user'));
        {
            setUser(storedUser);
            fetchGuardrails();
            fetchFolders();
            fetchModels(storedUser);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [navigate]);

    const fetchGuardrails = async () => {
        try {
            const response = await getUserGuardrails();
            const list = response.data.classifiers || [];
            setGuardrails(list);
            // Self-heal the sidebar Recents: drop any 'guardrail' entry whose
            // rule set no longer exists (deleted in another tab/session, or
            // before delete cleared recents). Keeps the list from showing
            // dead links like a deleted "Default (copy)".
            const liveIds = new Set(list.map(c => String(c.classifier_id)));
            getRecents('guardrail').forEach(r => {
                if (!liveIds.has(String(r.id))) forgetRecent('guardrail', r.id);
            });
            const trainingOnes = list.filter(c => c.status === 'training');
            if (trainingOnes.length > 0) {
                setTrainingIds(new Set(trainingOnes.map(c => c.classifier_id)));
            }
        } catch {
            // handled by finally
        } finally {
            setLoading(false);
        }
    };

    const fetchFolders = async () => {
        try {
            const res = await getGuardrailFolders();
            setFolders(res.data.folders || []);
        } catch {
            setFolders([]);
        }
    };

    // Refresh BOTH after anything that can change membership (folders list +
    // each rule set's folder_id).
    const refreshGroups = async () => { await fetchFolders(); await fetchGuardrails(); };

    const fetchModels = async (u) => {
        try {
            const res = await getUserModels(u.user_id);
            setModels(res.data.models || []);
        } catch {
            setModels([]);
        }
    };

    // Poll training status every 5s for rule sets currently training (shared
    // logic with the per-model view: tolerate transient poll failures).
    useEffect(() => {
        if (trainingIds.size === 0) return;
        const interval = setInterval(async () => {
            const ids = [...trainingIds];
            const updated = await Promise.all(
                ids.map(id => getTrainingStatus(id).then(r => r.data).catch(() => null))
            );
            setGuardrails(prev => prev.map(c => {
                const upd = updated.find(u => u && u.classifier_id === c.classifier_id);
                return upd ? { ...c, status: upd.status, model_path: upd.model_path, training_log: upd.training_log, training_phase_detail: upd.training_phase_detail, has_error: upd.has_error } : c;
            }));
            const stillTraining = new Set(
                ids.filter((id, i) => updated[i] === null || updated[i].is_training)
            );
            setTrainingIds(stillTraining);
        }, 5000);
        return () => clearInterval(interval);
    }, [trainingIds]);

    // Resume polling an import job kicked off before a remount/navigation.
    useEffect(() => {
        const jid = sessionStorage.getItem(IMPORT_JOB_KEY);
        if (jid) { setImportPhase('Working…'); pollImportJob(Number(jid)); }
        return () => stopImportPoll();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const handleDelete = async (classifierId) => {
        const ok = await showConfirmDialog({
            title: 'Delete rule set?',
            message: 'This will permanently remove the rule set and ALL its rules and cognitive elements.',
            confirmText: 'Delete',
            cancelText: 'Cancel',
            variant: 'danger',
        });
        if (!ok) return;
        const close = showLoadingDialog({ title: 'Deleting rule set', message: 'Removing rule set and its trained model...' });
        try {
            await deleteClassifier(classifierId);
            setGuardrails(prev => prev.filter(c => c.classifier_id !== classifierId));
            // Drop it from the sidebar Recents so it doesn't linger / 404.
            forgetRecent('guardrail', classifierId);
            close();
            await showAlertDialog({ title: 'Deleted', message: 'Rule set has been removed.', variant: 'success' });
        } catch {
            close();
            await showAlertDialog({ title: 'Error', message: 'Failed to delete rule set.', variant: 'error' });
        }
    };

    const handleCreate = async () => {
        if (!newName.trim()) return;
        setIsModalOpen(false);
        const close = showLoadingDialog({ title: 'Creating rule set', message: 'Setting up your new rule set...' });
        try {
            const res = await createGuardrail(newName.trim());
            setNewName('');
            close();
            const id = res?.data?.classifier?.classifier_id;
            if (id) navigate(`/classifiers/${id}/rules`);
            else await fetchGuardrails();
        } catch (error) {
            close();
            await showAlertDialog({
                title: 'Error',
                message: error?.response?.data?.detail || 'Failed to create rule set',
                variant: 'error',
            });
        }
    };

    const openClone = (g) => {
        if (models.length === 0) {
            // No dead-end: add a model right here, then come back to clone.
            setAddModelOpen(true);
            return;
        }
        setCloneSource(g);
        // Default to the first model that isn't the rule set's current one.
        const firstOther = models.find(m => m.model_id !== g.model_id) || models[0];
        setCloneTargetModelId(String(firstOther.model_id));
        setCloneOpen(true);
    };

    const handleClone = async () => {
        if (!cloneSource || !cloneTargetModelId) return;
        setCloneBusy(true);
        try {
            const res = await cloneClassifierToModel(cloneSource.classifier_id, Number(cloneTargetModelId));
            setCloneOpen(false);
            setCloneSource(null);
            await fetchGuardrails();
            const newId = res?.data?.classifier?.classifier_id;
            const targetName = models.find(m => String(m.model_id) === String(cloneTargetModelId))?.name || 'the model';
            const goOpen = await showConfirmDialog({
                title: 'Copied to another model',
                message: `A copy was created on “${targetName}”. It's untrained — train it to run on that model. Open it now?`,
                confirmText: 'Open copy',
                cancelText: 'Stay here',
                variant: 'info',
            });
            if (goOpen && newId) navigate(`/classifiers/${newId}/rules`);
        } catch (error) {
            await showAlertDialog({
                title: 'Error',
                message: error?.response?.data?.detail || 'Failed to copy this rule set.',
                variant: 'error',
            });
        } finally {
            setCloneBusy(false);
        }
    };

    // --- Fork bookmarked Community rule sets into new private rule sets -------
    const openForkPicker = async () => {
        setForkModalOpen(true);
        setForkLoading(true);
        setSelectedForkIds(new Set());
        setExpandedForkIds(new Set());
        try {
            // Which sets the user bookmarked (only forkable ones have a public_id),
            // hydrated to full objects (member count, description) from the public list.
            const [bk, pub] = await Promise.all([
                getRuleSetBookmarks(user.user_id),
                getPublicRuleSets(),
            ]);
            const ids = new Set((bk.data?.bookmarks || []).filter((b) => b.public_id).map((b) => b.rule_set_id));
            const all = pub.data?.rule_sets || [];
            setBookmarkedSets(all.filter((rs) => ids.has(rs.rule_set_id)));
        } catch {
            setBookmarkedSets([]);
        } finally {
            setForkLoading(false);
        }
    };

    const toggleForkSelect = (id) => {
        setSelectedForkIds((prev) => {
            const next = new Set(prev);
            const k = String(id);
            next.has(k) ? next.delete(k) : next.add(k);
            return next;
        });
    };

    const handleForkSelected = async () => {
        const chosen = bookmarkedSets.filter((rs) => selectedForkIds.has(String(rs.rule_set_id)) && rs.public_id);
        if (chosen.length === 0) return;
        setForkBusy(true);
        let lastNewId = null;
        let ok = 0;
        const failed = [];
        for (const set of chosen) {
            try {
                const res = await forkPublicRuleSet(set.public_id);
                lastNewId = res?.data?.classifier?.classifier_id ?? lastNewId;
                ok += 1;
            } catch {
                failed.push(set.name);
            }
        }
        setForkBusy(false);
        setForkModalOpen(false);
        await fetchGuardrails();
        if (ok > 0 && failed.length === 0 && chosen.length === 1 && lastNewId) {
            navigate(`/classifiers/${lastNewId}/rules`);
            return;
        }
        if (ok > 0) {
            await showAlertDialog({
                title: 'Forked',
                message: `${ok} rule set${ok === 1 ? '' : 's'} added to your workspace${failed.length ? ` (${failed.length} failed: ${failed.join(', ')})` : ''}.`,
                variant: failed.length ? 'warning' : 'success',
            });
        } else {
            await showAlertDialog({ title: 'Could not fork', message: `Failed to fork: ${failed.join(', ')}.`, variant: 'error' });
        }
    };

    const stopImportPoll = () => {
        if (importPollRef.current) { clearInterval(importPollRef.current); importPollRef.current = null; }
    };

    const pollImportJob = (jobId) => {
        stopImportPoll();
        importPollRef.current = setInterval(async () => {
            let job;
            try {
                ({ data: job } = await getBundleJob(jobId));
            } catch (err) {
                if (err?.response?.status === 404) {
                    stopImportPoll();
                    setImportPhase(null);
                    sessionStorage.removeItem(IMPORT_JOB_KEY);
                }
                return;
            }
            if (job.status === 'running') {
                setImportPhase(job.phase || 'Working…');
                return;
            }
            stopImportPoll();
            setImportPhase(null);
            sessionStorage.removeItem(IMPORT_JOB_KEY);
            if (job.status === 'done') {
                const r = job.result || {};
                const tierLabel = r.evaluated ? 'model + calibration + evaluation'
                    : r.calibrated ? 'model + calibration' : 'model only';
                await showAlertDialog({
                    title: 'Rule set imported',
                    message: `“${r.name || 'Rule set'}” was added (${tierLabel}${r.base_model ? `; base model ${r.base_model}` : ''}).`,
                    variant: 'success',
                });
                await fetchGuardrails();
            } else {
                await showAlertDialog({
                    title: 'Import failed',
                    message: job.error || 'Could not import this bundle.',
                    variant: 'error',
                });
            }
        }, 1500);
    };

    const handleImportFile = async (e) => {
        const file = e.target.files?.[0];
        e.target.value = '';
        if (!file) return;
        try {
            const { data } = await startImport(file);
            sessionStorage.setItem(IMPORT_JOB_KEY, String(data.job_id));
            setImportPhase('Starting…');
            Swal.fire({ icon: 'info', title: 'Import started', text: 'Running in the background — you can keep working.', timer: 1800, showConfirmButton: false });
            pollImportJob(data.job_id);
        } catch (err) {
            await showAlertDialog({
                title: 'Import failed',
                message: err?.response?.data?.detail || 'Could not start the import.',
                variant: 'error',
            });
        }
    };

    // --- Folders (manual grouping) ------------------------------------------
    const toggleCollapse = (folderId) => {
        setCollapsed(prev => {
            const next = new Set(prev);
            next.has(folderId) ? next.delete(folderId) : next.add(folderId);
            return next;
        });
    };

    const handleCreateFolder = async () => {
        const name = newFolderName.trim();
        if (!name) return;
        setFolderModalOpen(false);
        setNewFolderName('');
        try {
            await createGuardrailFolder(name);
            await fetchFolders();
        } catch {
            await showAlertDialog({ title: 'Error', message: 'Could not create the folder.', variant: 'error' });
        }
    };

    const handleRenameFolder = (folder) => {
        setRenameValue(folder.name);
        setRenameTarget(folder);
    };

    const submitRenameFolder = async () => {
        const folder = renameTarget;
        const name = renameValue.trim();
        if (!folder || !name) return;
        setRenameTarget(null);
        try {
            await renameGuardrailFolder(folder.folder_id, name);
            await fetchFolders();
        } catch {
            await showAlertDialog({ title: 'Error', message: 'Could not rename the folder.', variant: 'error' });
        }
    };

    const handleDeleteFolder = async (folder) => {
        const count = guardrails.filter(g => g.folder_id === folder.folder_id).length;
        const ok = await showConfirmDialog({
            title: `Delete “${folder.name}”?`,
            message: count > 0
                ? `This will permanently DELETE the folder AND the ${count} rule set${count === 1 ? '' : 's'} inside it — including any that are mid-training. Those rule sets and their rules will be lost. This can't be undone.`
                : 'This will permanently delete the folder.',
            confirmText: count > 0 ? `Delete folder + ${count} rule set${count === 1 ? '' : 's'}` : 'Delete folder',
            cancelText: 'Cancel',
            variant: 'danger',
        });
        if (!ok) return;
        // Folder delete cascade-deletes the rule sets inside it — forget their
        // Recents entries too so the sidebar doesn't keep dead links.
        const cascadedIds = guardrails
            .filter(g => g.folder_id === folder.folder_id)
            .map(g => g.classifier_id);
        try {
            await deleteGuardrailFolder(folder.folder_id);
            cascadedIds.forEach(id => forgetRecent('guardrail', id));
            await refreshGroups();
        } catch {
            await showAlertDialog({ title: 'Error', message: 'Could not delete the folder.', variant: 'error' });
        }
    };

    // Move via the modal (folderId may be a number or null = ungroup).
    const handleMove = async (folderId) => {
        if (!moveTarget) return;
        const g = moveTarget;
        setMoveTarget(null);
        await doAssign(g.classifier_id, folderId);
    };

    // Shared by the modal and drag-and-drop.
    const doAssign = async (classifierId, folderId) => {
        try {
            await assignGuardrailFolder(classifierId, folderId);
            await refreshGroups();
        } catch (err) {
            await showAlertDialog({
                title: 'Error',
                message: err?.response?.data?.detail || 'Could not move the rule set.',
                variant: 'error',
            });
        }
    };

    // Drag-and-drop: drop a rule set onto a folder (or the Ungrouped zone) to
    // move it in/out. `key` is a folder_id or null (ungrouped).
    const onDropTo = (folderId) => (e) => {
        e.preventDefault();
        const id = draggingId;
        setDragOverKey(null);
        setDraggingId(null);
        if (id == null) return;
        const cur = guardrails.find(g => g.classifier_id === id);
        if (cur && (cur.folder_id ?? null) === (folderId ?? null)) return; // no-op
        doAssign(id, folderId);
    };
    const allowDrop = (key) => (e) => { e.preventDefault(); if (dragOverKey !== key) setDragOverKey(key); };

    // One rule set card — reused across folder sections and the Ungrouped list.
    // Draggable: drop onto a folder (or the Ungrouped zone) to move it.
    const renderCard = (c) => (
        <div
            key={c.classifier_id}
            style={{ ...cardWrapStyle, opacity: draggingId === c.classifier_id ? 0.4 : 1 }}
            draggable
            onDragStart={(e) => { setDraggingId(c.classifier_id); e.dataTransfer.effectAllowed = 'move'; }}
            onDragEnd={() => { setDraggingId(null); setDragOverKey(null); }}
        >
            <ResourceCard
                title={c.name}
                subtitle={c.model_name
                    ? <><FiCpu size={14} style={{ marginRight: '6px' }} />{c.model_name}</>
                    : <span style={{ color: '#94a3b8', fontStyle: 'italic' }}>No model yet</span>}
                icon={FiLayers}
                onClick={() => navigate(`/classifiers/${c.classifier_id}/rules`)}
                onDelete={() => handleDelete(c.classifier_id)}
            />
            <div style={cardFooterStyle}>
                <StatusBadge status={c.status || 'untrained'} />
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <button
                        onClick={(e) => { e.stopPropagation(); setMoveTarget(c); }}
                        style={cloneBtnStyle}
                        title="Move this rule set to a folder"
                    >
                        <FiMove size={12} /> Move
                    </button>
                    <button
                        onClick={(e) => { e.stopPropagation(); setExportTarget(c); }}
                        style={cloneBtnStyle}
                        title="Save this rule set in the public library's format, ready to contribute"
                    >
                        <FiDownload size={12} /> Export
                    </button>
                    {c.model_id ? (
                        <button
                            onClick={(e) => { e.stopPropagation(); openClone(c); }}
                            style={cloneBtnStyle}
                            title="Copy this rule set onto another model"
                        >
                            <FiCopy size={12} /> Apply to model
                        </button>
                    ) : (
                        <span style={{ fontSize: '0.74rem', color: '#94a3b8' }}>Pick a model to train</span>
                    )}
                </div>
            </div>
            {c.status === 'training' && (
                <div style={{ ...cardFooterStyle, justifyContent: 'flex-start', borderRadius: 0, marginTop: 0 }}>
                    <span style={{ fontSize: '0.78rem', color: '#fcd34d', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <FiRefreshCw size={12} style={{ animation: 'spin 1s linear infinite' }} /> {c.training_phase_detail || 'Training...'}
                    </span>
                </div>
            )}
            {c.status === 'error' && (c.training_phase_detail || c.training_log) && (() => {
                const msg = c.training_phase_detail || (typeof c.training_log === 'string' ? c.training_log : 'Training failed');
                return (
                    <div style={errorBannerStyle} title={msg}>
                        <FiAlertCircle size={12} /> {msg.slice(0, 100)}{msg.length > 100 ? '…' : ''}
                    </div>
                );
            })()}
        </div>
    );

    const ungrouped = guardrails.filter(g => !g.folder_id);

    if (!user) return null;

    return (
        <Layout>
            <header className="page-header">
                <div>
                    <Breadcrumb items={[
                        { label: 'Hub', icon: FiHome, to: '/workspace' },
                        { label: 'Rule Sets', icon: FiShield },
                    ]} />
                    <h1>Rule Sets</h1>
                    <p>Build a rule set, then pick the model it runs on at train time.</p>
                </div>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
                    <button onClick={() => setFolderModalOpen(true)} style={importBtnStyle} title="Create a folder to group rule sets">
                        <FiFolderPlus size={15} /> New folder
                    </button>
                    <input
                        ref={importInputRef}
                        type="file"
                        accept=".zip,application/zip"
                        style={{ display: 'none' }}
                        onChange={handleImportFile}
                    />
                    <button
                        onClick={() => importInputRef.current?.click()}
                        style={{ ...importBtnStyle, opacity: importPhase ? 0.7 : 1, cursor: importPhase ? 'default' : 'pointer' }}
                        disabled={!!importPhase}
                        title={importPhase ? 'An import is already running in the background' : 'Import a rule set bundle (.gavel.zip) shared by another user'}
                    >
                        {importPhase
                            ? <><FiRefreshCw size={15} style={{ animation: 'spin 1s linear infinite' }} /> {importPhase}</>
                            : <><FiUploadCloud size={15} /> Import</>}
                    </button>
                    <button
                        onClick={openForkPicker}
                        style={importBtnStyle}
                        title="Fork a rule set you bookmarked from the Community into your workspace"
                    >
                        <FiGitBranch size={15} /> Fork bookmarked
                    </button>
                    <ReactiveButton label="Create Rule Set" onClick={() => setIsModalOpen(true)} Icon={FiPlus} />
                </div>
            </header>

            <InlineHelp content={manualRuleConfig} />

            {folders.length > 0 && (
                <p style={{ color: '#64748b', fontSize: '0.82rem', margin: '0 0 18px' }}>
                    Drag a rule set onto a folder to add it, or onto “Ungrouped” to take it out — or use the <strong>Move</strong> button on a card.
                </p>
            )}

            {loading ? (
                <div style={{ textAlign: 'center', padding: '60px', color: '#94a3b8' }}>Loading...</div>
            ) : guardrails.length === 0 ? (
                <div className="empty-state">
                    <FiInbox size={64} style={{ color: '#64748b', marginBottom: '20px' }} />
                    <h2 style={{ color: '#f1f5f9', margin: '0 0 8px' }}>No Rule Sets Yet</h2>
                    <p style={{ color: '#94a3b8', margin: '0 0 20px' }}>Create your first rule set — name it, build rules, then pick a model to train on.</p>
                    <ReactiveButton label="Create Your First Rule Set" onClick={() => setIsModalOpen(true)} Icon={FiPlus} />
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
                    {folders.map(f => {
                        const items = guardrails.filter(g => g.folder_id === f.folder_id);
                        const isCollapsed = collapsed.has(f.folder_id);
                        const isOver = dragOverKey === f.folder_id;
                        return (
                            <section
                                key={f.folder_id}
                                onDragOver={allowDrop(f.folder_id)}
                                onDragLeave={() => setDragOverKey(k => (k === f.folder_id ? null : k))}
                                onDrop={onDropTo(f.folder_id)}
                                style={{ ...sectionStyle, ...(isOver ? sectionDropActiveStyle : {}) }}
                            >
                                <div style={folderHeaderStyle}>
                                    <button
                                        onClick={() => toggleCollapse(f.folder_id)}
                                        style={{ ...iconBtnStyle, border: 'none', background: 'none' }}
                                        title={isCollapsed ? 'Expand folder' : 'Collapse folder'}
                                    >
                                        {isCollapsed ? <FiChevronRight size={16} /> : <FiChevronDown size={16} />}
                                    </button>
                                    <FiFolder size={16} style={{ color: '#93c5fd', flexShrink: 0 }} />
                                    <h3
                                        onClick={() => toggleCollapse(f.folder_id)}
                                        style={{ margin: 0, color: '#e2e8f0', fontSize: '1rem', fontWeight: 700, cursor: 'pointer' }}
                                    >{f.name}</h3>
                                    <span style={folderCountStyle}>{items.length}</span>
                                    <button onClick={() => handleRenameFolder(f)} style={iconBtnStyle} title="Rename folder"><FiEdit2 size={13} /></button>
                                    <button onClick={() => handleDeleteFolder(f)} style={iconBtnStyle} title="Delete folder (also deletes the rule sets inside)"><FiTrash2 size={13} /></button>
                                </div>
                                {isCollapsed ? null : items.length === 0 ? (
                                    <div style={emptyFolderHintStyle}>Empty — drag a rule set here, or use “Move” on a card.</div>
                                ) : (
                                    <div className="stats-grid" style={gridStyle}>{items.map(renderCard)}</div>
                                )}
                            </section>
                        );
                    })}

                    {/* Ungrouped — labelled (and a drop zone) only when folders exist;
                      * otherwise this is just the flat list of all rule sets. */}
                    <section
                        onDragOver={folders.length > 0 ? allowDrop('ungrouped') : undefined}
                        onDragLeave={() => setDragOverKey(k => (k === 'ungrouped' ? null : k))}
                        onDrop={folders.length > 0 ? onDropTo(null) : undefined}
                        style={{ ...sectionStyle, ...(dragOverKey === 'ungrouped' ? sectionDropActiveStyle : {}) }}
                    >
                        {folders.length > 0 && (
                            <div style={folderHeaderStyle}>
                                <FiInbox size={16} style={{ color: '#94a3b8', flexShrink: 0 }} />
                                <h3 style={{ margin: 0, color: '#cbd5e1', fontSize: '1rem', fontWeight: 700 }}>Ungrouped</h3>
                                <span style={folderCountStyle}>{ungrouped.length}</span>
                            </div>
                        )}
                        {ungrouped.length > 0 ? (
                            <div className="stats-grid" style={gridStyle}>{ungrouped.map(renderCard)}</div>
                        ) : folders.length > 0 ? (
                            <div style={emptyFolderHintStyle}>Nothing ungrouped — drag a rule set here to remove it from its folder.</div>
                        ) : null}
                    </section>
                </div>
            )}

            <GlassModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} title="New Rule Set">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                    <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: 0 }}>
                        Name your rule set. You'll build its rules next and choose which model it runs on when you train.
                    </p>
                    <input className="glass-input" placeholder="e.g. Finance-Guard-v1" value={newName} onChange={e => setNewName(e.target.value)} autoFocus maxLength={120} />
                    <ReactiveButton label="Create & Build Rules" onClick={handleCreate} Icon={FaRocket} style={{ justifyContent: 'center', width: '100%' }} />
                </div>
            </GlassModal>

            <GlassModal isOpen={cloneOpen} onClose={() => setCloneOpen(false)} title="Apply to another model">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                    <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: 0 }}>
                        Copy <strong style={{ color: '#e2e8f0' }}>{cloneSource?.name}</strong>'s rule set onto another model.
                        The copy is independent and starts untrained — you'll train it for that model.
                    </p>
                    <label style={{ fontSize: '0.85rem', color: '#cbd5e1', marginBottom: '-6px' }}>Target model</label>
                    <GlassSelect
                        value={cloneTargetModelId}
                        onChange={setCloneTargetModelId}
                        placeholder="Select a model"
                        options={models.map(m => ({
                            value: m.model_id,
                            label: `${m.name}${m.model_id === cloneSource?.model_id ? ' (current)' : ''}`,
                        }))}
                    />
                    <ReactiveButton
                        label={cloneBusy ? 'Copying...' : 'Copy rule set'}
                        onClick={handleClone}
                        Icon={FiCopy}
                        style={{ justifyContent: 'center', width: '100%' }}
                    />
                </div>
            </GlassModal>

            <GlassModal isOpen={forkModalOpen} onClose={() => setForkModalOpen(false)} title="Fork bookmarked rule sets">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>
                        Pick the bookmarked Community rule sets to copy into your workspace — choose as many as you like.
                        Each copy is private and starts untrained; you'll pick a model when you train it.
                    </p>
                    {forkLoading ? (
                        <div style={{ textAlign: 'center', padding: '24px', color: '#94a3b8' }}>Loading your bookmarks…</div>
                    ) : bookmarkedSets.length === 0 ? (
                        <div style={{ textAlign: 'center', padding: '24px', color: '#94a3b8', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'center' }}>
                            <span>No bookmarked rule sets yet — bookmark some from the Community first.</span>
                            <button
                                onClick={() => { setForkModalOpen(false); navigate('/community/rule-sets'); }}
                                style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', padding: '8px 16px', borderRadius: '10px', border: '1px solid rgba(96, 165, 250, 0.45)', background: 'rgba(59, 130, 246, 0.18)', color: '#bfdbfe', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 600 }}
                            >
                                <FiLayers size={14} /> Browse Community Rule Sets
                            </button>
                        </div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '300px', overflowY: 'auto' }}>
                            {bookmarkedSets.map((set) => {
                                const selected = selectedForkIds.has(String(set.rule_set_id));
                                const count = Array.isArray(set.member_rules) ? set.member_rules.length : 0;
                                const desc = (set.description || '').trim();
                                const expanded = expandedForkIds.has(String(set.rule_set_id));
                                const long = desc.length > 130;
                                return (
                                    <div
                                        key={set.rule_set_id}
                                        role="checkbox"
                                        aria-checked={selected}
                                        onClick={() => toggleForkSelect(set.rule_set_id)}
                                        style={{
                                            padding: '14px 16px', borderRadius: '12px', cursor: 'pointer',
                                            border: selected ? '2px solid #a78bfa' : '1px solid rgba(148, 163, 184, 0.18)',
                                            background: selected ? 'rgba(139, 92, 246, 0.18)' : 'rgba(15, 23, 42, 0.55)',
                                            transition: 'all 0.15s',
                                        }}
                                    >
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                            <input type="checkbox" checked={selected} readOnly style={{ accentColor: '#a78bfa', width: 16, height: 16, flexShrink: 0 }} />
                                            <span style={{
                                                fontWeight: 600, color: '#f1f5f9', fontSize: '0.9rem',
                                                flex: 1, minWidth: 0, overflowWrap: 'anywhere',
                                                display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
                                            }} title={set.name}>{set.name}</span>
                                            <span style={{
                                                flexShrink: 0, fontSize: '0.66rem', fontWeight: 700, letterSpacing: '0.04em',
                                                padding: '2px 8px', borderRadius: 999, color: '#93c5fd',
                                                background: 'rgba(59, 130, 246, 0.18)', border: '1px solid rgba(96, 165, 250, 0.40)',
                                            }}>{count} RULE{count === 1 ? '' : 'S'}</span>
                                        </div>
                                        {desc && (
                                            <div style={{ marginTop: '4px', marginLeft: 26 }}>
                                                <div style={{
                                                    fontSize: '0.78rem', color: '#cbd5e1', lineHeight: 1.5, overflowWrap: 'anywhere',
                                                    ...(long && !expanded ? { display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' } : {}),
                                                }}>{desc}</div>
                                                {long && (
                                                    <button
                                                        type="button"
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            setExpandedForkIds((prev) => {
                                                                const next = new Set(prev);
                                                                const k = String(set.rule_set_id);
                                                                next.has(k) ? next.delete(k) : next.add(k);
                                                                return next;
                                                            });
                                                        }}
                                                        style={{ marginTop: 2, padding: 0, background: 'none', border: 'none', color: '#a5b4fc', fontSize: '0.74rem', fontWeight: 600, cursor: 'pointer' }}
                                                    >
                                                        {expanded ? 'Show less' : 'Show more'}
                                                    </button>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                    {bookmarkedSets.length > 0 && (
                        <ReactiveButton
                            label={forkBusy
                                ? 'Forking…'
                                : selectedForkIds.size > 1 ? `Fork ${selectedForkIds.size} Rule Sets` : 'Fork into my workspace'}
                            onClick={handleForkSelected}
                            Icon={FiGitBranch}
                            disabled={selectedForkIds.size === 0 || forkBusy}
                        />
                    )}
                </div>
            </GlassModal>

            <AddModelModal
                isOpen={addModelOpen}
                onClose={() => setAddModelOpen(false)}
                userId={user?.user_id}
                onAdded={() => fetchModels(user)}
            />

            {exportTarget && (
                <ExportToRegistryModal
                    open={!!exportTarget}
                    onClose={() => setExportTarget(null)}
                    kind="ruleset"
                    entityId={exportTarget.classifier_id}
                    defaultTitle={exportTarget.name || ''}
                />
            )}

            <GlassModal isOpen={folderModalOpen} onClose={() => setFolderModalOpen(false)} title="New folder">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                    <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: 0 }}>
                        Name a folder to group rule sets. You can rename it anytime.
                    </p>
                    <input className="glass-input" placeholder="e.g. Finance policies" value={newFolderName} onChange={e => setNewFolderName(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') handleCreateFolder(); }} autoFocus maxLength={120} />
                    <ReactiveButton label="Create folder" onClick={handleCreateFolder} Icon={FiFolderPlus} style={{ justifyContent: 'center', width: '100%' }} />
                </div>
            </GlassModal>

            <GlassModal isOpen={!!renameTarget} onClose={() => setRenameTarget(null)} title="Rename folder">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                    <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: 0 }}>
                        Give this folder a new name.
                    </p>
                    <input
                        className="glass-input"
                        value={renameValue}
                        onChange={e => setRenameValue(e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter') submitRenameFolder(); }}
                        autoFocus
                        maxLength={120}
                    />
                    <ReactiveButton label="Save" onClick={submitRenameFolder} Icon={FiEdit2} style={{ justifyContent: 'center', width: '100%' }} />
                </div>
            </GlassModal>

            <GlassModal isOpen={!!moveTarget} onClose={() => setMoveTarget(null)} title={`Move “${moveTarget?.name || ''}”`}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: '0 0 4px' }}>
                        Pick a folder for this rule set, or take it out.
                    </p>
                    {folders.length === 0 && (
                        <p style={{ color: '#94a3b8', fontSize: '0.8rem', margin: 0 }}>No folders yet — create one with “New folder”.</p>
                    )}
                    {folders.map(f => (
                        <button
                            key={f.folder_id}
                            onClick={() => handleMove(f.folder_id)}
                            style={{ ...moveRowStyle, ...(moveTarget?.folder_id === f.folder_id ? moveRowActiveStyle : {}) }}
                        >
                            <FiFolder size={14} /> {f.name}{moveTarget?.folder_id === f.folder_id ? '  • current' : ''}
                        </button>
                    ))}
                    <button
                        onClick={() => handleMove(null)}
                        style={{ ...moveRowStyle, ...(!moveTarget?.folder_id ? moveRowActiveStyle : {}) }}
                    >
                        <FiInbox size={14} /> Ungrouped{!moveTarget?.folder_id ? '  • current' : ''}
                    </button>
                </div>
            </GlassModal>
        </Layout>
    );
};

const cardWrapStyle = { display: 'flex', flexDirection: 'column' };
const cardFooterStyle = { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap', padding: '8px 12px', background: 'rgba(2, 6, 23, 0.40)', borderRadius: '0 0 12px 12px', borderTop: '1px solid rgba(148, 163, 184, 0.10)', marginTop: '-4px' };
const cloneBtnStyle = { display: 'flex', alignItems: 'center', gap: '4px', padding: '4px 10px', borderRadius: '8px', border: '1px solid rgba(96, 165, 250, 0.40)', background: 'rgba(59, 130, 246, 0.18)', color: '#bfdbfe', cursor: 'pointer', fontSize: '0.74rem', fontWeight: '600', transition: 'all 0.15s' };
const importBtnStyle = { display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 16px', borderRadius: '10px', border: '1px solid rgba(96, 165, 250, 0.45)', background: 'rgba(59, 130, 246, 0.18)', color: '#bfdbfe', cursor: 'pointer', fontSize: '0.9rem', fontWeight: '600', transition: 'all 0.15s' };
const errorBannerStyle = { display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', background: 'rgba(239, 68, 68, 0.18)', color: '#fecaca', fontSize: '0.75rem', borderRadius: '0 0 12px 12px', borderTop: '1px solid rgba(248, 113, 113, 0.30)' };
const sectionStyle = { borderRadius: '14px', padding: '10px 12px', border: '1px solid transparent', transition: 'background 0.12s, border-color 0.12s' };
const sectionDropActiveStyle = { background: 'rgba(59, 130, 246, 0.10)', border: '1px dashed rgba(96, 165, 250, 0.6)' };
const gridStyle = { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '24px' };
const folderHeaderStyle = { display: 'flex', alignItems: 'center', gap: '10px', padding: '0 2px 12px', borderBottom: '1px solid rgba(148, 163, 184, 0.14)', marginBottom: '18px' };
const folderCountStyle = { fontSize: '0.74rem', fontWeight: 700, color: '#94a3b8', background: 'rgba(148, 163, 184, 0.16)', borderRadius: '99px', padding: '1px 9px', marginRight: 'auto' };
const iconBtnStyle = { display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 28, height: 28, borderRadius: '8px', border: '1px solid rgba(148, 163, 184, 0.25)', background: 'rgba(2, 6, 23, 0.40)', color: '#cbd5e1', cursor: 'pointer', flexShrink: 0 };
const emptyFolderHintStyle = { color: '#64748b', fontSize: '0.85rem', fontStyle: 'italic', padding: '10px 4px' };
const moveRowStyle = { display: 'flex', alignItems: 'center', gap: '8px', width: '100%', textAlign: 'left', padding: '10px 12px', borderRadius: '10px', border: '1px solid rgba(148, 163, 184, 0.22)', background: 'rgba(2, 6, 23, 0.40)', color: '#e2e8f0', cursor: 'pointer', fontSize: '0.88rem', fontWeight: 600 };
const moveRowActiveStyle = { border: '1px solid rgba(96, 165, 250, 0.6)', background: 'rgba(59, 130, 246, 0.18)', color: '#bfdbfe' };

export default Guardrails;
