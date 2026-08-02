// Build Rule from Bookmarked CEs — guardrail-agnostic.
//
// The manual twin of the AI Rule Generation wizard (Pipeline A). Composes a
// rule from the user's bookmarked Cognitive Elements — organized into NAMED
// GROUPS with a firing CONDITION written over the group names (the
// gavel-rules condition grammar,
// e.g. "all of required and 1 of option_1") — generates the rule's Test Set +
// calibration set, then lands the finished draft in Drafts — never attached to
// a guardrail. Rendered as the BODY of BuildRuleFromCEsModal, opened from the
// "Build Rule from CEs" button on Browse (no route change).
//
// Steps: Pick CEs → Learn the logic → Groups & Condition → Name → Test & Calibration.
//
// The rule is created is_ready=FALSE on the Name step (so it stays hidden and
// is auto-wiped by boot recovery if abandoned). When the user kicks off the
// final step the modal closes; the rule is finalized (embedded + is_ready=TRUE)
// by the background tray job only AFTER its test set finishes, so it never
// appears half-built. `onClose` closes the modal.
import { useState, useEffect, useRef, useCallback } from 'react';
import { FiArrowLeft, FiArrowRight, FiCheckSquare } from 'react-icons/fi';
import ReactiveButton from '../components/ReactiveButton/ReactiveButton';
import RuleDefaultsStep from '../components/RuleDefaults/RuleDefaultsStep';
import RuleLogicPreview from '../components/RuleLogicPreview/RuleLogicPreview';
import RoleLogicGuide from '../components/RoleLogicGuide/RoleLogicGuide';
import GroupConditionEditor from '../components/GroupConditionEditor/GroupConditionEditor';
import { showAlertDialog } from '../components/ConfirmDialog/confirmDialog';
import { normalizeCategoryValue } from '../utils/categoryUtils';
import { normalizeGroups, validateEditorState } from '../utils/ruleLogic';
import {
    getCEBookmarks,
    getAllCategories,
    createDraftRuleFromBookmarks,
    finalizeRule,
    discardUnreadyRule,
} from '../api';

const STEP_LABELS = ['Pick CEs', 'Learn the logic', 'Groups & Condition', 'Name', 'Test & Calibration'];

function readUser() {
    try { return JSON.parse(sessionStorage.getItem('user') || 'null'); } catch { return null; }
}

// baseRule.groups ({gname: [{ce_id, name}]}) -> editable [{name, ceIds}].
// Members without a ce_id are resolved by name against the base CE list.
function groupsToEditable(groups, baseCes) {
    const byName = new Map((baseCes || []).map((c) => [c.name, c.ce_id]));
    return Object.entries(normalizeGroups(groups)).map(([name, members]) => ({
        name,
        ceIds: members
            .map((m) => (m.ce_id != null ? m.ce_id : byName.get(m.name)))
            .filter((id) => id != null),
    }));
}

export default function BuildRuleFromCEs({ onClose, baseRule = null }) {
    const user = readUser();
    // EDIT mode: `baseRule` carries an existing rule to start from. We pre-load
    // its CEs/groups/condition/categories/name and start at Pick CEs (step 1)
    // with those CEs already SELECTED — the user can add or remove before
    // continuing — then the groups carry through to Groups & Condition, and we
    // FORCE a new name. The result is a brand-new draft; the original rule is
    // never touched.
    const isEdit = !!baseRule;
    // Only CEs with a ce_id can be carried (we relink by id).
    const baseCes = (baseRule?.ces || []).filter((c) => c && c.ce_id != null);

    const [loading, setLoading] = useState(true);
    const [ceBookmarks, setCeBookmarks] = useState([]);
    const [availableCategories, setAvailableCategories] = useState([]);

    const [step, setStep] = useState(1);
    const [ceSearch, setCeSearch] = useState('');   // filter Pick-CEs by name
    const [selectedCes, setSelectedCes] = useState(() => new Set(baseCes.map((c) => c.ce_id)));
    // The rule's logic: named groups over the selected CEs + a condition
    // expression over the group names. Seeded from the base rule in edit
    // mode; otherwise initialized when the user reaches step 3.
    const [groupList, setGroupList] = useState(() => (baseRule?.groups ? groupsToEditable(baseRule.groups, baseCes) : []));
    const [condition, setCondition] = useState(baseRule?.condition || '');
    const [nameInput, setNameInput] = useState(baseRule?.name || '');
    const [selectedCategories, setSelectedCategories] = useState(() => new Set((baseRule?.categories || []).filter(Boolean)));
    const [creating, setCreating] = useState(false);
    const [createdRuleId, setCreatedRuleId] = useState(null);

    // Refs so the unmount cleanup sees current values without re-binding.
    const createdRuleIdRef = useRef(null);
    const committedRef = useRef(false);
    useEffect(() => { createdRuleIdRef.current = createdRuleId; }, [createdRuleId]);

    // Discard the provisional (is_ready=FALSE) rule if the user navigates away
    // before finishing. Tab-close won't fire this, but such rows are wiped by
    // boot recovery, so nothing half-baked survives either way.
    useEffect(() => () => {
        if (createdRuleIdRef.current && !committedRef.current) {
            discardUnreadyRule(createdRuleIdRef.current).catch(() => {});
        }
    }, []);

    const load = useCallback(async () => {
        if (!user?.user_id) { onClose?.(); return; }
        try {
            const [cesRes, catRes] = await Promise.all([
                getCEBookmarks(user.user_id),
                getAllCategories().catch(() => ({ data: [] })),
            ]);
            // In edit mode, merge the base rule's CEs into the pool so they show
            // in Pick/Groups even if they aren't in the user's bookmarks.
            const bm = cesRes.data?.bookmarks || [];
            const pool = [...bm];
            baseCes.forEach((bc) => {
                if (!pool.some((x) => x.ce_id === bc.ce_id)) {
                    pool.push({ ce_id: bc.ce_id, name: bc.name || `CE_${bc.ce_id}`, category: bc.category || '' });
                }
            });
            setCeBookmarks(pool);
            // GET /library/categories returns a plain array of strings.
            const catData = Array.isArray(catRes.data) ? catRes.data : [];
            const names = catData.map((c) => normalizeCategoryValue(c)).filter(Boolean);
            // Include any base-rule categories not already in the global list.
            const allNames = Array.from(new Set([...names, ...((baseRule?.categories) || []).filter(Boolean)])).sort();
            setAvailableCategories(allNames);
        } catch {
            setCeBookmarks([]);
        } finally {
            setLoading(false);
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    useEffect(() => { load(); }, [load]);

    // Entering Groups & Condition: prune group members that were deselected in
    // Pick CEs, and seed a sensible default (one "required" group holding every
    // selected CE, fired by "all of required") the first time through.
    const enterGroupsStep = () => {
        const selected = Array.from(selectedCes);
        let nextGroups = groupList
            .map((g) => ({ ...g, ceIds: g.ceIds.filter((id) => selectedCes.has(id)) }))
            .filter((g) => g.ceIds.length > 0 || g.name.trim() !== '');
        let nextCondition = condition;
        if (nextGroups.length === 0) {
            nextGroups = [{ name: 'required', ceIds: selected }];
            if (!nextCondition.trim()) nextCondition = 'all of required';
        }
        setGroupList(nextGroups);
        setCondition(nextCondition);
        setStep(3);
    };

    const handleGroupsNext = () => {
        const err = validateEditorState(groupList, condition);
        if (err) return showAlertDialog({ title: 'Fix the logic first', message: err, variant: 'info' });
        const grouped = new Set(groupList.flatMap((g) => g.ceIds));
        const ungrouped = Array.from(selectedCes).filter((id) => !grouped.has(id));
        if (ungrouped.length > 0) {
            const names = ungrouped.map((id) => ceBookmarks.find((c) => c.ce_id === id)?.name || `CE_${id}`);
            return showAlertDialog({
                title: 'Ungrouped CEs',
                message: `Every selected CE must belong to a group (a rule only trains on its group members). Ungrouped: ${names.join(', ')}. Add them to a group or deselect them.`,
                variant: 'warning',
            });
        }
        setStep(4);
    };

    const handleCreate = async () => {
        const ceIds = Array.from(selectedCes);
        if (!nameInput.trim()) return showAlertDialog({ title: 'Rule name required', message: 'Provide a rule name.', variant: 'info' });
        if (isEdit && nameInput.trim() === (baseRule?.name || '').trim()) return showAlertDialog({ title: 'Rename required', message: 'Editing creates a NEW draft — give it a name different from the original rule.', variant: 'warning' });
        if (ceIds.length < 2) return showAlertDialog({ title: 'Not enough CEs', message: 'A rule must contain at least 2 Cognitive Elements so the rule set can distinguish between them.', variant: 'warning' });
        if (selectedCategories.size === 0) return showAlertDialog({ title: 'Category required', message: 'Pick at least one category for this rule before creating it.', variant: 'warning' });
        const err = validateEditorState(groupList, condition);
        if (err) return showAlertDialog({ title: 'Fix the logic first', message: err, variant: 'info' });

        // Wire shape: { group_name: [ce_id, ...] }. The backend resolves ids to
        // CE names, validates the condition against the group names, and dedups
        // by structural fingerprint (409/400 with a readable detail on failure).
        const groupsPayload = {};
        groupList.forEach((g) => { groupsPayload[g.name.trim()] = g.ceIds; });

        setCreating(true);
        try {
            const res = await createDraftRuleFromBookmarks(
                nameInput.trim(), groupsPayload, condition.trim(), Array.from(selectedCategories),
            );
            const ruleId = res.data?.rule_id;
            if (!ruleId) throw new Error('Missing rule_id');
            setCreatedRuleId(ruleId);
            setStep(5);
        } catch (e) {
            showAlertDialog({
                title: 'Could not create rule',
                message: e.response?.data?.detail || 'Failed to create rule.',
                variant: 'error',
            });
        } finally {
            setCreating(false);
        }
    };

    // The user kicked off background generation. Mark committed BEFORE we close
    // so the unmount cleanup never discards the now-building rule, then close the
    // modal. The rule stays HIDDEN (is_ready=FALSE) — it only appears (in Drafts)
    // once finalizeWhenReady() reveals it, after the test set finishes.
    const handleStarted = () => {
        committedRef.current = true;
        onClose?.();
    };

    // Reveal the rule — embed + flip is_ready=TRUE. RuleDefaultsStep calls this
    // from its background job ONLY after the rule's test/calibration set is fully
    // built, so the rule never shows half-built in Drafts/Browse.
    const finalizeWhenReady = () => finalizeRule(createdRuleId, Array.from(selectedCes));

    // { gname: [names] } for the live preview / summary.
    const groupsForPreview = {};
    groupList.forEach((g) => {
        groupsForPreview[g.name || '?'] = g.ceIds.map((id) => ceBookmarks.find((c) => c.ce_id === id)?.name || `CE_${id}`);
    });

    return (
        <div>
            <p style={{ margin: '0 0 16px', color: '#94a3b8', fontSize: '0.9rem' }}>
                {isEdit
                    ? <>Editing <strong style={{ color: '#e2e8f0' }}>{baseRule.name}</strong> — adjust its cognitive elements, groups and condition, then give it a new name. This creates a new draft in your Library; the original rule is unchanged.</>
                    : 'Compose a rule from your bookmarked Cognitive Elements. The finished rule lands in your Drafts.'}
            </p>

            <div style={cardStyle}>
                {/* Step indicator */}
                <div style={{ display: 'flex', gap: 4, marginBottom: 18 }}>
                    {STEP_LABELS.map((label, idx) => {
                        const s = idx + 1;
                        const on = step === s, done = step > s;
                        return (
                            <div key={s} style={{ flex: 1, textAlign: 'center' }}>
                                <div style={{
                                    height: 4, borderRadius: 2, marginBottom: 6,
                                    background: done || on ? 'linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%)' : 'rgba(148, 163, 184, 0.18)',
                                    boxShadow: (done || on) ? '0 2px 6px -1px rgba(139, 92, 246, 0.55)' : 'none',
                                    transition: 'background 0.3s',
                                }} />
                                <span style={{ fontSize: '0.72rem', fontWeight: 600, color: on || done ? '#c4b5fd' : '#64748b' }}>{label}</span>
                            </div>
                        );
                    })}
                </div>

                {loading ? (
                    <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading your bookmarked CEs…</div>
                ) : (
                    <>
                        {step === 1 && (() => {
                            const q = ceSearch.trim().toLowerCase();
                            const visibleCes = q
                                ? ceBookmarks.filter((c) => (c.name || '').toLowerCase().includes(q))
                                : ceBookmarks;
                            return (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                                <p style={{ margin: 0, fontSize: '0.88rem', color: '#94a3b8' }}>Select the cognitive elements for your rule.</p>
                                {ceBookmarks.length > 0 && (
                                    <input
                                        className="glass-input"
                                        style={{ marginBottom: 0 }}
                                        placeholder="Search CEs by name…"
                                        value={ceSearch}
                                        onChange={(e) => setCeSearch(e.target.value)}
                                    />
                                )}
                                <div style={{ maxHeight: 320, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
                                    {ceBookmarks.length === 0 ? (
                                        <div style={{ textAlign: 'center', padding: 24, color: '#94a3b8' }}>
                                            No bookmarked CEs. Bookmark some Cognitive Elements from Browse CEs first.
                                        </div>
                                    ) : visibleCes.length === 0 ? (
                                        <div style={{ textAlign: 'center', padding: 24, color: '#94a3b8' }}>
                                            No CEs match “{ceSearch}”.
                                        </div>
                                    ) : visibleCes.map((c) => {
                                        const checked = selectedCes.has(c.ce_id);
                                        return (
                                            <label key={c.ce_id} style={{
                                                display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
                                                borderRadius: 10, cursor: 'pointer', transition: 'all 0.15s',
                                                border: checked ? '2px solid #a78bfa' : '1px solid rgba(148, 163, 184, 0.18)',
                                                background: checked ? 'rgba(139, 92, 246, 0.18)' : 'rgba(15, 23, 42, 0.55)',
                                            }}>
                                                <input type="checkbox" checked={checked} style={{ accentColor: '#a78bfa', width: 16, height: 16 }}
                                                    onChange={(e) => setSelectedCes((prev) => { const n = new Set(prev); e.target.checked ? n.add(c.ce_id) : n.delete(c.ce_id); return n; })}
                                                />
                                                <span style={{ fontWeight: 600, color: '#f1f5f9', fontSize: '0.88rem' }}>{c.name}</span>
                                                <span style={{ fontSize: '0.72rem', color: '#94a3b8', marginLeft: 'auto', background: 'rgba(2, 6, 23, 0.55)', padding: '2px 8px', borderRadius: 6, border: '1px solid rgba(148, 163, 184, 0.14)' }}>{c.category}</span>
                                            </label>
                                        );
                                    })}
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ fontSize: '0.82rem', color: '#94a3b8' }}>{selectedCes.size} selected</span>
                                    <ReactiveButton label="Next" Icon={FiArrowRight}
                                        onClick={() => { if (selectedCes.size === 0) return showAlertDialog({ title: 'Select CEs', message: 'Pick at least one CE.', variant: 'info' }); setStep(2); }}
                                    />
                                </div>
                            </div>
                            );
                        })()}

                        {step === 2 && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                                <RoleLogicGuide />
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <ReactiveButton label="Back" onClick={() => setStep(1)} Icon={FiArrowLeft} />
                                    <ReactiveButton label="Next" Icon={FiArrowRight} onClick={enterGroupsStep} />
                                </div>
                            </div>
                        )}

                        {step === 3 && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                                <p style={{ margin: 0, fontSize: '0.88rem', color: '#94a3b8' }}>
                                    Organize the CEs into named groups, then write the firing condition over the
                                    group names — the preview updates live below.
                                </p>
                                {/* Real-time firing-logic preview for the current groups + condition. */}
                                <RuleLogicPreview title="Firing logic — live preview" groups={groupsForPreview} condition={condition} />
                                <GroupConditionEditor
                                    pool={Array.from(selectedCes).map((id) => ({
                                        ce_id: id,
                                        name: ceBookmarks.find((c) => c.ce_id === id)?.name || `CE_${id}`,
                                    }))}
                                    groupList={groupList}
                                    condition={condition}
                                    onChange={({ groupList: nextGroups, condition: nextCondition }) => {
                                        setGroupList(nextGroups);
                                        setCondition(nextCondition);
                                    }}
                                />
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <ReactiveButton label="Back" onClick={() => setStep(2)} Icon={FiArrowLeft} />
                                    <ReactiveButton label="Next" Icon={FiArrowRight} onClick={handleGroupsNext} />
                                </div>
                            </div>
                        )}

                        {step === 4 && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                                <p style={{ margin: 0, fontSize: '0.88rem', color: '#94a3b8' }}>{isEdit ? 'Give your edited rule a new name.' : 'Give your rule a descriptive name.'}</p>
                                <input value={nameInput} onChange={(e) => setNameInput(e.target.value)} placeholder="e.g., phishing_content_creation" maxLength={120}
                                    style={{ padding: '14px 16px', borderRadius: 12, border: '2px solid rgba(148, 163, 184, 0.22)', background: 'rgba(2, 6, 23, 0.55)', color: '#f1f5f9', fontSize: '0.95rem', fontWeight: 500, outline: 'none', fontFamily: 'inherit' }}
                                />
                                {isEdit && nameInput.trim() === (baseRule?.name || '').trim() && (
                                    <p style={{ margin: 0, fontSize: '0.82rem', color: '#fbbf24' }}>
                                        Editing creates a new draft — the name must differ from “{baseRule.name}”.
                                    </p>
                                )}
                                <p style={{ margin: 0, fontSize: '0.82rem', color: '#64748b' }}>
                                    The rule&apos;s explanation is written automatically from the misuse scenario you confirm in the last step.
                                </p>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                    <p style={{ margin: 0, fontSize: '0.88rem', color: '#94a3b8' }}>
                                        Categories <span style={{ color: '#fca5a5', fontWeight: 700 }}>*</span>
                                        <span style={{ fontWeight: 400 }}> (pick at least one — drives library filtering and search)</span>
                                    </p>
                                    {availableCategories.length === 0 ? (
                                        <p style={{ margin: 0, fontSize: '0.82rem', color: '#94a3b8', fontStyle: 'italic' }}>No categories available yet.</p>
                                    ) : (
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, maxHeight: 180, overflowY: 'auto', padding: 4 }}>
                                            {availableCategories.map((cat) => {
                                                const active = selectedCategories.has(cat);
                                                return (
                                                    <button key={cat} type="button"
                                                        onClick={() => setSelectedCategories((prev) => { const n = new Set(prev); n.has(cat) ? n.delete(cat) : n.add(cat); return n; })}
                                                        style={{
                                                            padding: '6px 12px', borderRadius: 999,
                                                            border: active ? '1px solid rgba(167, 139, 250, 0.75)' : '1px solid rgba(148, 163, 184, 0.22)',
                                                            background: active ? 'rgba(139, 92, 246, 0.22)' : 'rgba(15, 23, 42, 0.55)',
                                                            color: active ? '#ddd6fe' : '#cbd5e1', fontSize: '0.78rem', fontWeight: 600, cursor: 'pointer',
                                                            boxShadow: active ? '0 0 0 3px rgba(139, 92, 246, 0.18)' : 'none',
                                                        }}
                                                    >{cat}</button>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <ReactiveButton label="Back" onClick={() => setStep(3)} Icon={FiArrowLeft} />
                                    <ReactiveButton label={creating ? 'Creating…' : (isEdit ? 'Save as New Rule' : 'Create Rule')} onClick={handleCreate} Icon={FiCheckSquare}
                                        disabled={creating || selectedCategories.size === 0 || !nameInput.trim() || (isEdit && nameInput.trim() === (baseRule?.name || '').trim())} />
                                </div>
                            </div>
                        )}

                        {step === 5 && createdRuleId && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                                {/* Rule summary — name, categories, the groups and the
                                  * condition, mirroring the AI flow's review. */}
                                <div style={{
                                    background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.12), rgba(99, 102, 241, 0.10))',
                                    border: '1px solid rgba(148, 163, 184, 0.20)', borderRadius: 14, padding: 16,
                                    display: 'flex', flexDirection: 'column', gap: 12,
                                }}>
                                    <div>
                                        <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8' }}>Rule</div>
                                        <div style={{ fontWeight: 800, color: '#f8fafc', fontSize: '1.05rem', marginTop: 2 }}>{nameInput || 'Untitled rule'}</div>
                                    </div>
                                    {selectedCategories.size > 0 && (
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                            {Array.from(selectedCategories).map((c) => (
                                                <span key={c} style={{ fontSize: '0.72rem', fontWeight: 700, color: '#ddd6fe', background: 'rgba(139, 92, 246, 0.22)', border: '1px solid rgba(167, 139, 250, 0.45)', borderRadius: 999, padding: '3px 10px' }}>{c}</span>
                                            ))}
                                        </div>
                                    )}
                                    <RuleLogicPreview title={`Firing logic · ${selectedCes.size} cognitive elements`} groups={groupsForPreview} condition={condition} />
                                </div>
                                <RuleDefaultsStep ruleId={createdRuleId} onDone={handleStarted} finalize={finalizeWhenReady} />
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}

// Light inner wrapper — the modal already provides the outer card chrome, so
// this just groups the step indicator + body with a subtle divider.
const cardStyle = {
    padding: '4px 2px 2px',
};
