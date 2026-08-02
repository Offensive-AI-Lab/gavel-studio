import React, { useEffect, useRef } from 'react';
import {
    FiCpu, FiZap, FiUploadCloud, FiCheck, FiAlertCircle, FiLoader,
    FiX, FiChevronRight, FiChevronUp, FiActivity,
} from 'react-icons/fi';
import { useTaskTray } from '../../contexts/TaskTrayContext';
import './TaskTray.css';

// Icon lookup per task kind. Extend as needed.
const KIND_ICON = {
    rule: FiCpu,
    ce: FiZap,
    publish: FiUploadCloud,
};

const STATUS_LABEL = {
    running: 'Working',
    'needs-input': 'Action needed',
    success: 'Done',
    error: 'Failed',
};

// Higher = louder; the pill borrows the dominant status's color to flag
// what's most important across all running tasks.
const STATUS_PRIORITY = {
    'needs-input': 4,
    error: 3,
    running: 2,
    success: 1,
};

const dominantStatus = (tasks) => {
    if (!tasks.length) return 'running';
    return tasks.reduce(
        (best, t) => (STATUS_PRIORITY[t.status] > STATUS_PRIORITY[best] ? t.status : best),
        'success',
    );
};

// ---------------------------------------------------------------------------
// Inner chip — what the user sees inside the expanded panel. Same layout as
// before, just lifted into its own component for clarity.
// ---------------------------------------------------------------------------

const TaskChip = ({ task, onOpen, onDismiss }) => {
    const KindIcon = KIND_ICON[task.kind] || FiCpu;
    const StatusIcon = (() => {
        if (task.status === 'running') return FiLoader;
        if (task.status === 'needs-input') return FiChevronRight;
        if (task.status === 'success') return FiCheck;
        if (task.status === 'error') return FiAlertCircle;
        return FiLoader;
    })();

    const isClickable = task.status === 'needs-input' || (
        (task.status === 'success' || task.status === 'error') && task.openHandler
    );
    const showDismiss = task.status === 'success' || task.status === 'error';

    return (
        <div
            className={`task-chip task-chip--${task.status} ${isClickable ? 'is-clickable' : ''}`}
            onClick={isClickable ? () => onOpen(task.id) : undefined}
            role={isClickable ? 'button' : undefined}
            tabIndex={isClickable ? 0 : undefined}
            onKeyDown={isClickable ? (e) => { if (e.key === 'Enter' || e.key === ' ') onOpen(task.id); } : undefined}
        >
            <div className="task-chip__kind-icon">
                <KindIcon />
            </div>
            <div className="task-chip__body">
                <div className="task-chip__title">{task.title}</div>
                {task.subtitle && (
                    <div className="task-chip__subtitle">{task.subtitle}</div>
                )}
                <div className="task-chip__status-line">
                    <span className="task-chip__status-dot" aria-hidden="true" />
                    <span>{STATUS_LABEL[task.status]}</span>
                </div>
            </div>
            <div className="task-chip__status-icon">
                <StatusIcon className={task.status === 'running' ? 'is-spinning' : ''} />
            </div>
            {showDismiss && (
                <button
                    className="task-chip__close"
                    onClick={(e) => { e.stopPropagation(); onDismiss(task.id); }}
                    aria-label="Dismiss task"
                >
                    <FiX />
                </button>
            )}
        </div>
    );
};

// ---------------------------------------------------------------------------
// Pill — collapsed state. Always visible whenever there is ≥1 task. Click
// to expand; status color reflects the dominant task state across all chips.
// ---------------------------------------------------------------------------

const TrayPill = ({ tasks, status, expanded, onToggle }) => {
    const runningCount = tasks.filter((t) => t.status === 'running').length;
    const needsInputCount = tasks.filter((t) => t.status === 'needs-input').length;
    const errorCount = tasks.filter((t) => t.status === 'error').length;
    const successCount = tasks.filter((t) => t.status === 'success').length;

    const summary = (() => {
        if (needsInputCount > 0) {
            return `${needsInputCount} need${needsInputCount === 1 ? 's' : ''} you`;
        }
        if (errorCount > 0) {
            return `${errorCount} failed`;
        }
        if (runningCount > 0) {
            return `${runningCount} running`;
        }
        if (successCount > 0) {
            return `${successCount} done`;
        }
        return `${tasks.length} task${tasks.length === 1 ? '' : 's'}`;
    })();

    return (
        <button
            className={`tray-pill tray-pill--${status} ${expanded ? 'is-expanded' : ''}`}
            onClick={onToggle}
            aria-expanded={expanded}
            aria-label={`Background tasks: ${summary}. Click to ${expanded ? 'collapse' : 'expand'}.`}
        >
            <span className="tray-pill__icon-wrap">
                <FiActivity className="tray-pill__icon" />
                <span className="tray-pill__count">{tasks.length}</span>
            </span>
            <span className="tray-pill__summary">{summary}</span>
            <span className="tray-pill__chevron" aria-hidden="true">
                {expanded ? <FiChevronUp /> : <FiChevronRight />}
            </span>
        </button>
    );
};

// ---------------------------------------------------------------------------
// Tray — pill + (when expanded) panel of chips. Click outside collapses.
// ---------------------------------------------------------------------------

export const TaskTray = () => {
    const { tasks, open, dismiss, expanded, setExpanded, toggleExpanded } = useTaskTray();
    const rootRef = useRef(null);

    // Click outside the tray collapses the panel. Skipped when collapsed
    // already so we don't pay for the document listener at idle.
    useEffect(() => {
        if (!expanded) return undefined;
        const handler = (e) => {
            if (rootRef.current && !rootRef.current.contains(e.target)) {
                setExpanded(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [expanded, setExpanded]);

    if (!tasks || tasks.length === 0) return null;

    const status = dominantStatus(tasks);

    return (
        <div
            ref={rootRef}
            className={`task-tray ${expanded ? 'is-expanded' : 'is-collapsed'}`}
            role="region"
            aria-label="Background tasks"
        >
            <TrayPill tasks={tasks} status={status} expanded={expanded} onToggle={toggleExpanded} />

            {expanded && (
                <div className="task-tray__panel">
                    {tasks.map((task) => (
                        <TaskChip key={task.id} task={task} onOpen={open} onDismiss={dismiss} />
                    ))}
                </div>
            )}
        </div>
    );
};

export default TaskTray;
