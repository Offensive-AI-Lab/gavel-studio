// ComputeBadge — "which GPU am I on" indicator.
//
// Shows the compute backend that WOULD run a given workload right now (training
// by default), so on the training page you can see at a glance what you're
// committed to: a remote GPU or this machine. Polls
// /compute/status (public, cheap) and never breaks the page — on error it just
// hides itself.
//
// Reused across workloads via the `workload` prop ('training' | 'inference' |
// 'realtime'); the visible label matches the wording the realtime monitor uses.

import { useEffect, useState } from 'react';
import { FiServer, FiCpu, FiHardDrive } from 'react-icons/fi';
import { getComputeStatus } from '../../api';

const REFRESH_MS = 30000;

// provider name (backend) -> how we present it
const PROVIDER_META = {
    remote_worker: { label: 'Remote GPU',   Icon: FiServer,   color: '#6ee7b7', bg: 'rgba(16, 185, 129, 0.15)', border: 'rgba(52, 211, 153, 0.40)' },
    local:         { label: 'This machine',  Icon: FiHardDrive, color: '#cbd5e1', bg: 'rgba(148, 163, 184, 0.14)', border: 'rgba(148, 163, 184, 0.35)' },
};

const ComputeBadge = ({ workload = 'training', prefix = 'Training on' }) => {
    const [info, setInfo] = useState(null);   // { provider, accelerator, detail }
    const [failed, setFailed] = useState(false);

    useEffect(() => {
        let alive = true;
        const tick = () => {
            getComputeStatus()
                .then((res) => {
                    if (!alive) return;
                    const w = res?.data?.workloads?.[workload];
                    if (w && w.provider) { setInfo(w); setFailed(false); }
                    else setFailed(true);
                })
                .catch(() => { if (alive) setFailed(true); });
        };
        tick();
        const id = setInterval(tick, REFRESH_MS);
        return () => { alive = false; clearInterval(id); };
    }, [workload]);

    // Don't render anything until we know — and never let a failed probe break
    // the page (the indicator is informational, not load-bearing).
    if (failed && !info) return null;
    if (!info) return null;

    const meta = PROVIDER_META[info.provider] || {
        label: info.provider, Icon: FiCpu, color: '#cbd5e1',
        bg: 'rgba(148, 163, 184, 0.14)', border: 'rgba(148, 163, 184, 0.35)',
    };
    const accel = (info.accelerator || '').toUpperCase();
    const title = [info.detail, accel && `Accelerator: ${accel}`]
        .filter(Boolean).join(' • ') || 'Active compute backend';
    const { Icon } = meta;

    return (
        <span
            title={title}
            aria-label={`${prefix}: ${meta.label}`}
            style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '7px',
                padding: '5px 12px',
                borderRadius: '999px',
                background: meta.bg,
                border: `1px solid ${meta.border}`,
                color: meta.color,
                fontSize: '0.8rem',
                fontWeight: 600,
                whiteSpace: 'nowrap',
            }}
        >
            <Icon size={14} />
            <span style={{ opacity: 0.85 }}>{prefix}:</span>
            <strong style={{ fontWeight: 800 }}>{meta.label}</strong>
            {/* Show the real accelerator (CUDA / MPS / CPU) so a no-GPU machine
              * is clearly marked. Hide only "REMOTE", which is meaningless here. */}
            {accel && accel !== 'REMOTE' && (
                <span style={{ opacity: 0.75, fontWeight: 600 }}>· {accel}</span>
            )}
        </span>
    );
};

export default ComputeBadge;
