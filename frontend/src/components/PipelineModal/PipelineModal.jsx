import React from 'react';

const PipelineModal = ({
  icon,
  title,
  subtitle,
  children,
  showSpinner = true,
  variant = 'dark',
}) => {
  const isDark = variant === 'dark';

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '14px',
        alignItems: 'center',
        color: isDark ? '#e2e8f0' : '#111827',
        textAlign: 'center',
      }}
    >
      {icon ? (
        <div
          style={{
            width: 62,
            height: 62,
            borderRadius: '18px',
            display: 'grid',
            placeItems: 'center',
            background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(37,99,235,0.08)',
            fontSize: '28px',
          }}
        >
          {icon}
        </div>
      ) : null}
      <div style={{ fontSize: '26px', fontWeight: 700, lineHeight: 1.2 }}>{title}</div>
      {subtitle ? (
        <div style={{ color: isDark ? '#d1d5db' : '#4b5563', fontSize: '16px', lineHeight: 1.5, maxWidth: '460px' }}>
          {subtitle}
        </div>
      ) : null}
      {showSpinner ? (
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', justifyContent: 'center', marginTop: '6px' }}>
          <div
            className="swal2-loader"
            style={{ display: 'block', borderColor: isDark ? 'rgba(255,255,255,0.15)' : 'rgba(37,99,235,0.12)', borderTopColor: isDark ? '#38bdf8' : '#2563eb' }}
          />
        </div>
      ) : null}
      {children ? <div style={{ width: '100%', marginTop: '4px', textAlign: 'left' }}>{children}</div> : null}
    </div>
  );
};

export default PipelineModal;
