import Swal from 'sweetalert2';
import './ConfirmDialog.css';

/**
 * Show a polished confirm dialog and resolve to true if the user confirms,
 * false if they cancel / dismiss. Replaces ad-hoc Swal.fire calls so the
 * design stays consistent across the app.
 *
 * Variants:
 *   'danger'  — red accent, destructive action (default for delete)
 *   'warning' — amber accent
 *   'info'    — indigo accent
 *
 * Pass `iconSvg` as a custom inline SVG string to override the default
 * variant icon.
 */
export const showConfirmDialog = async ({
    title,
    message,
    messageHtml,        // pre-escaped HTML body — use this when `message` isn't expressive enough (lists, emphasis, multi-paragraph). Caller is responsible for escaping any user-supplied substrings.
    confirmText = 'Confirm',
    cancelText = 'Cancel',
    variant = 'info',
    iconSvg,
} = {}) => {
    let bodyHtml = '';
    if (messageHtml) {
        bodyHtml = `<div class="gd-confirm__message">${messageHtml}</div>`;
    } else if (message) {
        bodyHtml = `<div class="gd-confirm__message">${escapeHtml(message)}</div>`;
    }

    const html = `
        <div class="gd-confirm">
            <div class="gd-confirm__icon gd-confirm__icon--${variant}">
                ${iconSvg || DEFAULT_ICONS[variant] || DEFAULT_ICONS.info}
            </div>
            <div class="gd-confirm__title">${escapeHtml(title || '')}</div>
            ${bodyHtml}
        </div>
    `;

    const result = await Swal.fire({
        html,
        showCancelButton: true,
        confirmButtonText: confirmText,
        cancelButtonText: cancelText,
        showCloseButton: true,
        customClass: {
            popup: `gd-confirm-popup gd-confirm-popup--${variant}`,
            confirmButton: `gd-confirm-btn gd-confirm-btn--${variant}`,
            cancelButton: 'gd-confirm-btn gd-confirm-btn--cancel',
        },
        buttonsStyling: false,
        focusCancel: variant === 'danger',
        // Suppress all of Swal's built-in chrome — our html supplies the icon and title.
    });

    return Boolean(result.isConfirmed);
};

export const escapeHtml = (s) => String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

const DEFAULT_ICONS = {
    danger: `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path>
            <path d="M10 11v6"></path>
            <path d="M14 11v6"></path>
            <path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"></path>
        </svg>
    `,
    warning: `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
            <line x1="12" y1="9" x2="12" y2="13"></line>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
    `,
    info: `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="16" x2="12" y2="12"></line>
            <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
    `,
    success: `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
    `,
    error: `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="15" y1="9" x2="9" y2="15"></line>
            <line x1="9" y1="9" x2="15" y2="15"></line>
        </svg>
    `,
};


/**
 * Single-button alert dialog. Same visual language as showConfirmDialog,
 * but with only an OK button — used for status notifications like
 * "Saved", "Bookmark added", "Deleted", etc. Variants:
 *   'success' — green
 *   'info'    — indigo
 *   'warning' — amber
 *   'error'   — red
 *
 * Resolves when the user dismisses (any path), so callers can `await` it
 * if they want to know when the dialog closed.
 */
export const showAlertDialog = async ({
    title,
    message,
    messageHtml,
    confirmText = 'OK',
    variant = 'info',
    iconSvg,
} = {}) => {
    let bodyHtml = '';
    if (messageHtml) {
        bodyHtml = `<div class="gd-confirm__message">${messageHtml}</div>`;
    } else if (message) {
        bodyHtml = `<div class="gd-confirm__message">${escapeHtml(message)}</div>`;
    }

    const html = `
        <div class="gd-confirm">
            <div class="gd-confirm__icon gd-confirm__icon--${variant}">
                ${iconSvg || DEFAULT_ICONS[variant] || DEFAULT_ICONS.info}
            </div>
            <div class="gd-confirm__title">${escapeHtml(title || '')}</div>
            ${bodyHtml}
        </div>
    `;

    await Swal.fire({
        html,
        confirmButtonText: confirmText,
        showCloseButton: true,
        customClass: {
            popup: `gd-confirm-popup gd-confirm-popup--${variant}`,
            confirmButton: `gd-confirm-btn gd-confirm-btn--${variant}`,
        },
        buttonsStyling: false,
    });
};


/**
 * Non-dismissible loading dialog with a spinner and a message. Used for
 * short-lived "Processing..." / "Creating..." / "Uploading..." overlays
 * while an API call is in flight. Same visual language as the other
 * dialogs — replaces raw Swal.fire({ didOpen: showLoading }) usages so
 * the polished design is consistent everywhere.
 *
 * Returns a `close()` function. Callers must invoke it (typically in a
 * try/finally) to dismiss the dialog when the operation completes.
 *
 * Example:
 *   const close = showLoadingDialog({ title: 'Creating model', message: 'Uploading file...' });
 *   try {
 *       await api.post(...);
 *   } finally {
 *       close();
 *   }
 */
export const showLoadingDialog = ({
    title = 'Processing',
    message,
    variant = 'info',
} = {}) => {
    const bodyHtml = message
        ? `<div class="gd-confirm__message" data-gd-message>${escapeHtml(message)}</div>`
        : '<div class="gd-confirm__message" data-gd-message></div>';
    const spinnerSvg = `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="animation: gd-spin 1.1s linear infinite">
            <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
        </svg>
    `;
    const html = `
        <div class="gd-confirm">
            <div class="gd-confirm__icon gd-confirm__icon--${variant}">
                ${spinnerSvg}
            </div>
            <div class="gd-confirm__title">${escapeHtml(title)}</div>
            ${bodyHtml}
        </div>
    `;

    Swal.fire({
        html,
        showConfirmButton: false,
        showCloseButton: false,
        allowOutsideClick: false,
        allowEscapeKey: false,
        customClass: {
            popup: `gd-confirm-popup gd-confirm-popup--${variant}`,
        },
        buttonsStyling: false,
    });

    // Return the close() function, with an `update(message)` helper attached so
    // long-running callers (e.g. a file upload) can live-update the text — e.g.
    // an upload percentage — without reopening the dialog. Backward compatible:
    // existing callers still just call close().
    const close = () => Swal.close();
    close.update = (newMessage) => {
        const el = Swal.getHtmlContainer()?.querySelector('[data-gd-message]');
        if (el) el.textContent = newMessage ?? '';
    };
    return close;
};

export default showConfirmDialog;
