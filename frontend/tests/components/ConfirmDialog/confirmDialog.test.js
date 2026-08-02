// confirmDialog has two surfaces:
//   * escapeHtml — pure utility, easy to pin down
//   * showConfirmDialog / showAlertDialog — wrap Swal.fire with a styled
//     popup. We mock Swal.fire and verify the right options pass through:
//     variant CSS class, focus-cancel for danger, escaped user input, etc.

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock Swal BEFORE importing the SUT.
vi.mock('sweetalert2', () => ({
    default: { fire: vi.fn() },
}));

import Swal from 'sweetalert2';
import { escapeHtml, showConfirmDialog, showAlertDialog } from '../../../src/components/ConfirmDialog/confirmDialog';


describe('escapeHtml', () => {
    it('escapes the five HTML metacharacters', () => {
        expect(escapeHtml('<script>')).toBe('&lt;script&gt;');
        expect(escapeHtml('a & b')).toBe('a &amp; b');
        expect(escapeHtml('"quoted"')).toBe('&quot;quoted&quot;');
        expect(escapeHtml("it's")).toBe('it&#39;s');
    });

    it('returns empty string for null / undefined', () => {
        // Defensive: callers pass user-supplied values that may be missing.
        expect(escapeHtml(null)).toBe('');
        expect(escapeHtml(undefined)).toBe('');
    });

    it('coerces non-strings to string before escaping', () => {
        expect(escapeHtml(42)).toBe('42');
        expect(escapeHtml(true)).toBe('true');
    });

    it('preserves harmless characters', () => {
        expect(escapeHtml('hello world 123')).toBe('hello world 123');
    });
});

describe('showConfirmDialog', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        Swal.fire.mockResolvedValue({ isConfirmed: true });
    });

    it('passes confirm + cancel button text through to Swal', async () => {
        await showConfirmDialog({
            title: 'Sure?',
            message: 'Cannot be undone',
            confirmText: 'Yes',
            cancelText: 'No',
        });
        const opts = Swal.fire.mock.calls[0][0];
        expect(opts.confirmButtonText).toBe('Yes');
        expect(opts.cancelButtonText).toBe('No');
        expect(opts.showCancelButton).toBe(true);
    });

    it('uses the variant for popup + button class names', async () => {
        await showConfirmDialog({ title: 'X', variant: 'danger' });
        const opts = Swal.fire.mock.calls[0][0];
        expect(opts.customClass.popup).toContain('danger');
        expect(opts.customClass.confirmButton).toContain('danger');
    });

    it('focuses the cancel button by default for the danger variant', async () => {
        // The danger variant runs destructive actions — a stray Enter
        // shouldn't auto-confirm.
        await showConfirmDialog({ title: 'X', variant: 'danger' });
        expect(Swal.fire.mock.calls[0][0].focusCancel).toBe(true);
    });

    it('does NOT focus cancel for the info variant', async () => {
        await showConfirmDialog({ title: 'X', variant: 'info' });
        expect(Swal.fire.mock.calls[0][0].focusCancel).toBe(false);
    });

    it('escapes user-supplied title and message', async () => {
        await showConfirmDialog({ title: '<x>', message: '<y>' });
        const opts = Swal.fire.mock.calls[0][0];
        expect(opts.html).toContain('&lt;x&gt;');
        expect(opts.html).toContain('&lt;y&gt;');
        expect(opts.html).not.toContain('<x>');
        expect(opts.html).not.toContain('<y>');
    });

    it('uses messageHtml as-is when provided (caller is responsible for escaping)', async () => {
        await showConfirmDialog({
            title: 'safe title',
            messageHtml: '<strong>bold</strong>',
        });
        const opts = Swal.fire.mock.calls[0][0];
        expect(opts.html).toContain('<strong>bold</strong>');
    });

    it('returns true when the user confirms', async () => {
        Swal.fire.mockResolvedValueOnce({ isConfirmed: true });
        const result = await showConfirmDialog({ title: 'x' });
        expect(result).toBe(true);
    });

    it('returns false when the user cancels or dismisses', async () => {
        Swal.fire.mockResolvedValueOnce({ isConfirmed: false, dismiss: 'cancel' });
        expect(await showConfirmDialog({ title: 'x' })).toBe(false);

        Swal.fire.mockResolvedValueOnce({ isConfirmed: undefined });
        expect(await showConfirmDialog({ title: 'x' })).toBe(false);
    });

    it('falls back to default icon when iconSvg is missing', async () => {
        await showConfirmDialog({ title: 'x', variant: 'warning' });
        // Default warning icon should be embedded in the html.
        expect(Swal.fire.mock.calls[0][0].html).toContain('<svg');
    });

    it('uses caller-supplied iconSvg when given', async () => {
        const iconSvg = '<svg data-test="custom-icon" />';
        await showConfirmDialog({ title: 'x', iconSvg });
        expect(Swal.fire.mock.calls[0][0].html).toContain('data-test="custom-icon"');
    });
});

describe('showAlertDialog', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        Swal.fire.mockResolvedValue({ isConfirmed: true });
    });

    it('does not show a cancel button (single OK)', async () => {
        await showAlertDialog({ title: 'Saved' });
        const opts = Swal.fire.mock.calls[0][0];
        // showCancelButton should not be set (or be false).
        expect(opts.showCancelButton).toBeFalsy();
    });

    it('uses the supplied confirm text', async () => {
        await showAlertDialog({ title: 'Done', confirmText: 'Got it' });
        expect(Swal.fire.mock.calls[0][0].confirmButtonText).toBe('Got it');
    });

    it('escapes user-supplied title and message', async () => {
        await showAlertDialog({ title: '<x>', message: '<y>' });
        const opts = Swal.fire.mock.calls[0][0];
        expect(opts.html).toContain('&lt;x&gt;');
        expect(opts.html).toContain('&lt;y&gt;');
    });
});
