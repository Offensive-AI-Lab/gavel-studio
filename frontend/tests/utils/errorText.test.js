// The helper that keeps a non-string `detail` out of JSX. React 19 throws
// "Objects are not valid as a React child", so every shape FastAPI can send
// has to come back as text.
import { describe, it, expect } from 'vitest';
import { errorText, detailText } from '../../src/utils/errorText';

const axiosError = (detail, status = 500) => ({
    response: { status, data: { detail } },
});

describe('detailText', () => {
    it('returns a plain string detail', () => {
        expect(detailText(axiosError('model refused'))).toBe('model refused');
    });

    it('reads message out of the missing-key contract object', () => {
        expect(detailText(axiosError({ code: 'OPENAI_KEY_MISSING', message: 'No API key is set.' }, 503)))
            .toBe('No API key is set.');
    });

    it('joins a pydantic validation list', () => {
        const detail = [{ loc: ['body', 'a'], msg: 'field required' }, { msg: 'too long' }];
        expect(detailText(axiosError(detail, 422))).toBe('field required; too long');
    });

    it('returns null when there is no usable detail', () => {
        expect(detailText(axiosError(undefined))).toBeNull();
        expect(detailText(axiosError('   '))).toBeNull();
        expect(detailText(new Error('boom'))).toBeNull();
        expect(detailText(null)).toBeNull();
    });

    it('accepts a bare response body or a plain string', () => {
        expect(detailText({ detail: 'from body' })).toBe('from body');
        expect(detailText('already text')).toBe('already text');
    });
});

describe('errorText', () => {
    it('prefers the backend detail', () => {
        expect(errorText(axiosError('backend said no'), 'fallback')).toBe('backend said no');
    });

    it('falls back to the exception message, then the caller fallback', () => {
        expect(errorText(new Error('network down'), 'fallback')).toBe('network down');
        expect(errorText({}, 'fallback')).toBe('fallback');
    });

    it('never returns an object, whatever detail holds', () => {
        const weird = axiosError({ nested: { deep: true } });
        const out = errorText(weird, 'Something went wrong.');
        expect(typeof out).toBe('string');
        expect(out).toBe('Something went wrong.');
    });
});
