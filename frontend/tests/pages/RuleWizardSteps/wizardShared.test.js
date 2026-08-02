// Unit tests for wizardShared.js — the pure helper + style module shared by
// every step component in the rule wizard. It imports nothing (no React, no
// ../api, no routing), so these are plain deterministic unit tests with no
// mocking, rendering, or network involved.

import { describe, it, expect, vi } from 'vitest';
import {
    getStepState,
    startStep,
    completeStep,
    errorStep,
    card,
    muted,
    primaryBtn,
    secondaryBtn,
    fieldStyle,
    successBanner,
    errorBanner,
} from '../../../src/pages/RuleWizardSteps/wizardShared';

describe('getStepState', () => {
    it('returns the persisted step state when present', () => {
        const stepState = { status: 'completed', data: { foo: 'bar' } };
        const run = { steps: { step1: stepState } };
        expect(getStepState(run, 'step1')).toBe(stepState);
    });

    it('returns the default pending state when the step is absent', () => {
        const run = { steps: { other: { status: 'completed', data: {} } } };
        expect(getStepState(run, 'missing')).toEqual({ status: 'pending', data: {} });
    });

    it('returns the default when run.steps is undefined', () => {
        const run = {};
        expect(getStepState(run, 'step1')).toEqual({ status: 'pending', data: {} });
    });

    it('returns the default when run is null', () => {
        expect(getStepState(null, 'step1')).toEqual({ status: 'pending', data: {} });
    });

    it('returns the default when run is undefined', () => {
        expect(getStepState(undefined, 'step1')).toEqual({ status: 'pending', data: {} });
    });

    it('returns the default when the stored value for the step is falsy', () => {
        // run.steps exists but the key maps to a falsy value (e.g. 0/null).
        const run = { steps: { step1: null } };
        expect(getStepState(run, 'step1')).toEqual({ status: 'pending', data: {} });
    });

    it('returns a brand-new default object each call (not a shared singleton)', () => {
        const a = getStepState({}, 'x');
        const b = getStepState({}, 'y');
        expect(a).not.toBe(b);
        expect(a).toEqual(b);
    });

    it('does not mutate the run object', () => {
        const run = { steps: { step1: { status: 'pending', data: {} } } };
        const snapshot = JSON.parse(JSON.stringify(run));
        getStepState(run, 'step1');
        getStepState(run, 'missing');
        expect(run).toEqual(snapshot);
    });
});

describe('startStep', () => {
    it('calls onPatchStep with in_progress status and the provided data', () => {
        const onPatchStep = vi.fn();
        const data = { prompt: 'hello' };
        startStep(onPatchStep, 'step2', data);
        expect(onPatchStep).toHaveBeenCalledTimes(1);
        expect(onPatchStep).toHaveBeenCalledWith('step2', { status: 'in_progress', data });
    });

    it('passes the same data reference through (no copying)', () => {
        const onPatchStep = vi.fn();
        const data = { a: 1 };
        startStep(onPatchStep, 's', data);
        expect(onPatchStep.mock.calls[0][1].data).toBe(data);
    });

    it('returns whatever onPatchStep returns', () => {
        const onPatchStep = vi.fn(() => 'result-token');
        expect(startStep(onPatchStep, 's', {})).toBe('result-token');
    });

    it('forwards undefined data unchanged', () => {
        const onPatchStep = vi.fn();
        startStep(onPatchStep, 's', undefined);
        expect(onPatchStep).toHaveBeenCalledWith('s', { status: 'in_progress', data: undefined });
    });
});

describe('completeStep', () => {
    it('calls onPatchStep with completed status and the provided data', () => {
        const onPatchStep = vi.fn();
        const data = { result: 42 };
        completeStep(onPatchStep, 'step3', data);
        expect(onPatchStep).toHaveBeenCalledTimes(1);
        expect(onPatchStep).toHaveBeenCalledWith('step3', { status: 'completed', data });
    });

    it('returns whatever onPatchStep returns', () => {
        const onPatchStep = vi.fn(() => Promise.resolve('done'));
        const returned = completeStep(onPatchStep, 's', {});
        expect(returned).toBeInstanceOf(Promise);
    });

    it('passes the same data reference through', () => {
        const onPatchStep = vi.fn();
        const data = { x: 'y' };
        completeStep(onPatchStep, 's', data);
        expect(onPatchStep.mock.calls[0][1].data).toBe(data);
    });
});

describe('errorStep', () => {
    it('calls onPatchStep with error status and merges the message into data', () => {
        const onPatchStep = vi.fn();
        errorStep(onPatchStep, 'step4', 'boom', { keep: true });
        expect(onPatchStep).toHaveBeenCalledTimes(1);
        expect(onPatchStep).toHaveBeenCalledWith('step4', {
            status: 'error',
            data: { keep: true, error: 'boom' },
        });
    });

    it('defaults data to an empty object when omitted', () => {
        const onPatchStep = vi.fn();
        errorStep(onPatchStep, 'step4', 'kaboom');
        expect(onPatchStep).toHaveBeenCalledWith('step4', {
            status: 'error',
            data: { error: 'kaboom' },
        });
    });

    it('error message wins over an error key already present in data', () => {
        const onPatchStep = vi.fn();
        errorStep(onPatchStep, 's', 'new-error', { error: 'old-error', other: 1 });
        expect(onPatchStep.mock.calls[0][1].data).toEqual({ error: 'new-error', other: 1 });
    });

    it('does not mutate the passed-in data object (spread copy)', () => {
        const onPatchStep = vi.fn();
        const data = { keep: 'me' };
        errorStep(onPatchStep, 's', 'err', data);
        // Original untouched...
        expect(data).toEqual({ keep: 'me' });
        // ...and the patched data is a new object.
        expect(onPatchStep.mock.calls[0][1].data).not.toBe(data);
    });

    it('handles undefined / empty error messages', () => {
        const onPatchStep = vi.fn();
        errorStep(onPatchStep, 's', undefined);
        expect(onPatchStep.mock.calls[0][1].data).toEqual({ error: undefined });
    });

    it('returns whatever onPatchStep returns', () => {
        const onPatchStep = vi.fn(() => 'r');
        expect(errorStep(onPatchStep, 's', 'e')).toBe('r');
    });
});

describe('style constants', () => {
    it('card has the expected shape', () => {
        expect(card).toMatchObject({
            borderRadius: 16,
            padding: 20,
            marginBottom: 14,
        });
        expect(typeof card.background).toBe('string');
        expect(typeof card.border).toBe('string');
        expect(typeof card.boxShadow).toBe('string');
    });

    it('muted exposes a color', () => {
        expect(muted).toEqual({ color: '#94a3b8' });
    });

    it('primaryBtn is a clickable gradient button style', () => {
        expect(primaryBtn).toMatchObject({
            borderRadius: 10,
            color: '#fff',
            fontWeight: 600,
            border: 'none',
            cursor: 'pointer',
        });
        expect(primaryBtn.background).toContain('gradient');
    });

    it('secondaryBtn is a clickable bordered button style', () => {
        expect(secondaryBtn).toMatchObject({
            borderRadius: 10,
            color: '#cbd5e1',
            fontWeight: 600,
            cursor: 'pointer',
        });
        expect(secondaryBtn.border).toContain('1px solid');
    });

    it('fieldStyle fills width and uses inherited font', () => {
        expect(fieldStyle).toMatchObject({
            width: '100%',
            borderRadius: 8,
            color: '#e2e8f0',
            fontFamily: 'inherit',
            fontSize: 14,
        });
    });

    it('successBanner is a flex banner with green accents', () => {
        expect(successBanner).toMatchObject({
            borderRadius: 8,
            fontSize: 13,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            color: '#6ee7b7',
        });
    });

    it('errorBanner is a flex banner with red accents', () => {
        expect(errorBanner).toMatchObject({
            borderRadius: 8,
            fontSize: 13,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            color: '#fca5a5',
        });
    });

    it('success and error banners differ only in their accent colors', () => {
        expect(successBanner.color).not.toBe(errorBanner.color);
        expect(successBanner.display).toBe(errorBanner.display);
        expect(successBanner.borderRadius).toBe(errorBanner.borderRadius);
    });

    it('exports are all defined', () => {
        for (const style of [card, muted, primaryBtn, secondaryBtn, fieldStyle, successBanner, errorBanner]) {
            expect(style).toBeTypeOf('object');
            expect(style).not.toBeNull();
        }
    });
});
