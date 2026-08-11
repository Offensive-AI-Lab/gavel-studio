// Contract tests for src/api.js against the gavel-rules-era backend:
//   * publish-era exports are GONE (library contributions moved to PRs),
//   * syncLibrary gates the gavel:libraryChanged dispatch on the kept
//     SyncResponse delta counters (no neutral_synced key anymore),
//   * the rule-editor endpoints speak the {groups: {name: [ce_id]}, condition}
//     v2 shape at the documented URLs.
// The axios instance is mocked; no network.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const mocks = vi.hoisted(() => {
    const instance = {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
        // api.js registers a response interceptor at import time (the
        // missing-OpenAI-key modal hook) — see tests/api.openAiKey.test.js.
        interceptors: { response: { use: vi.fn() } },
    };
    return { instance };
});

vi.mock('axios', () => ({
    default: { create: vi.fn(() => mocks.instance) },
}));

import * as api from '../src/api';

const listenLibraryChanged = () => {
    const spy = vi.fn();
    window.addEventListener('gavel:libraryChanged', spy);
    return spy;
};

beforeEach(() => {
    mocks.instance.get.mockReset().mockResolvedValue({ data: {} });
    mocks.instance.post.mockReset().mockResolvedValue({ data: {} });
    mocks.instance.put.mockReset().mockResolvedValue({ data: {} });
    mocks.instance.delete.mockReset().mockResolvedValue({ data: {} });
});

afterEach(() => {
    // Listeners are added per-test with fresh spies; drop any leftovers.
    vi.restoreAllMocks();
});

describe('api.js — publish machinery removed', () => {
    it('no longer exports the publish/adopt/rename-era functions', () => {
        [
            'publishCE',
            'publishRule',
            'publishRuleSet',
            'adoptPublicCE',
            'checkLibraryName',
            'getPublicRecord',
            'cleanupLocalDrafts',
            'createPublicRule',
            'createManualRule',
            'createAIRule',
        ].forEach((name) => {
            expect(api[name], `${name} should be removed`).toBeUndefined();
        });
    });
});

describe('api.js — syncLibrary change gate', () => {
    const zeroes = {
        changed: false,
        rules_added: 0,
        ces_added: 0,
        rule_sets_added: 0,
        rules_refreshed: 0,
        ces_refreshed: 0,
        rule_sets_refreshed: 0,
        categories_synced: 0,
    };

    it('does NOT dispatch gavel:libraryChanged on a no-op pull', async () => {
        mocks.instance.get.mockResolvedValue({ data: { ...zeroes } });
        const spy = listenLibraryChanged();
        await api.syncLibrary();
        expect(mocks.instance.get).toHaveBeenCalledWith('/library/sync', { params: {} });
        expect(spy).not.toHaveBeenCalled();
        window.removeEventListener('gavel:libraryChanged', spy);
    });

    it.each([
        'rules_added',
        'ces_added',
        'rule_sets_added',
        'rules_refreshed',
        'ces_refreshed',
        'rule_sets_refreshed',
        'categories_synced',
    ])('dispatches gavel:libraryChanged when %s > 0', async (field) => {
        mocks.instance.get.mockResolvedValue({ data: { ...zeroes, [field]: 2 } });
        const spy = listenLibraryChanged();
        await api.syncLibrary();
        expect(spy).toHaveBeenCalledTimes(1);
        window.removeEventListener('gavel:libraryChanged', spy);
    });

    it('ignores the dead neutral_synced key (dropped from SyncResponse)', async () => {
        mocks.instance.get.mockResolvedValue({ data: { ...zeroes, neutral_synced: 5 } });
        const spy = listenLibraryChanged();
        await api.syncLibrary();
        expect(spy).not.toHaveBeenCalled();
        window.removeEventListener('gavel:libraryChanged', spy);
    });

    it('passes force=true through as a query param', async () => {
        mocks.instance.get.mockResolvedValue({ data: { ...zeroes } });
        await api.syncLibrary({ force: true });
        expect(mocks.instance.get).toHaveBeenCalledWith('/library/sync', { params: { force: true } });
    });
});

describe('api.js — cancelTraining', () => {
    it('posts to /classifiers/{id}/training/cancel', async () => {
        await api.cancelTraining(5);
        expect(mocks.instance.post).toHaveBeenCalledWith('/classifiers/5/training/cancel');
    });

    it('dispatches gavel:libraryChanged (the row status changed)', async () => {
        const spy = listenLibraryChanged();
        await api.cancelTraining(5);
        expect(spy).toHaveBeenCalledTimes(1);
        window.removeEventListener('gavel:libraryChanged', spy);
    });

    it('propagates the 409 (run already over) and fires no change event', async () => {
        const err = { response: { status: 409, data: { detail: 'Rule set is not training' } } };
        mocks.instance.post.mockRejectedValue(err);
        const spy = listenLibraryChanged();
        await expect(api.cancelTraining(5)).rejects.toBe(err);
        expect(spy).not.toHaveBeenCalled();
        window.removeEventListener('gavel:libraryChanged', spy);
    });
});

describe('api.js — v2 rule-editor endpoints ({groups, condition})', () => {
    it('createDraftRuleFromBookmarks posts name/groups/condition/categories to /ai/rules/from-bookmarked-ce', async () => {
        await api.createDraftRuleFromBookmarks(
            'My Rule',
            { required: [1, 2], option_1: [3] },
            'all of required and 1 of option_1',
            [7],
        );
        expect(mocks.instance.post).toHaveBeenCalledWith('/ai/rules/from-bookmarked-ce', {
            name: 'My Rule',
            groups: { required: [1, 2], option_1: [3] },
            condition: 'all of required and 1 of option_1',
            categories: [7],
            description: '',
        });
    });

    it('saveEditedRule posts groups/condition/new_name/add_bookmark to /rules/setup/{id}/save-edited', async () => {
        await api.saveEditedRule(9, {
            groups: { required: [1] },
            condition: '1 of required',
            new_name: 'Fork',
        });
        expect(mocks.instance.post).toHaveBeenCalledWith('/rules/setup/9/save-edited', {
            groups: { required: [1] },
            condition: '1 of required',
            new_name: 'Fork',
            add_bookmark: false,
        });
    });

    it('checkRuleDuplicate posts the probe shape to /rules/check-duplicate', async () => {
        await api.checkRuleDuplicate({
            groups: { required: [1] },
            condition: 'all of required',
            classifier_id: 5,
            exclude_setup_id: 9,
        });
        expect(mocks.instance.post).toHaveBeenCalledWith('/rules/check-duplicate', {
            groups: { required: [1] },
            condition: 'all of required',
            classifier_id: 5,
            exclude_setup_id: 9,
        });
    });

    it('updateRuleLogic puts groups/condition to /rules/setup/{id}', async () => {
        await api.updateRuleLogic(4, { groups: { g: [1] }, condition: 'all of g' });
        expect(mocks.instance.put).toHaveBeenCalledWith('/rules/setup/4', {
            groups: { g: [1] },
            condition: 'all of g',
        });
    });
});
