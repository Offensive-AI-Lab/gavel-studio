// Behavior tests for the BuildRuleFromCEs wizard body.
//
// Now rendered as the BODY of BuildRuleFromCEsModal (no longer a routed page):
// it composes a rule from the user's bookmarked Cognitive Elements across five
// steps (Pick CEs -> Learn the logic -> Groups & Condition -> Name -> Test &
// Calibration), then creates a provisional draft and (via the background tray)
// finalizes it. The rule's logic is named CE GROUPS + a CONDITION expression
// over the group names (v2 grammar) — no roles.
//
// Strategy:
//   * mock '../api' so nothing hits the network — each export returns benign data
//   * mock showAlertDialog (confirmDialog) so validation prompts are observable
//     and never open a real Swal popup (which the console.error fail-fast hates)
//   * stub RuleDefaultsStep so step 5 is deterministic and exposes onDone/finalize

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { TaskTrayProvider } from '../../src/contexts/TaskTrayContext';

// --- mock the API ---
vi.mock('../../src/api', () => ({
    getCEBookmarks: vi.fn(() => Promise.resolve({ data: { bookmarks: [] } })),
    getAllCategories: vi.fn(() => Promise.resolve({ data: [] })),
    createDraftRuleFromBookmarks: vi.fn(() => Promise.resolve({ data: { rule_id: 99 } })),
    finalizeRule: vi.fn(() => Promise.resolve({ data: {} })),
    discardUnreadyRule: vi.fn(() => Promise.resolve({ data: {} })),
}));

// --- mock showAlertDialog so validation prompts are observable + silent ---
vi.mock('../../src/components/ConfirmDialog/confirmDialog', () => ({
    showAlertDialog: vi.fn(() => Promise.resolve()),
}));

// --- stub the final step so step 5 is deterministic ---
// The real component calls onDone immediately (navigate away) and finalize only
// AFTER its background test-set generation finishes. The stub exposes both as
// separate buttons so tests can drive each independently.
vi.mock('../../src/components/RuleDefaults/RuleDefaultsStep', () => ({
    default: ({ ruleId, onDone, finalize }) => (
        <div data-testid="rule-defaults-step">
            <span data-testid="rds-rule-id">{String(ruleId)}</span>
            <button onClick={onDone}>finish-defaults</button>
            <button onClick={finalize}>finalize-defaults</button>
        </div>
    ),
}));

import {
    getCEBookmarks,
    getAllCategories,
    createDraftRuleFromBookmarks,
    finalizeRule,
    discardUnreadyRule,
} from '../../src/api';
import { showAlertDialog } from '../../src/components/ConfirmDialog/confirmDialog';
import BuildRuleFromCEs from '../../src/pages/BuildRuleFromCEs';

const USER = { user_id: 42, email: 'a@b.c' };

const BOOKMARKS = [
    { ce_id: 1, name: 'Alpha CE', category: 'Safety' },
    { ce_id: 2, name: 'Beta CE', category: 'Privacy' },
    { ce_id: 3, name: 'Gamma CE', category: 'Ethics' },
];

const setUser = (u = USER) => {
    sessionStorage.setItem('token', 'fake-token');
    sessionStorage.setItem('user', JSON.stringify(u));
};

// Module-scoped so every test (and the inline renders below) shares one spy.
let onCloseMock = vi.fn();
const renderPage = () => render(
    <TaskTrayProvider>
        <BuildRuleFromCEs onClose={onCloseMock} />
    </TaskTrayProvider>,
);

// Wait for the initial load() effect to resolve (loading -> false).
const waitLoaded = async () =>
    waitFor(() => expect(screen.queryByText(/Loading your bookmarked CEs/i)).not.toBeInTheDocument());

// Select a CE checkbox by visible name on step 1.
const toggleCe = (name) => {
    const label = screen.getByText(name).closest('label');
    fireEvent.click(within(label).getByRole('checkbox'));
};

// The condition text input of the Groups & Condition editor.
const conditionInput = () => screen.getByLabelText('Firing condition');


describe('BuildRuleFromCEs', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        onCloseMock = vi.fn();
        setUser();
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: [] } });
        getAllCategories.mockResolvedValue({ data: [] });
        createDraftRuleFromBookmarks.mockResolvedValue({ data: { rule_id: 99 } });
        finalizeRule.mockResolvedValue({ data: {} });
        discardUnreadyRule.mockResolvedValue({ data: {} });
    });

    // ---- mount / load -------------------------------------------------------

    it('closes the modal when no user is in session storage', async () => {
        sessionStorage.clear();
        renderPage();
        await waitFor(() => expect(onCloseMock).toHaveBeenCalled());
        // Never fetched bookmarks for a missing user.
        expect(getCEBookmarks).not.toHaveBeenCalled();
    });

    it('shows the loading state before data resolves, then clears it', async () => {
        renderPage();
        expect(screen.getByText(/Loading your bookmarked CEs/i)).toBeInTheDocument();
        await waitLoaded();
    });

    it('renders the intro copy and step indicator labels', async () => {
        renderPage();
        await waitLoaded();
        expect(screen.getByText(/Compose a rule from your bookmarked Cognitive Elements/i)).toBeInTheDocument();
        ['Pick CEs', 'Learn the logic', 'Groups & Condition', 'Name', 'Test & Calibration']
            .forEach((l) => expect(screen.getByText(l)).toBeInTheDocument());
    });

    it('fetches bookmarks with the user id and categories on mount', async () => {
        renderPage();
        await waitLoaded();
        expect(getCEBookmarks).toHaveBeenCalledWith(42);
        expect(getAllCategories).toHaveBeenCalled();
    });

    it('shows the empty state when there are no bookmarked CEs', async () => {
        renderPage();
        await waitLoaded();
        expect(screen.getByText(/No bookmarked CEs/i)).toBeInTheDocument();
        expect(screen.getByText(/0 selected/i)).toBeInTheDocument();
    });

    it('renders each bookmarked CE with its name and category', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        expect(screen.getByText('Alpha CE')).toBeInTheDocument();
        expect(screen.getByText('Beta CE')).toBeInTheDocument();
        expect(screen.getByText('Safety')).toBeInTheDocument();
        expect(screen.getByText('Privacy')).toBeInTheDocument();
    });

    it('falls back to an empty list when getCEBookmarks rejects', async () => {
        getCEBookmarks.mockRejectedValue(new Error('boom'));
        renderPage();
        await waitLoaded();
        expect(screen.getByText(/No bookmarked CEs/i)).toBeInTheDocument();
    });

    it('handles a categories response that is not an array', async () => {
        // catRes.data not an array -> availableCategories stays empty (no crash).
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        getAllCategories.mockResolvedValue({ data: { unexpected: true } });
        renderPage();
        await waitLoaded();
        // Proceed to the Name step and confirm the "no categories" hint shows.
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next'));            // -> step 2
        fireEvent.click(screen.getByText('Next'));            // -> step 3
        fireEvent.click(screen.getByText('Next'));            // -> step 4
        expect(screen.getByText(/No categories available yet/i)).toBeInTheDocument();
    });

    // ---- step 1: pick CEs ---------------------------------------------------

    it('warns and stays on step 1 when Next is clicked with nothing selected', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        fireEvent.click(screen.getByText('Next'));
        expect(showAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Select CEs' }),
        );
        // Still on step 1 (the "selected" counter is a step-1-only element).
        expect(screen.getByText(/0 selected/i)).toBeInTheDocument();
    });

    it('updates the selected counter as CEs are checked and unchecked', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        expect(screen.getByText(/1 selected/i)).toBeInTheDocument();
        toggleCe('Beta CE');
        expect(screen.getByText(/2 selected/i)).toBeInTheDocument();
        toggleCe('Alpha CE'); // uncheck
        expect(screen.getByText(/1 selected/i)).toBeInTheDocument();
    });

    // ---- step 2 -> 3: learn the logic / groups & condition ------------------

    it('advances through the guide into Groups & Condition with a seeded default group', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next'));   // -> step 2
        expect(screen.getByText(/How the firing logic is built/i)).toBeInTheDocument();
        fireEvent.click(screen.getByText('Next'));   // -> step 3
        expect(screen.getByText(/Organize the CEs into named groups/i)).toBeInTheDocument();
        // First visit seeds one "required" group with every selected CE and
        // the matching default condition.
        expect(screen.getByLabelText('Group 1 name')).toHaveValue('required');
        expect(conditionInput()).toHaveValue('all of required');
        // Member chips appear in the editor (and again in the live preview).
        expect(screen.getAllByText('Alpha CE').length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText('Beta CE').length).toBeGreaterThanOrEqual(1);
    });

    it('Back from the guide returns to Pick CEs', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        fireEvent.click(screen.getByText('Next'));   // -> step 2
        fireEvent.click(screen.getByText('Back'));   // -> step 1
        expect(screen.getByText(/Select the cognitive elements/i)).toBeInTheDocument();
    });

    it('supports renaming a group, adding a group, and editing the condition', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next'));   // 2
        fireEvent.click(screen.getByText('Next'));   // 3

        fireEvent.change(screen.getByLabelText('Group 1 name'), { target: { value: 'core' } });
        fireEvent.click(screen.getByRole('button', { name: /Add group/ }));
        // The new group gets a fresh default name and its own name input.
        expect(screen.getByLabelText('Group 2 name')).toBeInTheDocument();
        // The condition is generated: it follows the rename and picks up the
        // new group without the user touching the (read-only) text field.
        // (The new group reclaims the name "required", freed by the rename.)
        expect(conditionInput()).toHaveValue('all of core and all of required');
        fireEvent.change(screen.getByLabelText('How much of required must match'),
            { target: { value: 'unused' } });
        expect(conditionInput()).toHaveValue('all of core');
    });

    it('blocks Next when a selected CE belongs to no group', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next'));   // 2
        fireEvent.click(screen.getByText('Next'));   // 3
        // Remove Beta CE from the seeded group -> it becomes ungrouped.
        fireEvent.click(screen.getByRole('button', { name: 'Remove Beta CE from required' }));
        fireEvent.click(screen.getByText('Next'));
        expect(showAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Ungrouped CEs' }),
        );
        // Still on step 3.
        expect(conditionInput()).toBeInTheDocument();
    });

    it('blocks Next on an invalid group name', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next'));   // 2
        fireEvent.click(screen.getByText('Next'));   // 3
        fireEvent.change(screen.getByLabelText('Group 1 name'), { target: { value: 'Bad Name' } });
        fireEvent.click(screen.getByText('Next'));
        expect(showAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Fix the logic first' }),
        );
    });

    // ---- step 4: name + categories -----------------------------------------

    const gotoStep4 = async ({ categories = ['Safety & Harm', 'Privacy'] } = {}) => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        getAllCategories.mockResolvedValue({ data: categories });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next'));   // -> 2
        fireEvent.click(screen.getByText('Next'));   // -> 3
        fireEvent.click(screen.getByText('Next'));   // -> 4
    };

    it('renders the name input and category chips on the Name step', async () => {
        await gotoStep4();
        expect(screen.getByPlaceholderText(/phishing_content_creation/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Safety & Harm' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Privacy' })).toBeInTheDocument();
    });

    it('keeps Create Rule disabled until a name and a category are provided', async () => {
        await gotoStep4();
        const createBtn = () => screen.getByText('Create Rule').closest('button');
        expect(createBtn()).toBeDisabled();

        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'my_rule' } });
        expect(createBtn()).toBeDisabled(); // name alone is not enough

        fireEvent.click(screen.getByRole('button', { name: 'Safety & Harm' }));
        expect(createBtn()).not.toBeDisabled();
    });

    it('toggling a category chip on and off flips the Create button enablement', async () => {
        await gotoStep4();
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        const chip = screen.getByRole('button', { name: 'Privacy' });
        fireEvent.click(chip);
        expect(screen.getByText('Create Rule').closest('button')).not.toBeDisabled();
        fireEvent.click(chip); // deselect
        expect(screen.getByText('Create Rule').closest('button')).toBeDisabled();
    });

    it('Back from the Name step returns to Groups & Condition', async () => {
        await gotoStep4();
        fireEvent.click(screen.getByText('Back'));
        expect(screen.getByText(/Organize the CEs into named groups/i)).toBeInTheDocument();
    });

    it('dedupes and sorts category names from the API', async () => {
        // normalizeCategoryValue strips brackets; duplicates collapse; output sorted.
        await gotoStep4({ categories: ['Zebra', '[Apple]', 'Apple', 'Mango'] });
        const chips = screen.getAllByRole('button')
            .map((b) => b.textContent)
            .filter((t) => ['Apple', 'Mango', 'Zebra'].includes(t));
        expect(chips).toEqual(['Apple', 'Mango', 'Zebra']);
    });

    // ---- handleCreate -------------------------------------------------------

    it('creates the draft rule with trimmed name, groups, condition, and categories then advances to step 5', async () => {
        await gotoStep4();
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: '  my_rule  ' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety & Harm' }));
        fireEvent.click(screen.getByText('Create Rule'));

        await waitFor(() => expect(createDraftRuleFromBookmarks).toHaveBeenCalled());
        const [name, groups, condition, categories] = createDraftRuleFromBookmarks.mock.calls[0];
        expect(name).toBe('my_rule');
        expect(groups).toEqual({ required: [1, 2] });
        expect(condition).toBe('all of required');
        expect(categories).toEqual(['Safety & Harm']);

        await waitFor(() => expect(screen.getByTestId('rule-defaults-step')).toBeInTheDocument());
        expect(screen.getByTestId('rds-rule-id')).toHaveTextContent('99');
    });

    it('encodes a multi-group layout with the edited condition', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        getAllCategories.mockResolvedValue({ data: ['Safety'] });
        renderPage();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next')); // -> 2
        fireEvent.click(screen.getByText('Next')); // -> 3
        // Move Beta CE out of "required" into a new option_1 group.
        fireEvent.click(screen.getByRole('button', { name: 'Remove Beta CE from required' }));
        fireEvent.click(screen.getByRole('button', { name: /Add group/ }));
        fireEvent.change(screen.getByLabelText('Add a CE to option_1'), { target: { value: '2' } });
        fireEvent.click(screen.getByText('Next')); // -> 4

        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety' }));
        fireEvent.click(screen.getByText('Create Rule'));

        await waitFor(() => expect(createDraftRuleFromBookmarks).toHaveBeenCalled());
        const [, groups, condition] = createDraftRuleFromBookmarks.mock.calls[0];
        expect(groups).toEqual({ required: [1], option_1: [2] });
        // Generated by the pickers: both groups hold a single CE, so each reads
        // "all of" (the library's validator rejects "k of" when k == group size).
        expect(condition).toBe('all of required and all of option_1');
    });

    it('shows an error alert and stays on step 4 when create fails', async () => {
        createDraftRuleFromBookmarks.mockRejectedValue({ response: { data: { detail: 'nope' } } });
        await gotoStep4();
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety & Harm' }));
        fireEvent.click(screen.getByText('Create Rule'));

        await waitFor(() => expect(showAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Could not create rule', message: 'nope' }),
        ));
        // Did not advance to step 5.
        expect(screen.queryByTestId('rule-defaults-step')).not.toBeInTheDocument();
    });

    it('shows an error alert when the create response is missing a rule_id', async () => {
        createDraftRuleFromBookmarks.mockResolvedValue({ data: {} });
        await gotoStep4();
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety & Harm' }));
        fireEvent.click(screen.getByText('Create Rule'));
        await waitFor(() => expect(showAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Could not create rule' }),
        ));
        expect(screen.queryByTestId('rule-defaults-step')).not.toBeInTheDocument();
    });

    // ---- step 5: finalize ---------------------------------------------------

    it('closes the modal immediately on start, WITHOUT revealing the rule yet', async () => {
        await gotoStep4();
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety & Harm' }));
        fireEvent.click(screen.getByText('Create Rule'));
        await waitFor(() => expect(screen.getByTestId('rule-defaults-step')).toBeInTheDocument());

        // onDone fires immediately so the modal closes — but the rule is NOT
        // finalized (revealed) yet; that waits for the background test set.
        fireEvent.click(screen.getByText('finish-defaults'));
        await waitFor(() => expect(onCloseMock).toHaveBeenCalled());
        expect(finalizeRule).not.toHaveBeenCalled();
    });

    it('reveals the rule (finalizeRule with id + ce ids) only when finalize runs after the set is ready', async () => {
        await gotoStep4();
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety & Harm' }));
        fireEvent.click(screen.getByText('Create Rule'));
        await waitFor(() => expect(screen.getByTestId('rule-defaults-step')).toBeInTheDocument());

        // The background job calls finalize() after the test set finishes.
        fireEvent.click(screen.getByText('finalize-defaults'));
        await waitFor(() => expect(finalizeRule).toHaveBeenCalledWith(99, [1, 2]));
    });

    // ---- unmount cleanup ----------------------------------------------------

    it('does NOT discard the rule on unmount after it was committed (finished)', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        getAllCategories.mockResolvedValue({ data: ['Safety'] });
        const { unmount } = render(
            <TaskTrayProvider>
                <BuildRuleFromCEs onClose={onCloseMock} />
            </TaskTrayProvider>,
        );
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next')); // 2
        fireEvent.click(screen.getByText('Next')); // 3
        fireEvent.click(screen.getByText('Next')); // 4
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety' }));
        fireEvent.click(screen.getByText('Create Rule'));
        await waitFor(() => expect(screen.getByTestId('rule-defaults-step')).toBeInTheDocument());
        fireEvent.click(screen.getByText('finish-defaults'));
        await waitFor(() => expect(onCloseMock).toHaveBeenCalled());

        discardUnreadyRule.mockClear();
        // Unmounting now must NOT discard — committedRef was set true on finish.
        unmount();
        expect(discardUnreadyRule).not.toHaveBeenCalled();
    });

    it('does not discard anything on unmount when no rule was ever created', async () => {
        const { unmount } = renderPage();
        await waitLoaded();
        unmount();
        expect(discardUnreadyRule).not.toHaveBeenCalled();
    });

    it('discards the provisional rule on unmount (modal closed before kicking off the build)', async () => {
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        getAllCategories.mockResolvedValue({ data: ['Safety'] });
        const { unmount } = render(
            <TaskTrayProvider>
                <BuildRuleFromCEs onClose={onCloseMock} />
            </TaskTrayProvider>,
        );
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next')); // 2
        fireEvent.click(screen.getByText('Next')); // 3
        fireEvent.click(screen.getByText('Next')); // 4
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety' }));
        fireEvent.click(screen.getByText('Create Rule'));
        await waitFor(() => expect(screen.getByTestId('rule-defaults-step')).toBeInTheDocument());

        unmount();
        await waitFor(() => expect(discardUnreadyRule).toHaveBeenCalledWith(99));
    });
});
