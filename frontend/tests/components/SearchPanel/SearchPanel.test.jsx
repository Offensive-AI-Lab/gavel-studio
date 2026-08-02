// Behavior tests for the SearchPanel presentational component.
//
// SearchPanel is a pure, controlled component: it has no API calls, no
// routing, no contexts. Every behavior is driven by props and callbacks.
// These tests exercise each prop, branch, and user interaction.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import SearchPanel from '../../../src/components/SearchPanel/SearchPanel';

// A factory for the full set of callback props so each test starts fresh.
const makeProps = (overrides = {}) => ({
    query: '',
    onQueryChange: vi.fn(),
    categories: [],
    onCategoriesChange: vi.fn(),
    onTopKChange: vi.fn(),
    onSearch: vi.fn(),
    onReset: vi.fn(),
    loading: false,
    assetTypes: ['rule', 'ce'],
    onAssetTypesChange: vi.fn(),
    ...overrides,
});

describe('SearchPanel', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('initial render', () => {
        it('renders the default placeholder and core controls', () => {
            const props = makeProps();
            render(<SearchPanel {...props} />);

            expect(
                screen.getByPlaceholderText('Search rules, cognitive elements, and more...')
            ).toBeInTheDocument();
            // Search button + reset button visible.
            expect(screen.getByRole('button', { name: /Search/i })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: /Reset All/i })).toBeInTheDocument();
            // Asset type filter visible by default.
            expect(screen.getByText('Asset Type')).toBeInTheDocument();
            expect(screen.getByRole('button', { name: /Rules/i })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: /CEs/i })).toBeInTheDocument();
        });

        it('uses a custom searchPlaceholder when provided', () => {
            const props = makeProps();
            render(<SearchPanel {...props} searchPlaceholder="Find stuff" />);
            expect(screen.getByPlaceholderText('Find stuff')).toBeInTheDocument();
        });

        it('reflects the controlled query value in the input', () => {
            const props = makeProps({ query: 'hello' });
            render(<SearchPanel {...props} />);
            expect(screen.getByDisplayValue('hello')).toBeInTheDocument();
        });
    });

    describe('query input behavior', () => {
        it('calls onQueryChange with the typed value', () => {
            const props = makeProps();
            render(<SearchPanel {...props} />);
            const input = screen.getByPlaceholderText(/Search rules/i);
            fireEvent.change(input, { target: { value: 'abc' } });
            expect(props.onQueryChange).toHaveBeenCalledWith('abc');
        });

        it('does not render the clear button when query is empty', () => {
            const props = makeProps({ query: '' });
            render(<SearchPanel {...props} />);
            expect(screen.queryByTitle('Clear search')).not.toBeInTheDocument();
        });

        it('renders a clear button when query is non-empty and clears on click', () => {
            const props = makeProps({ query: 'something' });
            render(<SearchPanel {...props} />);
            const clearBtn = screen.getByTitle('Clear search');
            expect(clearBtn).toBeInTheDocument();
            fireEvent.click(clearBtn);
            expect(props.onQueryChange).toHaveBeenCalledWith('');
        });

        it('triggers onSearch when Enter is pressed', () => {
            const props = makeProps({ query: 'x' });
            render(<SearchPanel {...props} />);
            const input = screen.getByPlaceholderText(/Search rules/i);
            fireEvent.keyDown(input, { key: 'Enter' });
            expect(props.onSearch).toHaveBeenCalledTimes(1);
        });

        it('does not trigger onSearch for non-Enter keys', () => {
            const props = makeProps({ query: 'x' });
            render(<SearchPanel {...props} />);
            const input = screen.getByPlaceholderText(/Search rules/i);
            fireEvent.keyDown(input, { key: 'a' });
            expect(props.onSearch).not.toHaveBeenCalled();
        });
    });

    describe('search button state', () => {
        it('is disabled when query is empty and allowEmptyQuery is false', () => {
            const props = makeProps({ query: '' });
            render(<SearchPanel {...props} />);
            expect(screen.getByRole('button', { name: /Search/i })).toBeDisabled();
        });

        it('is disabled when query is only whitespace', () => {
            const props = makeProps({ query: '   ' });
            const { container } = render(<SearchPanel {...props} />);
            expect(container.querySelector('.search-btn')).toBeDisabled();
        });

        it('is enabled when query has content', () => {
            const props = makeProps({ query: 'real query' });
            const { container } = render(<SearchPanel {...props} />);
            const btn = container.querySelector('.search-btn');
            expect(btn).not.toBeDisabled();
            fireEvent.click(btn);
            expect(props.onSearch).toHaveBeenCalledTimes(1);
        });

        it('is enabled with an empty query when allowEmptyQuery is true', () => {
            const props = makeProps({ query: '' });
            render(<SearchPanel {...props} allowEmptyQuery={true} />);
            expect(screen.getByRole('button', { name: /Search/i })).not.toBeDisabled();
        });

        it('shows a loading spinner and is disabled while loading', () => {
            const props = makeProps({ query: 'x', loading: true });
            render(<SearchPanel {...props} />);
            const btn = screen.getByRole('button', { name: /Searching/i });
            expect(btn).toBeDisabled();
            expect(screen.getByText(/Searching/i)).toBeInTheDocument();
        });
    });

    describe('asset type filter', () => {
        it('hides the asset type filter when showAssetTypeFilter is false', () => {
            const props = makeProps();
            render(<SearchPanel {...props} showAssetTypeFilter={false} />);
            expect(screen.queryByText('Asset Type')).not.toBeInTheDocument();
        });

        it('marks active asset type buttons based on assetTypes prop', () => {
            const props = makeProps({ assetTypes: ['rule'] });
            render(<SearchPanel {...props} />);
            expect(screen.getByRole('button', { name: /Rules/i })).toHaveClass('active');
            expect(screen.getByRole('button', { name: /CEs/i })).not.toHaveClass('active');
        });

        it('removes a type when toggling an already-selected asset type', () => {
            const props = makeProps({ assetTypes: ['rule', 'ce'] });
            render(<SearchPanel {...props} />);
            fireEvent.click(screen.getByRole('button', { name: /Rules/i }));
            expect(props.onAssetTypesChange).toHaveBeenCalledWith(['ce']);
        });

        it('adds a type when toggling a not-yet-selected asset type', () => {
            const props = makeProps({ assetTypes: ['rule'] });
            render(<SearchPanel {...props} />);
            fireEvent.click(screen.getByRole('button', { name: /CEs/i }));
            expect(props.onAssetTypesChange).toHaveBeenCalledWith(['rule', 'ce']);
        });

        it('treats undefined assetTypes as an empty list when toggling', () => {
            const props = makeProps({ assetTypes: undefined });
            render(<SearchPanel {...props} />);
            fireEvent.click(screen.getByRole('button', { name: /Rules/i }));
            expect(props.onAssetTypesChange).toHaveBeenCalledWith(['rule']);
        });

        it('does nothing when onAssetTypesChange is not provided', () => {
            const props = makeProps({ onAssetTypesChange: undefined });
            render(<SearchPanel {...props} />);
            // Should not throw.
            fireEvent.click(screen.getByRole('button', { name: /Rules/i }));
            expect(screen.getByRole('button', { name: /Rules/i })).toBeInTheDocument();
        });
    });

    describe('categories', () => {
        it('shows the empty message when no categories are available', () => {
            const props = makeProps();
            render(<SearchPanel {...props} availableCategories={[]} />);
            expect(screen.getByText('No categories available.')).toBeInTheDocument();
        });

        it('renders a button per available category', () => {
            const props = makeProps();
            render(<SearchPanel {...props} availableCategories={['Safety', 'Privacy']} />);
            expect(screen.getByRole('button', { name: 'Safety' })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Privacy' })).toBeInTheDocument();
            expect(screen.queryByText('No categories available.')).not.toBeInTheDocument();
        });

        it('selects an unselected category on click', () => {
            const props = makeProps({ categories: [] });
            render(<SearchPanel {...props} availableCategories={['Safety', 'Privacy']} />);
            fireEvent.click(screen.getByRole('button', { name: 'Safety' }));
            expect(props.onCategoriesChange).toHaveBeenCalledWith(['Safety']);
        });

        it('deselects a selected category on click', () => {
            const props = makeProps({ categories: ['Safety'] });
            render(<SearchPanel {...props} availableCategories={['Safety', 'Privacy']} />);
            // The category option button is active.
            const optionBtn = screen.getByRole('button', { name: /Safety/ });
            expect(optionBtn).toHaveClass('active');
            fireEvent.click(optionBtn);
            expect(props.onCategoriesChange).toHaveBeenCalledWith([]);
        });

        it('renders the selected-chips region when categories are selected', () => {
            const props = makeProps({ categories: ['Safety'] });
            render(<SearchPanel {...props} availableCategories={['Safety', 'Privacy']} />);
            expect(screen.getByText('Selected:')).toBeInTheDocument();
            // The chip × button deselects the category.
            const selectedLabel = screen.getByText('Selected:');
            const chipsRegion = selectedLabel.parentElement;
            const removeBtn = within(chipsRegion).getByRole('button', { name: '×' });
            fireEvent.click(removeBtn);
            expect(props.onCategoriesChange).toHaveBeenCalledWith([]);
        });

        it('does not render the selected-chips region when nothing is selected', () => {
            const props = makeProps({ categories: [] });
            render(<SearchPanel {...props} availableCategories={['Safety']} />);
            expect(screen.queryByText('Selected:')).not.toBeInTheDocument();
        });

        it('treats a non-array categories prop as empty', () => {
            const props = makeProps({ categories: null });
            render(<SearchPanel {...props} availableCategories={['Safety']} />);
            // No "Selected:" region because normalizedCategories is [].
            expect(screen.queryByText('Selected:')).not.toBeInTheDocument();
            // Clicking still adds.
            fireEvent.click(screen.getByRole('button', { name: 'Safety' }));
            expect(props.onCategoriesChange).toHaveBeenCalledWith(['Safety']);
        });

        it('does nothing when onCategoriesChange is not provided', () => {
            const props = makeProps({ onCategoriesChange: undefined, categories: [] });
            render(<SearchPanel {...props} availableCategories={['Safety']} />);
            fireEvent.click(screen.getByRole('button', { name: 'Safety' }));
            expect(screen.getByRole('button', { name: 'Safety' })).toBeInTheDocument();
        });
    });

    describe('reset', () => {
        it('resets query, categories, asset types, topK and calls onReset', () => {
            const props = makeProps({ query: 'q', categories: ['Safety'] });
            render(<SearchPanel {...props} availableCategories={['Safety']} />);
            fireEvent.click(screen.getByRole('button', { name: /Reset All/i }));

            expect(props.onQueryChange).toHaveBeenCalledWith('');
            expect(props.onCategoriesChange).toHaveBeenCalledWith([]);
            expect(props.onAssetTypesChange).toHaveBeenCalledWith(['rule', 'ce']);
            expect(props.onTopKChange).toHaveBeenCalledWith(10);
            expect(props.onReset).toHaveBeenCalledTimes(1);
        });

        it('skips asset-type reset when onAssetTypesChange is absent', () => {
            const props = makeProps({ onAssetTypesChange: undefined, query: 'q' });
            render(<SearchPanel {...props} />);
            fireEvent.click(screen.getByRole('button', { name: /Reset All/i }));
            expect(props.onQueryChange).toHaveBeenCalledWith('');
            expect(props.onCategoriesChange).toHaveBeenCalledWith([]);
            expect(props.onTopKChange).toHaveBeenCalledWith(10);
            expect(props.onReset).toHaveBeenCalledTimes(1);
        });
    });
});
