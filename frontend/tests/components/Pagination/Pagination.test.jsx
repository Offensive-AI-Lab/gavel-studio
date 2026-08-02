// Pagination has subtle window-clamping logic that's easy to break:
// the visible-page window must never exceed maxVisibleButtons (5) and must
// shift when the current page is near either end. These tests pin down the
// boundary cases that real users will hit (page 1 of 1, last page, big lists).

import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Pagination from '../../../src/components/Pagination/Pagination';


describe('Pagination', () => {
    it('renders nothing when totalItems fits on one page', () => {
        // Single-page lists shouldn't show the controls — that's just clutter.
        const { container } = render(
            <Pagination currentPage={1} totalItems={5} pageSize={10} onPageChange={() => {}} />,
        );
        expect(container).toBeEmptyDOMElement();
    });

    it('renders nothing when totalItems exactly equals pageSize', () => {
        const { container } = render(
            <Pagination currentPage={1} totalItems={10} pageSize={10} onPageChange={() => {}} />,
        );
        expect(container).toBeEmptyDOMElement();
    });

    it('renders one page-number per page when count <= maxVisible (5)', () => {
        // 25 items / 10 per page = 3 pages. All three buttons should show.
        render(<Pagination currentPage={1} totalItems={25} pageSize={10} onPageChange={() => {}} />);
        expect(screen.getByRole('button', { name: '1' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: '2' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: '3' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: '4' })).not.toBeInTheDocument();
    });

    it('clamps the visible window to 5 buttons even with many pages', () => {
        // 100 items / 10 = 10 pages — max 5 visible at any time.
        render(<Pagination currentPage={5} totalItems={100} pageSize={10} onPageChange={() => {}} />);
        const numericButtons = screen
            .getAllByRole('button')
            .filter((b) => /^\d+$/.test(b.textContent));
        expect(numericButtons).toHaveLength(5);
    });

    it('shifts the window left when currentPage is near the end', () => {
        // currentPage=10 of 10 → visible should be 6,7,8,9,10 (not 8,9,10).
        render(<Pagination currentPage={10} totalItems={100} pageSize={10} onPageChange={() => {}} />);
        for (const n of [6, 7, 8, 9, 10]) {
            expect(screen.getByRole('button', { name: String(n) })).toBeInTheDocument();
        }
        expect(screen.queryByRole('button', { name: '5' })).not.toBeInTheDocument();
    });

    it('marks the active page with the active class', () => {
        render(<Pagination currentPage={3} totalItems={100} pageSize={10} onPageChange={() => {}} />);
        const active = screen.getByRole('button', { name: '3' });
        expect(active).toHaveClass('active');
    });

    it('disables prev on page 1 and next on the last page', () => {
        const { rerender } = render(
            <Pagination currentPage={1} totalItems={50} pageSize={10} onPageChange={() => {}} />,
        );
        const prev = screen.getByRole('button', { name: '<' });
        expect(prev).toBeDisabled();

        rerender(<Pagination currentPage={5} totalItems={50} pageSize={10} onPageChange={() => {}} />);
        const next = screen.getByRole('button', { name: '>' });
        expect(next).toBeDisabled();
    });

    it('calls onPageChange with currentPage-1 when prev is clicked', () => {
        const onChange = vi.fn();
        render(<Pagination currentPage={3} totalItems={50} pageSize={10} onPageChange={onChange} />);
        fireEvent.click(screen.getByRole('button', { name: '<' }));
        expect(onChange).toHaveBeenCalledWith(2);
    });

    it('calls onPageChange with the clicked page number', () => {
        const onChange = vi.fn();
        render(<Pagination currentPage={2} totalItems={50} pageSize={10} onPageChange={onChange} />);
        fireEvent.click(screen.getByRole('button', { name: '4' }));
        expect(onChange).toHaveBeenCalledWith(4);
    });
});
