import React from 'react';
import './Pagination.css';

const Pagination = ({ currentPage, totalItems, pageSize, onPageChange }) => {
    // If we don't have enough items for more than 1 page, don't show pagination
    if (totalItems <= pageSize) return null;

    const totalPages = Math.ceil(totalItems / pageSize);
    
    // We want to show a limited window of pages, e.g., max 5 page numbers
    // 1 [2] 3 4 5 ... 10
    // Let's keep it simple for now: Show up to 5 pages around the current page
    
    const renderPageNumbers = () => {
        const pages = [];
        const maxVisibleButtons = 5;
        let startPage = Math.max(1, currentPage - Math.floor(maxVisibleButtons / 2));
        let endPage = Math.min(totalPages, startPage + maxVisibleButtons - 1);

        if (endPage - startPage + 1 < maxVisibleButtons) {
            startPage = Math.max(1, endPage - maxVisibleButtons + 1);
        }

        for (let i = startPage; i <= endPage; i++) {
            pages.push(
                <button
                    key={i}
                    onClick={() => onPageChange(i)}
                    className={`pagination-btn ${currentPage === i ? 'active' : ''}`}
                >
                    {i}
                </button>
            );
        }
        return pages;
    };

    return (
        <div className="pagination-container">
            <button 
                className="pagination-btn"
                disabled={currentPage === 1}
                onClick={() => onPageChange(currentPage - 1)}
            >
                &lt;
            </button>
            
            {renderPageNumbers()}

            <button 
                className="pagination-btn"
                disabled={currentPage >= totalPages}
                onClick={() => onPageChange(currentPage + 1)}
            >
                &gt;
            </button>
        </div>
    );
};

export default Pagination;
