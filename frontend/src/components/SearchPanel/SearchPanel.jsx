import React from 'react';
import { FiSearch, FiFilter, FiX, FiRefreshCcw, FiCheckCircle } from 'react-icons/fi';
import './SearchPanel.css';

const SearchPanel = ({
    query,
    onQueryChange,
    categories,
    onCategoriesChange,
    onTopKChange,
    onSearch,
    onReset,
    loading,
    assetTypes,
    onAssetTypesChange,
    showAssetTypeFilter = true,
    searchPlaceholder = 'Search rules, cognitive elements, and more...',
    availableCategories = [],
    allowEmptyQuery = false
}) => {
    const normalizedCategories = Array.isArray(categories) ? categories : [];

    const handleAssetTypeToggle = (type) => {
        if (!onAssetTypesChange) return;
        const current = assetTypes || [];
        if (current.includes(type)) {
            onAssetTypesChange(current.filter(t => t !== type));
        } else {
            onAssetTypesChange([...current, type]);
        }
    };

    const handleReset = () => {
        onQueryChange('');
        onCategoriesChange([]);
        if (onAssetTypesChange) {
            onAssetTypesChange(['rule', 'ce']);
        }
        onTopKChange(10);
        onReset();
    };


    const toggleCategory = (value) => {
        if (!onCategoriesChange) return;
        const exists = normalizedCategories.includes(value);
        if (exists) {
            onCategoriesChange(normalizedCategories.filter((item) => item !== value));
        } else {
            onCategoriesChange([...normalizedCategories, value]);
        }
    };

    return (
        <div className="search-panel">
            {/* Main Search Bar */}
            <div className="search-bar-container">
                <div className="search-input-group">
                    <FiSearch className="search-icon" />
                    <input
                        type="text"
                        className="search-input"
                        placeholder={searchPlaceholder}
                        value={query}
                        onChange={(e) => onQueryChange(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && onSearch()}
                    />
                    {query && (
                        <button
                            className="clear-btn"
                            onClick={() => onQueryChange('')}
                            title="Clear search"
                        >
                            <FiX />
                        </button>
                    )}
                </div>
                <button
                    className="search-btn"
                    onClick={onSearch}
                    disabled={loading || (!allowEmptyQuery && !query.trim())}
                >
                    {loading ? (
                        <>
                            <span className="spinner"></span>
                            Searching...
                        </>
                    ) : (
                        <>
                            <FiSearch /> Search
                        </>
                    )}
                </button>
            </div>

            {/* Filter Section */}
            <div className="filter-section">
                {/* Quick Filters */}
                <div className="quick-filters">
                    {showAssetTypeFilter && (
                        <div className="filter-item">
                            <label className="filter-label">Asset Type</label>
                            <div className="asset-type-selector">
                                <button
                                    className={`asset-btn ${assetTypes?.includes('rule') ? 'active' : ''}`}
                                    onClick={() => handleAssetTypeToggle('rule')}
                                >
                                    📋 Rules
                                </button>
                                <button
                                    className={`asset-btn ${assetTypes?.includes('ce') ? 'active' : ''}`}
                                    onClick={() => handleAssetTypeToggle('ce')}
                                >
                                    🧠 CEs
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {/* Categories */}
                <div className="categories-input-group">
                    <label className="filter-label">
                        <FiFilter size={16} /> Filter by Categories
                    </label>
                    {availableCategories.length === 0 ? (
                        <div className="category-empty">No categories available.</div>
                    ) : (
                        <>
                            <div className="category-option-list">
                                {availableCategories.map((category) => {
                                    const active = normalizedCategories.includes(category);
                                    return (
                                        <button
                                            type="button"
                                            key={category}
                                            className={`category-option ${active ? 'active' : ''}`}
                                            onClick={() => toggleCategory(category)}
                                        >
                                            {active && <FiCheckCircle className="category-option-icon" />}
                                            {category}
                                        </button>
                                    );
                                })}
                            </div>
                            {normalizedCategories.length > 0 && (
                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '12px', paddingTop: '12px', borderTop: '1px solid #e5e7eb' }}>
                                    <label style={{ fontSize: '0.85rem', color: '#6b7280', width: '100%' }}>Selected:</label>
                                    {normalizedCategories.map((cat) => (
                                        <span 
                                            key={cat}
                                            style={{
                                                display: 'inline-flex',
                                                alignItems: 'center',
                                                gap: '6px',
                                                padding: '6px 12px',
                                                backgroundColor: '#eff6ff',
                                                border: '1px solid #bfdbfe',
                                                borderRadius: '999px',
                                                fontSize: '0.9rem',
                                                color: '#1d4ed8',
                                                fontWeight: 600
                                            }}
                                        >
                                            {cat}
                                            <button
                                                onClick={() => toggleCategory(cat)}
                                                style={{
                                                    background: 'none',
                                                    border: 'none',
                                                    color: '#1d4ed8',
                                                    cursor: 'pointer',
                                                    padding: 0,
                                                    fontSize: '1rem',
                                                    lineHeight: 1
                                                }}
                                            >
                                                ×
                                            </button>
                                        </span>
                                    ))}
                                </div>
                            )}
                        </>
                    )}
                </div>

                {/* Action Buttons */}
                <div className="action-buttons">
                    <button
                        className="reset-btn"
                        onClick={handleReset}
                        title="Reset all filters"
                    >
                        <FiRefreshCcw /> Reset All
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SearchPanel;
