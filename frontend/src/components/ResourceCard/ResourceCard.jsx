import React from 'react';
import { FiTrash2 } from 'react-icons/fi';
import './ResourceCard.css';

const ResourceCard = ({ title, subtitle, icon: Icon, onClick, onDelete }) => {
    return (
        <div className="resource-card" onClick={onClick}>
            <div className="card-icon-wrapper">
                {Icon && <Icon size={20} />}
            </div>

            <div className="card-content">
                <h3 className="card-title">{title}</h3>
                <div className="card-subtitle">
                    {subtitle}
                </div>
            </div>

            {onDelete && (
                <button 
                    className="card-delete-btn"
                    onClick={(e) => {
                        e.stopPropagation(); 
                        onDelete();
                    }}
                    aria-label="Delete"
                >
                    <FiTrash2 className="trash-icon" />
                </button>
            )}
        </div>
    );
};

export default ResourceCard;