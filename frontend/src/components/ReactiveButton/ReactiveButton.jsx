import React from 'react';
import { FiPlus } from 'react-icons/fi';
import './ReactiveButton.css';

// Added 'disabled' and 'style' props
const ReactiveButton = ({ label, onClick, Icon: IconComponent = FiPlus, disabled = false, style = {} }) => {
    return (
        <button 
            className={`reactive-btn ${disabled ? 'disabled' : ''}`} 
            onClick={onClick}
            disabled={disabled}
            style={style}
        >
            {React.createElement(IconComponent, { size: 20 })}
            <span>{label}</span>
        </button>
    );
};

export default ReactiveButton;