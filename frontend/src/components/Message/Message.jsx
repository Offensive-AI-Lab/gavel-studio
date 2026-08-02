// src/components/Message/Message.jsx
import React from 'react';
import { Link } from 'react-router-dom';
import { FiCheckCircle, FiAlertCircle } from 'react-icons/fi';
import './Message.css';

const Message = ({ type = 'success', title, text, actionText, actionLink }) => {
    return (
        <div className="message-container">
            <div className="message-icon">
                {type === 'success' ? (
                    <FiCheckCircle size={60} color="#10B981" />
                ) : (
                    <FiAlertCircle size={60} color="#EF4444" />
                )}
            </div>
            
            <h2 className="message-title">{title}</h2>
            <p className="message-text">{text}</p>
            
            {actionLink && actionText && (
                <Link to={actionLink}>
                    <button className="message-btn">{actionText}</button>
                </Link>
            )}
        </div>
    );
};

export default Message;