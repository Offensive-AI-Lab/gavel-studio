import React from 'react';
import Sidebar from '../Sidebar/Sidebar';
import './Layout.css'; // Import the CSS here once, so you don't have to elsewhere

const Layout = ({
    children,
    currentModel,
    currentClassifier,
    modelId,
    classifierId,
    raw,
}) => {
    return (
        <div className="app-layout">
            {/* 1. The Fixed Sidebar */}
            <Sidebar
                currentModel={currentModel}
                currentClassifier={currentClassifier}
                modelId={modelId}
                classifierId={classifierId}
            />

            {/* 2. Main content — raw mode skips padding/max-width for full-height pages */}
            <main className={raw ? 'main-viewport main-viewport--raw' : 'main-viewport'}>
                {raw ? children : <div className="content-container">{children}</div>}
            </main>
        </div>
    );
};

export default Layout;