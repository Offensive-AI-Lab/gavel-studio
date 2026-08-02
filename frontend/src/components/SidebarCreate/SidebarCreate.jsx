// SidebarCreate — the global "create" entry point, in the sidebar.
// Clicking "Create" opens the shared CreateChooserModal (CE / Rule / Build Rule).

import { useState } from 'react';
import { FiPlus } from 'react-icons/fi';
import CreateChooserModal from '../CreateChooserModal/CreateChooserModal';

const SidebarCreate = () => {
    const [open, setOpen] = useState(false);
    return (
        <>
            <div className="nav-item" onClick={() => setOpen(true)}>
                <FiPlus className="nav-icon" />
                <span>Create</span>
            </div>
            <CreateChooserModal isOpen={open} onClose={() => setOpen(false)} />
        </>
    );
};

export default SidebarCreate;
