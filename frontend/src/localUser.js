// The single local user.
//
// GAVEL Studio has no login — it is a one-operator localhost application. The
// backend attributes every request to a fixed user id (see
// backend/utils/auth.py: LOCAL_USER_ID), and this is the frontend's mirror of
// that identity.
//
// Why it is still written into sessionStorage: ~28 components read
// `JSON.parse(sessionStorage.getItem('user'))` to get `user_id` for API calls
// and to decide what to render. Seeding the same object those components
// already expect means none of them had to change when login was removed —
// and there is exactly one place (here) to touch if identity ever comes back.

export const LOCAL_USER = {
    user_id: 1,           // must match backend utils/auth.py LOCAL_USER_ID
    username: 'local',
    email: 'local@gavel.local',
    display_name: 'Local User',
    is_team: false,
};

/**
 * Make the local user available to every component that reads it from
 * sessionStorage. Called once, before the router mounts.
 *
 * Merge-not-overwrite: `tutorial_seen` is flipped in the stored object by
 * Tutorial.jsx after the onboarding modal is dismissed, so a blind overwrite on
 * every reload would re-open the tutorial forever. We only fill in the identity
 * fields and leave anything already there alone.
 */
export function seedLocalUser() {
    let existing = null;
    try {
        existing = JSON.parse(sessionStorage.getItem('user') || 'null');
    } catch {
        existing = null;
    }
    sessionStorage.setItem('user', JSON.stringify({ ...LOCAL_USER, ...(existing || {}) }));
}
