import { LOCAL_USER } from '../localUser';

/**
 * Returns the acting user.
 *
 * There is no login: GAVEL Studio is a single-operator localhost app, so this
 * always resolves to the one local user and never redirects. Kept as a hook so
 * the pages that call it needed no edit, and so there is still exactly one
 * place to change if a real identity is ever reintroduced.
 */
const useRequireAuth = () => LOCAL_USER;

export default useRequireAuth;
