# GAVEL Studio — frontend

React + Vite single-page app for GAVEL Studio. See the [repository README](../README.md) for what Studio is and how to run the whole stack.

## Development

```bash
npm install
npm run dev        # http://localhost:5173 (expects the backend on :8000)
```

Leave `VITE_API_URL` unset for local development — the Vite dev proxy forwards API calls to the backend on `:8000`. Only set it (see `.env.example`) when the frontend is served from a different host than the backend.

## Tests

```bash
npm test
```
