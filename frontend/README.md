# GAVEL Studio — frontend

This is the GAVEL Studio interface: the pages where you browse rules, write Cognitive Elements, train and evaluate probes, and watch a conversation as it happens. It is a React app served by Vite.

It does nothing on its own. The GAVEL Studio backend has to be running as well.

Most people never start this part separately. The `docker compose` command in the [repository README](../README.md) starts the interface and the backend together, and it is the recommended way to run GAVEL Studio.

## Run it

You need [Node.js 20 or newer](https://nodejs.org) and the backend already running on port 8000.

```sh
npm install
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).

There is nothing to configure. The dev server passes API calls through to the backend at `http://127.0.0.1:8000`. Set `VITE_API_URL` only if the interface is served from a different machine than the backend; [`.env.example`](.env.example) shows the format.

For a static build instead of the dev server, run `npm run build` (the files land in `dist/`) and `npm run preview` to open that build locally.

## Test it

```sh
npm test
```

This runs the test suite once. `npm run test:watch` re-runs tests as you edit, `npm run test:coverage` writes a coverage report, and `npm run lint` checks code style.

## Learn more

* [Repository README](../README.md) — what GAVEL Studio is, how to run the whole application, and how to configure it.
* [GAVEL user guide](https://github.com/Offensive-AI-Lab/gavel-rules/blob/main/GUIDE.md) — Cognitive Elements, rules, and rule sets explained with an example.
