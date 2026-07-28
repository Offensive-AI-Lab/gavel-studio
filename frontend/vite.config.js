import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Inject a Content-Security-Policy as a <meta> tag — the second wall behind our
// output-escaping. Even if an XSS sink were ever introduced, `script-src 'self'`
// stops the browser from executing injected/inline script (so it can't read the
// sessionStorage token). Build-time ONLY: the Vite dev server uses inline + eval
// for HMR, which `script-src 'self'` forbids, so applying it in dev would break
// `npm run dev`. `connectSrc` is the set of origins axios is allowed to reach.
//
// NOTE: `frame-ancestors` is IGNORED inside a <meta> CSP — to block clickjacking
// set `X-Frame-Options: DENY` (or `frame-ancestors 'none'`) as a real response
// header on whatever static host serves the build (nginx/Caddy/etc.).
function cspPlugin(connectSrc) {
  const csp = [
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline'",   // React inline styles + SweetAlert-injected <style>
    "img-src 'self' data: https:",        // avatars / data-URI icons
    "font-src 'self' data:",
    `connect-src ${connectSrc}`,          // same-origin + the backend API origin
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
  ].join('; ')
  return {
    name: 'gavel-csp',
    apply: 'build',
    transformIndexHtml(html) {
      return html.replace(
        '</title>',
        `</title>\n    <meta http-equiv="Content-Security-Policy" content="${csp}" />`,
      )
    },
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  // axios (src/api.js) talks ONLY to the local backend (VITE_API_URL); all
  // registry traffic happens backend-side, so no other origin is needed in
  // connect-src. Allow both loopback spellings when unset.
  const sources = new Set(["'self'"])
  if (env.VITE_API_URL) {
    try { sources.add(new URL(env.VITE_API_URL).origin) }
    catch { sources.add(env.VITE_API_URL) }
  } else {
    sources.add('http://127.0.0.1:8000')
    sources.add('http://localhost:8000')
  }
  const connectSrc = [...sources].join(' ')

  return {
    plugins: [react(), cspPlugin(connectSrc)],
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: ['./tests/setup.js'],
      css: false,
      coverage: {
        provider: 'v8',
        reporter: ['text', 'html'],
        include: [
          'src/services/**',
          'src/contexts/**',
          'src/components/**',
          'src/pages/**',
          'src/hooks/**',
        ],
        // Tests, the entry points, and pure-presentation pieces aren't
        // useful coverage signal — exclude them so the report shows real
        // behavior coverage instead of being diluted by glue code.
        exclude: [
          '**/*.test.{js,jsx,ts,tsx}',
          'tests/**',
          'src/main.jsx',
          'src/App.jsx',
        ],
      },
    },
  }
})
