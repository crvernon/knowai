/**
 * Minimal Svelte front‑end that talks to the knowai FastAPI service.
 * Assumes the knowai container is reachable at `VITE_KNOWAI_API`
 * (e.g., http://localhost:8000 when running locally).
 *
 * File layout:
 *   src/
 *     ├── App.svelte      ← main UI component (included below)
 *     └── api.js          ← helper functions to call knowai
 *
 * To scaffold a new Svelte project quickly:
 *   npm create vite@latest my‑app -- --template svelte
 *   cd my‑app
 *   npm install
 *   # copy App.svelte and api.js into `src/`
 *   # add `VITE_KNOWAI_API=http://localhost:8000` to your `.env`
 *   npm run dev
 *
 * This single file includes *both* api.js and App.svelte for brevity.
 * Copy each piece into its own file as noted above.
 */
import { writable, get } from 'svelte/store';

export const sessionId = writable(null);
export const messages  = writable([]);   // [{ user, bot }]

const API_BASE = import.meta.env.VITE_KNOWAI_API || 'http://localhost:8000';

/** Initialise a KnowAIAgent session and store the session_id. */
export async function initKnowAI(vectorstoreUri = 's3://your‑bucket/your‑vec/') {
  const res  = await fetch(`${API_BASE}/initialize`, {
    method : 'POST',
    headers: { 'Content‑Type': 'application/json' },
    body   : JSON.stringify({ vectorstore_s3_uri: vectorstoreUri })
  });
  const data = await res.json();
  sessionId.set(data.session_id);
  return data.session_id;
}

/** Ask a question and append the turn to the `messages` store. */
export async function askKnowAI(question, selectedFiles = []) {
  const id = get(sessionId);
  if (!id) throw new Error('KnowAI session not initialised');

  const res  = await fetch(`${API_BASE}/ask`, {
    method : 'POST',
    headers: { 'Content‑Type': 'application/json' },
    body   : JSON.stringify({
      session_id    : id,
      question,
      selected_files: selectedFiles
    })
  });
  const data = await res.json();
  messages.update(m => [...m, { user: question, bot: data.generation }]);
  return data;
}
