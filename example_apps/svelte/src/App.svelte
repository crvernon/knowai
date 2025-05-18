<script>
  import { onMount }   from 'svelte';
  import { messages }  from './api.js';
  import { initKnowAI, askKnowAI } from './api.js';

  let question = '';
  let fileList = '';

  onMount(() => {
    initKnowAI();            // download vector‑store and start a session
  });

  async function send() {
    if (!question) return;
    await askKnowAI(question, fileList.split(',').map(s => s.trim()).filter(Boolean));
    question = '';
  }
</script>

<style>
  .chat { max-width: 600px; margin: auto; font-family: sans-serif; }
  .turn { margin: 0.5rem 0; }
  .user { font-weight: bold; }
  .bot  { margin-left: 1rem; }
</style>

<div class="chat">
  {#each $messages as m}
    <div class="turn">
      <div class="user">You: {m.user}</div>
      <div class="bot">🤖 {m.bot}</div>
    </div>
  {/each}

  <input  bind:value={question} placeholder="Ask a question…"   on:keydown={(e)=>e.key==='Enter' && send()} />
  <input  bind:value={fileList} placeholder="file1.pdf,file2.pdf" />
  <button on:click={send}>Send</button>
</div>
