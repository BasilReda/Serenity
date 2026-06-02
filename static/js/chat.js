/* ── Session ─────────────────────────────────────────────────────────────── */
let userId = localStorage.getItem('serenity_uid');
if (!userId) {
  userId = 'user_' + Math.random().toString(36).slice(2, 10);
  localStorage.setItem('serenity_uid', userId);
}
document.getElementById('session-display').textContent = userId;

/* ── Node → pipeline step map ────────────────────────────────────────────── */
const NODE_MAP = {
  language_detector:       'step-language',
  translate_to_english:    'step-language',
  emotion_detector:        'step-emotion',
  input_guardrail:         'step-guardrail-in',
  intent_detector:         'step-intent',
  query_classifier_agent:  'step-complexity',
  HyDE:                    'step-retrieval',
  RAG:                     'step-retrieval',
  ReRanker:                'step-retrieval',
  grade_document:          'step-retrieval',
  query_rewrite:           'step-retrieval',
  mental_health_chatbot:   'step-generate',
  general_handler:         'step-generate',
  // output_guardrail:        'step-guardrail-out',
};

/* ── Pipeline helpers ────────────────────────────────────────────────────── */
let activeStep = null;

function resetPipeline() {
  document.querySelectorAll('.step').forEach(el =>
    el.classList.remove('active', 'done', 'blocked')
  );
  activeStep = null;
}

function activateStep(id) {
  if (!id) return;
  if (activeStep && activeStep !== id) {
    const prev = document.getElementById(activeStep);
    if (prev) { prev.classList.remove('active'); prev.classList.add('done'); }
  }
  const el = document.getElementById(id);
  if (el) { el.classList.add('active'); el.classList.remove('done', 'blocked'); }
  activeStep = id;
}

function blockStep(id) {
  if (!id) return;
  const el = document.getElementById(id);
  if (el) { el.classList.add('blocked'); el.classList.remove('active', 'done'); }
  activeStep = null;
}

function finishPipeline() {
  if (activeStep) {
    const el = document.getElementById(activeStep);
    if (el) { el.classList.remove('active'); el.classList.add('done'); }
  }
  activeStep = null;
}

/* ── Status bar ──────────────────────────────────────────────────────────── */
function setStatus(text, state = 'idle') {
  document.getElementById('status-text').textContent = text;
  const dot = document.getElementById('status-dot');
  dot.className = 'status-dot';
  if (state === 'busy')  dot.classList.add('busy');
  if (state === 'error') dot.classList.add('error');
}

/* ── DOM helpers ─────────────────────────────────────────────────────────── */
function hideEmpty() {
  const el = document.getElementById('empty-state');
  if (el) el.remove();
}

function scrollBottom() {
  const c = document.getElementById('messages');
  c.scrollTo({ top: c.scrollHeight, behavior: 'smooth' });
}

function timestamp() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function addMessage(role, text, emotion = null) {
  hideEmpty();
  const list = document.getElementById('messages');

  const row = document.createElement('div');
  row.className = `msg ${role}`;

  const av = document.createElement('div');
  av.className = 'avatar';
  av.textContent = role === 'user' ? 'Y' : 'S';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const bub = document.createElement('div');
  bub.className = 'bubble';
  bub.textContent = text;

  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.textContent = timestamp();

  if (emotion && role === 'user') {
    const tag = document.createElement('span');
    tag.className = 'emotion-tag';
    tag.textContent = emotion;
    meta.appendChild(tag);
  }

  body.appendChild(bub);
  body.appendChild(meta);
  row.appendChild(av);
  row.appendChild(body);
  list.appendChild(row);
  scrollBottom();
  return bub;
}

function showTyping() {
  hideEmpty();
  const list = document.getElementById('messages');
  const row  = document.createElement('div');
  row.id = 'typing';
  row.className = 'msg bot';

  const av = document.createElement('div');
  av.className = 'avatar';
  av.textContent = 'S';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const t = document.createElement('div');
  t.className = 'typing';
  [1,2,3].forEach(() => {
    const d = document.createElement('span');
    d.className = 'tdot';
    t.appendChild(d);
  });

  body.appendChild(t);
  row.appendChild(av);
  row.appendChild(body);
  list.appendChild(row);
  scrollBottom();
}

function hideTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

/* ── Send message ─────────────────────────────────────────────────────────── */
let busy = false;

async function sendMessage() {
  const input   = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const message = input.value.trim();
  if (!message || busy) return;

  busy = true;
  sendBtn.disabled = true;
  input.value = '';
  input.style.height = 'auto';

  resetPipeline();
  addMessage('user', message);
  showTyping();
  setStatus('Processing…', 'busy');

  let emotion = null;

  try {
    const res = await fetch('/chat/stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message, user_id: userId }),
    });

    // FIXED: Catch HTTP errors immediately before trying to read the stream
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Server error (${res.status}): ${errText}`);
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();   // keep incomplete line for next chunk

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;

        let ev;
        try { ev = JSON.parse(raw); } catch { continue; }

        /* ── status event ── */
        if (ev.type === 'status') {
          const stepId = NODE_MAP[ev.node] || null;
          if (ev.data.includes('BLOCKED')) {
            blockStep(stepId);
            setStatus('Blocked', 'error');
          } else {
            activateStep(stepId);
            const label = ev.data.length > 52 ? ev.data.slice(0, 52) + '…' : ev.data;
            setStatus(label, 'busy');
          }
          /* extract emotion from status string */
          const em = ev.data.match(/Detected Emotion:\s*(\w+)/i);
          if (em) emotion = em[1].toLowerCase();
        }

        /* ── response event ── */
        if (ev.type === 'response') {
          hideTyping();
          finishPipeline();
          addMessage('bot', ev.data, null);
          setStatus('Ready');
        }

        /* ── error event ── */
        if (ev.type === 'error') {
          hideTyping();
          finishPipeline();
          addMessage('bot', 'Something went wrong processing that request.');
          setStatus('Error', 'error');
          console.error('[Pipeline Error]', ev.data);
        }

        /* ── done event ── */
        if (ev.type === 'done') {
          hideTyping();
          finishPipeline();
          setStatus('Ready');
        }
      }
    }
  } catch (err) {
    hideTyping();
    finishPipeline();
    addMessage('bot', 'Connection error. Please check your server and try again.');
    setStatus('Offline', 'error');
    console.error('[Serenity]', err);
  }

  busy = false;
  sendBtn.disabled = false;
  input.focus();
}

/* ── New chat ─────────────────────────────────────────────────────────────── */
function newChat() {
  userId = 'user_' + Math.random().toString(36).slice(2, 10);
  localStorage.setItem('serenity_uid', userId);
  document.getElementById('session-display').textContent = userId;

  document.getElementById('messages').innerHTML = `
    <div class="empty-state" id="empty-state">
      <div class="empty-icon">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#f5f0e8" stroke-width="1.5" stroke-linecap="round">
          <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
        </svg>
      </div>
      <h2 class="empty-heading">You are not alone</h2>
      <p class="empty-body">Share what is on your mind. I am here to listen and support you on your journey.</p>
    </div>`;

  resetPipeline();
  setStatus('Ready');
}

/* ── Keyboard ─────────────────────────────────────────────────────────────── */
const chatInput = document.getElementById('chat-input');

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

chatInput.addEventListener('input', function () {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});