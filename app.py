"""
Synapse — A Zenith Company
Dark, mobile-first waiting room UI
"""

import os
import sys
import re
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Synapse",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,300;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
  font-family: 'Inter Tight', sans-serif;
  background-color: #05030D;
  color: #e8e8e8;
}

.stApp {
  background:
    radial-gradient(circle at 50% 30%, rgba(138,92,246,0.25), transparent 40%),
    linear-gradient(180deg, #0E0A1F 0%, #05030D 100%) !important;
  min-height: 100vh;
}

/* Edge streaks — on body so they escape all Streamlit containers */
body::before, body::after {
  content: '';
  position: fixed;
  bottom: -30px;
  width: 4px;
  height: 320px;
  background: linear-gradient(to top, rgba(167,139,250,1) 0%, rgba(139,92,246,0.75) 35%, rgba(109,40,217,0.35) 65%, transparent 100%);
  box-shadow: 0 0 22px 7px rgba(139,92,246,0.75), 0 0 65px 18px rgba(109,40,217,0.4);
  border-radius: 3px;
  pointer-events: none;
  z-index: 99999;
  animation: streak-breathe 3.5s ease-in-out infinite;
}
body::before { left: 38px;  transform: rotate(-20deg); transform-origin: bottom center; }
body::after  { right: 38px; transform: rotate(20deg);  transform-origin: bottom center; animation-delay: 0.7s; }

#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; display: none; }

.block-container {
  max-width: 560px;
  padding: 0 1.25rem 7rem 1.25rem;
  margin: 0 auto;
}

/* Animated edge streaks */
.streak-left, .streak-right {
  position: fixed;
  bottom: -24px;
  width: 2px;
  height: 270px;
  background: linear-gradient(to top, rgba(139,92,246,1) 0%, rgba(109,40,217,0.55) 55%, transparent 100%);
  box-shadow: 0 0 18px 3px rgba(139,92,246,0.65), 0 0 55px 8px rgba(109,40,217,0.35);
  border-radius: 2px;
  pointer-events: none;
  z-index: 9999;
  animation: streak-breathe 3.5s ease-in-out infinite;
}
.streak-left  { left: 52px;  transform: rotate(-22deg); transform-origin: bottom center; }
.streak-right { right: 52px; transform: rotate(22deg);  transform-origin: bottom center; animation-delay: 0.6s; }

@keyframes streak-breathe {
  0%, 100% { opacity: 0.32; }
  50%       { opacity: 0.88; }
}

/* Hero */
.hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 3rem 0 2rem 0;
}

.logo-glow {
  position: relative;
  width: 210px;
  height: 210px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1.25rem;
}

.logo-glow::before {
  content: '';
  position: absolute;
  inset: -20px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(109,40,217,0.35) 0%, rgba(91,33,182,0.12) 55%, transparent 72%);
  animation: glow-pulse 3.5s ease-in-out 3 forwards;
}

@keyframes glow-pulse {
  0%,100% { transform: scale(1);    opacity: 0.55; }
  50%      { transform: scale(1.14); opacity: 1;   }
}

.logo-svg {
  width: 190px;
  height: 190px;
  position: relative;
  z-index: 1;
  animation: pop-in 0.9s cubic-bezier(0.16,1,0.3,1) both;
}

@keyframes pop-in {
  from { opacity: 0; transform: scale(0.8) translateY(8px); }
  to   { opacity: 1; transform: scale(1)   translateY(0);   }
}

.hb-line {
  stroke-dasharray: 600;
  stroke-dashoffset: 600;
  animation: draw-line 1.4s 0.3s cubic-bezier(0.4,0,0.2,1) forwards;
}

@keyframes draw-line {
  to { stroke-dashoffset: 0; }
}

.app-name {
  font-family: 'Inter Tight', sans-serif;
  font-size: 4.1rem;
  font-weight: 800;
  font-style: italic;
  letter-spacing: -0.045em;
  color: #ffffff;
  line-height: 1;
  text-align: center;
  animation: rise 0.7s 0.5s cubic-bezier(0.16,1,0.3,1) both;
}

.company-tag {
  font-family: 'Inter Tight', sans-serif;
  font-size: 0.68rem;
  font-weight: 300;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: #5a5a7a;
  margin-top: 0.5rem;
  text-align: center;
  animation: rise 0.7s 0.65s cubic-bezier(0.16,1,0.3,1) both;
}

@keyframes rise {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0);    }
}

/* Conversation */
.bubble-user {
  background: #12062a;
  border: 1px solid #2e1065;
  color: #c4b5fd;
  border-radius: 18px 18px 4px 18px;
  padding: 0.875rem 1.125rem;
  max-width: 86%;
  margin-left: auto;
  font-size: 0.91rem;
  line-height: 1.55;
  font-family: 'Inter Tight', sans-serif;
}

.resp-wrap { display: flex; flex-direction: column; gap: 0.7rem; }

.resp-card {
  background: #0b0b1a;
  border: 1px solid #1c1c30;
  border-radius: 14px;
  padding: 1.1rem 1.25rem;
}

.card-label {
  font-family: 'Inter Tight', sans-serif;
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}

.card-body {
  font-size: 0.88rem;
  color: #999;
  line-height: 1.74;
  font-family: 'Inter Tight', sans-serif;
}

.q-card {
  background: #0b0b1a;
  border: 1px solid #1e1640;
  border-radius: 14px;
  padding: 1.1rem 1.25rem;
}

.q-item {
  display: flex;
  gap: 0.7rem;
  align-items: flex-start;
  margin: 0.5rem 0;
  font-size: 0.88rem;
  color: #b8a9d9;
  line-height: 1.5;
  font-family: 'Inter Tight', sans-serif;
}

.q-arrow { color: #7c3aed; font-weight: 700; flex-shrink: 0; }

.emerg-card {
  background: #140505;
  border: 1.5px solid #991b1b;
  border-radius: 14px;
  padding: 1.25rem;
  font-size: 0.88rem;
  color: #fca5a5;
  line-height: 1.65;
}

.emerg-title {
  font-family: 'Inter Tight', sans-serif;
  font-weight: 700;
  font-size: 0.92rem;
  color: #ef4444;
  margin-bottom: 0.6rem;
}

.src-item {
  background: #0a0a18;
  border: 1px solid #181828;
  border-radius: 10px;
  padding: 0.75rem 1rem;
  margin: 0.4rem 0;
  font-size: 0.8rem;
}

.src-name { font-weight: 500; color: #ccc; margin-bottom: 0.15rem; }
.src-sub  { font-size: 0.7rem; color: #4a4a6a; }

.conf-bar  { height: 3px; background: #1a1a2a; border-radius: 2px; margin-top: 0.45rem; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg,#5b21b6,#8b5cf6); }

.turn-div { border: none; border-top: 1px solid #141424; margin: 1.25rem 0; }

/* Disclaimer */
.disclaimer {
  text-align: center;
  font-size: 0.72rem;
  color: #5a5a7a;
  letter-spacing: 0.02em;
  line-height: 1.7;
  padding: 0.75rem 0.5rem 0.6rem 0.5rem;
  border-top: 1px solid #14142a;
  margin-bottom: 0.4rem;
  animation: rise 0.7s 0.8s both;
  font-family: 'Inter Tight', sans-serif;
}

.not-red { color: #8b0000; font-weight: 700; }

/* Input */
.stTextArea label { display: none !important; }
.stTextArea textarea {
  background: #090916 !important;
  border: 1px solid #1e1e38 !important;
  border-radius: 14px !important;
  color: #e0e0e0 !important;
  font-family: 'Inter Tight', sans-serif !important;
  font-size: 0.93rem !important;
  padding: 0.9rem 1rem !important;
  resize: none !important;
}
.stTextArea textarea:focus {
  border-color: #5b21b6 !important;
  box-shadow: 0 0 0 3px rgba(91,33,182,0.14) !important;
}
.stTextArea textarea::placeholder { color: #3a3a5a !important; }

.stButton > button {
  background: transparent !important;
  color: #fff !important;
  border: 1.5px solid #6d28d9 !important;
  border-radius: 12px !important;
  padding: 0.72rem 2rem !important;
  font-family: 'Inter Tight', sans-serif !important;
  font-size: 0.88rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  width: 100% !important;
  transition: background 0.2s, border-color 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
  background: rgba(109,40,217,0.13) !important;
  border-color: #8b5cf6 !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

[data-testid="stSidebar"] { background: #060614 !important; border-right: 1px solid #16162a !important; }

.empty-hint {
  text-align: center;
  color: #68688a;
  font-size: 0.85rem;
  line-height: 1.8;
  padding: 1.25rem 0 1.75rem 0;
  animation: rise 0.7s 0.9s both;
  font-family: 'Inter Tight', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Streaks injected as real fixed divs — body::before/::after get clipped by Streamlit's React shell
st.markdown("""
<style>
@keyframes streak-breathe-real {
  0%, 100% { opacity: 0.32; }
  50%       { opacity: 0.92; }
}
</style>
<div style="
  position:fixed; bottom:-30px; left:38px;
  width:4px; height:320px;
  background:linear-gradient(to top, rgba(167,139,250,1) 0%, rgba(139,92,246,0.75) 35%, rgba(109,40,217,0.35) 65%, transparent 100%);
  box-shadow:0 0 22px 7px rgba(139,92,246,0.75), 0 0 65px 18px rgba(109,40,217,0.4);
  border-radius:3px; pointer-events:none; z-index:99999;
  transform:rotate(-20deg); transform-origin:bottom center;
  animation:streak-breathe-real 3.5s ease-in-out infinite;
"></div>
<div style="
  position:fixed; bottom:-30px; right:38px;
  width:4px; height:320px;
  background:linear-gradient(to top, rgba(167,139,250,1) 0%, rgba(139,92,246,0.75) 35%, rgba(109,40,217,0.35) 65%, transparent 100%);
  box-shadow:0 0 22px 7px rgba(139,92,246,0.75), 0 0 65px 18px rgba(109,40,217,0.4);
  border-radius:3px; pointer-events:none; z-index:99999;
  transform:rotate(20deg); transform-origin:bottom center;
  animation:streak-breathe-real 3.5s ease-in-out infinite; animation-delay:0.7s;
"></div>
""", unsafe_allow_html=True)

# Heartbeat audio — plays once per session in sync with the 3 logo pulses
st.markdown("""
<script>
(function() {
  var KEY = '_syn_hb_played';
  if (sessionStorage.getItem(KEY)) return;
  sessionStorage.setItem(KEY, '1');

  var PULSE_SEC  = 3.5;   // must match glow-pulse animation duration
  var PULSE_COUNT = 3;

  function makeBeat(ctx, startTime, hz, vol, durSec) {
    var n   = Math.floor(ctx.sampleRate * durSec);
    var buf = ctx.createBuffer(1, n, ctx.sampleRate);
    var d   = buf.getChannelData(0);
    for (var i = 0; i < n; i++) {
      var t   = i / ctx.sampleRate;
      var env = Math.exp(-t * 22) * (1 - Math.exp(-t * 140));
      d[i]    = env * Math.sin(2 * Math.PI * hz * t) * vol;
    }
    var src = ctx.createBufferSource();
    src.buffer = buf;
    var g = ctx.createGain();
    src.connect(g);
    g.connect(ctx.destination);
    src.start(startTime);
  }

  function scheduleAll(ctx) {
    var t0 = ctx.currentTime + 0.25;   // slight lead-in
    for (var i = 0; i < PULSE_COUNT; i++) {
      var t = t0 + i * PULSE_SEC;
      makeBeat(ctx, t,        80, 0.85, 0.28);  // lub
      makeBeat(ctx, t + 0.22, 62, 0.65, 0.22);  // dub
    }
  }

  try {
    var AC  = window.AudioContext || window.webkitAudioContext;
    var ctx = new AC();

    if (ctx.state === 'running') {
      scheduleAll(ctx);
    } else {
      // Resume on first user touch/click then schedule
      var fired = false;
      function tryPlay() {
        if (fired) return;
        fired = true;
        ctx.resume().then(function() { scheduleAll(ctx); }).catch(function(){});
        document.removeEventListener('click',      tryPlay);
        document.removeEventListener('touchstart', tryPlay);
        document.removeEventListener('keydown',    tryPlay);
      }
      ctx.resume().then(function() {
        if (ctx.state === 'running') { fired = true; scheduleAll(ctx); }
      }).catch(function(){});
      document.addEventListener('click',      tryPlay);
      document.addEventListener('touchstart', tryPlay);
      document.addEventListener('keydown',    tryPlay);
    }
  } catch(e) {}
})();
</script>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init():
    for k, v in {
        "conversation": [], "chunks_built": False, "chunks": [],
        "hybrid": None, "api_key": os.getenv("OPENAI_API_KEY", ""),
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    key_in = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if key_in:
        st.session_state.api_key = key_in
    st.markdown("---")
    st.markdown("### 📊 Metrics")
    try:
        from evaluation.evaluator import Evaluator
        s = Evaluator().summary()
        if "avg_recall_at_k" in s:
            st.metric("Recall@5", f"{s['avg_recall_at_k']:.0%}")
            st.metric("Precision@5", f"{s['avg_precision_at_k']:.0%}")
            st.metric("MRR", f"{s['avg_mrr']:.2f}")
        else:
            st.caption("Ask a question to start tracking.")
    except Exception:
        st.caption("Metrics appear after first query.")
    st.markdown("---")
    st.caption("Synapse searches PubMed research to help you prepare for your doctor visit. Not a diagnostic tool.")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero">
  <div class="logo-glow">
    <svg class="logo-svg" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="lg" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stop-color="#2e1065" stop-opacity="0.8"/>
          <stop offset="35%"  stop-color="#6d28d9"/>
          <stop offset="65%"  stop-color="#8b5cf6"/>
          <stop offset="100%" stop-color="#ddd6fe" stop-opacity="0.9"/>
        </linearGradient>
        <filter id="gl">
          <feGaussianBlur stdDeviation="2" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <polyline class="hb-line"
        points="8,100 65,100 78,100 92,36 108,164 122,100 138,100 192,100"
        fill="none"
        stroke="url(#lg)"
        stroke-width="4"
        stroke-linecap="round"
        stroke-linejoin="round"
        filter="url(#gl)"
      />
    </svg>
  </div>
  <h1 class="app-name">SYNAPSE</h1>
  <p class="company-tag">A Zenith Company</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Response renderer
# ---------------------------------------------------------------------------

def render(result):
    if result.get("is_emergency"):
        st.markdown(f"""<div class="emerg-card">
          <div class="emerg-title">🚨 Please speak with medical staff now</div>
          {result['answer'].replace(chr(10),'<br>')}
        </div>""", unsafe_allow_html=True)
        return

    ans = result["answer"]
    rm = re.search(r'📋 WHAT THE RESEARCH SAYS\s*(.*?)(?=🔬|❓|⚠️|$)', ans, re.DOTALL)
    em = re.search(r'🔬 WHAT YOUR DOCTOR WILL EVALUATE\s*(.*?)(?=📋|❓|⚠️|$)', ans, re.DOTALL)
    qm = re.search(r'❓ QUESTIONS TO ASK YOUR DOCTOR TODAY\s*(.*?)(?=📋|🔬|⚠️|$)', ans, re.DOTALL)

    research  = rm.group(1).strip() if rm else ""
    evaluate  = em.group(1).strip() if em else ""
    questions = qm.group(1).strip() if qm else ""

    if not any([research, evaluate, questions]):
        st.markdown(f'<div class="resp-card"><div class="card-body">{ans}</div></div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="resp-wrap">', unsafe_allow_html=True)

    if research:
        st.markdown(f"""<div class="resp-card">
          <div class="card-label" style="color:#7c3aed">📋 What the Research Says</div>
          <div class="card-body">{research.replace(chr(10),'<br>')}</div>
        </div>""", unsafe_allow_html=True)

    if evaluate:
        st.markdown(f"""<div class="resp-card">
          <div class="card-label" style="color:#0e7490">🔬 What Your Doctor Will Evaluate</div>
          <div class="card-body">{evaluate.replace(chr(10),'<br>')}</div>
        </div>""", unsafe_allow_html=True)

    if questions:
        lines = [re.sub(r'^[\d\.\-\*→]+\s*','', l.strip()) for l in questions.split('\n') if l.strip()]
        qhtml = ''.join(f'<div class="q-item"><span class="q-arrow">→</span><span>{l}</span></div>' for l in lines if l)
        st.markdown(f"""<div class="q-card">
          <div class="card-label" style="color:#8b5cf6">❓ Questions to Ask Your Doctor Today</div>
          {qhtml}
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if result.get("sources"):
        with st.expander("📄 Research Sources", expanded=False):
            for i, s in enumerate(result["sources"], 1):
                conf = s.get("confidence_pct", 0)
                st.markdown(f"""<div class="src-item">
                  <div class="src-name">[{i}] {s.get('title','Unknown')}</div>
                  <div class="src-sub">PMID {s.get('pmid','—')} · {conf}% confidence</div>
                  <div class="conf-bar"><div class="conf-fill" style="width:{conf}%"></div></div>
                </div>""", unsafe_allow_html=True)
                if s.get("url"):
                    st.markdown(f"[View on PubMed ↗]({s['url']})")


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

if st.session_state.conversation:
    for turn in st.session_state.conversation:
        st.markdown(f'<div class="bubble-user">{turn["query"]}</div>', unsafe_allow_html=True)
        render(turn["result"])
        st.markdown('<hr class="turn-div">', unsafe_allow_html=True)
else:
    st.markdown("""<div class="empty-hint">
      Describe your symptoms or ask anything<br>you want to understand before seeing your doctor.
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Disclaimer + input
# ---------------------------------------------------------------------------

st.markdown("""
<div class="disclaimer">
  This is <span class="not-red">NOT</span> a diagnosis tool.<br>
  Please consult your physician for emergencies.
</div>
""", unsafe_allow_html=True)

with st.form("q_form", clear_on_submit=True):
    query = st.text_area("Your question", placeholder="What's on your mind? Describe your symptoms or ask a question...", height=88, label_visibility="collapsed")
    submitted = st.form_submit_button("Ask Synapse →")

# ---------------------------------------------------------------------------
# Submit handler
# ---------------------------------------------------------------------------

if submitted and query.strip():
    if not st.session_state.api_key:
        st.error("Add your OpenAI API key in the sidebar.")
        st.stop()

    api_key = st.session_state.api_key

    with st.spinner("Searching the research..."):
        try:
            from Generation.answer_generator import check_emergency, AnswerGenerator, EMERGENCY_RESPONSE
            from Retrieval.reranker import Reranker

            if check_emergency(query):
                result = {"answer": EMERGENCY_RESPONSE, "is_emergency": True, "sources": [], "query": query, "model": "emergency_bypass"}
            else:
                if not st.session_state.chunks_built or not st.session_state.chunks:
                    from Data.fetch_and_chunk import load_chunks
                    chunks = load_chunks("processed_chunks.pkl")
                    st.session_state.chunks = chunks
                    st.session_state.chunks_built = True
                    from Retrieval.hybrid_retriever import HybridRetriever
                    hybrid = HybridRetriever.load("hybrid_index", fusion="linear", alpha=0.7)
                    st.session_state.hybrid = hybrid

                hybrid = st.session_state.hybrid
                if hybrid is None:
                    from Retrieval.hybrid_retriever import HybridRetriever
                    hybrid = HybridRetriever(fusion="linear", alpha=0.7)
                    hybrid.build(st.session_state.chunks, api_key=api_key)
                    st.session_state.hybrid = hybrid

                retrieval_results = hybrid.search(query, api_key=api_key, top_k=10)
                reranker = Reranker(strategy="llm")
                reranked = reranker.rerank(query, retrieval_results, api_key=api_key, top_k=3)
                generator = AnswerGenerator(model="gpt-4o-mini")
                result = generator.generate(query, reranked, api_key=api_key, reranker=reranker)

                try:
                    from evaluation.evaluator import Evaluator
                    Evaluator().evaluate_retrieval(query, retrieval_results, [], k=5)
                except Exception:
                    pass

        except Exception as e:
            result = {"answer": f"Something went wrong: {str(e)}", "is_emergency": False, "sources": [], "query": query, "model": "error"}

    st.session_state.conversation.append({"query": query, "result": result})
    st.rerun()
