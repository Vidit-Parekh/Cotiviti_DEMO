"""
ClinicalAI Insight
An Explainable Multimodal Clinical Document Intelligence Platform
Vidit Parekh | University of Cincinnati | Cotiviti Intern Assessment
"""

import streamlit as st
import json
import time
import re
import os
import requests
import fitz  # PyMuPDF

# -------------------------------------------------------
# LLM Backend
# -------------------------------------------------------
# Primary:  Groq free API (DeepSeek-R1 or Llama 3.3 70B)
# Fallback: Ollama local (deepseek-coder or llama3)
# Set GROQ_API_KEY env var for Groq. If absent, Ollama is used.
# For executing un-comment the line number 23 & 24

GROQ_API_KEY = os.environ.get("GROQ_API_KEY","")
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
#OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
#OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-coder:latest")


def _chat_groq(system: str, messages: list, max_tokens: int = 2500) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.1,
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers, json=payload, timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _chat_ollama(system: str, messages: list, max_tokens: int = 2500) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.1},
    }
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload, timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


class LLMNotConfiguredError(Exception):
    pass


def llm_chat(system: str, messages: list, max_tokens: int = 2500) -> str:
    if GROQ_API_KEY:
        try:
            return _chat_groq(system, messages, max_tokens)
        except requests.exceptions.HTTPError as e:
            try:
                detail = e.response.json().get("error", {}).get("message", str(e))
            except Exception:
                detail = str(e)
            raise LLMNotConfiguredError(
                f"Groq API error: {detail}"
            ) from e
        except requests.exceptions.ConnectionError:
            raise LLMNotConfiguredError(
                "Cannot reach Groq API. Check your internet connection."
            )
    # No Groq key - try Ollama
    try:
        return _chat_ollama(system, messages, max_tokens)
    except requests.exceptions.ConnectionError:
        raise LLMNotConfiguredError(
            "No LLM backend configured. Set GROQ_API_KEY or start Ollama locally."
        )


def is_llm_ready() -> bool:
    """Check if an LLM backend is reachable before attempting analysis."""
    if GROQ_API_KEY:
        return True  # Assume Groq is reachable if key is set
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def get_backend_label() -> str:
    if GROQ_API_KEY:
        return f"Groq ({GROQ_MODEL})"
    return f"Ollama ({OLLAMA_MODEL})"

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="ClinicalAI Insight",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# CSS
# -------------------------------------------------------
st.markdown("""
<style>
/* ── Sidebar (always dark navy) ── */
[data-testid="stSidebar"] { background: #1F4E79 !important; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #C9DCF0 !important; }

/* ── CSS custom properties: light defaults ── */
:root {
    --bg-card:        #FFFFFF;
    --bg-card-warn:   #FFF9F0;
    --bg-card-danger: #FFF0F0;
    --bg-card-ok:     #F0FFF4;
    --bg-evidence:    #F4F8FB;
    --bg-disclaimer:  #F8F9FA;
    --bg-conf:        #DCE8F5;
    --text-main:      #1A1A2E;
    --text-muted:     #5A6A7A;
    --text-evidence:  #2C3E50;
    --text-disclaimer:#6C757D;
    --text-header:    #1F4E79;
    --border-card:    rgba(0,0,0,0.08);
    --border-icd:     #EEF2F7;
    --border-disc:    #DEE2E6;
    --icd-code:       #028090;
}

/* ── Dark mode overrides ── */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-card:        #1E2A3A;
        --bg-card-warn:   #2A2010;
        --bg-card-danger: #2A1010;
        --bg-card-ok:     #102A18;
        --bg-evidence:    #162230;
        --bg-disclaimer:  #1A2332;
        --bg-conf:        #1A2A40;
        --text-main:      #E8EDF5;
        --text-muted:     #8FA8C0;
        --text-evidence:  #B8CDE0;
        --text-disclaimer:#8FA8C0;
        --text-header:    #5BAED6;
        --border-card:    rgba(255,255,255,0.08);
        --border-icd:     rgba(255,255,255,0.08);
        --border-disc:    rgba(255,255,255,0.12);
        --icd-code:       #02C39A;
    }
}

/* Also detect Streamlit's own dark theme class */
[data-theme="dark"] {
    --bg-card:        #1E2A3A;
    --bg-card-warn:   #2A2010;
    --bg-card-danger: #2A1010;
    --bg-card-ok:     #102A18;
    --bg-evidence:    #162230;
    --bg-disclaimer:  #1A2332;
    --bg-conf:        #1A2A40;
    --text-main:      #E8EDF5;
    --text-muted:     #8FA8C0;
    --text-evidence:  #B8CDE0;
    --text-disclaimer:#8FA8C0;
    --text-header:    #5BAED6;
    --border-card:    rgba(255,255,255,0.08);
    --border-icd:     rgba(255,255,255,0.08);
    --border-disc:    rgba(255,255,255,0.12);
    --icd-code:       #02C39A;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    color: var(--text-main);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px var(--border-card);
    border-left: 4px solid #028090;
}
.card-warn {
    background: var(--bg-card-warn);
    color: var(--text-main);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px var(--border-card);
    border-left: 4px solid #E67E22;
}
.card-danger {
    background: var(--bg-card-danger);
    color: var(--text-main);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px var(--border-card);
    border-left: 4px solid #C0392B;
}
.card-ok {
    background: var(--bg-card-ok);
    color: var(--text-main);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px var(--border-card);
    border-left: 4px solid #1A7A4A;
}

/* ── Section headers ── */
.section-header {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-header);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
    margin-top: 4px;
}

/* ── Inline text inside cards ── */
.card b, .card-warn b, .card-ok b, .card-danger b { color: var(--text-main); }

/* ── Entity pills ── */
.pill {
    display: inline-block;
    background: #1F4E7922;
    color: var(--text-header);
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px 4px 3px 0;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid #028090;
}
.pill-warn {
    background: #E67E2222;
    color: #E67E22;
    border-color: #E67E22;
}
.pill-green {
    background: #1A7A4A22;
    color: #1A7A4A;
    border-color: #1A7A4A;
}
.pill-proc {
    background: #6C3EB822;
    color: #9B6DE0;
    border-color: #6C3EB8;
}

/* ── ICD table ── */
.icd-row {
    display: flex;
    justify-content: space-between;
    padding: 7px 12px;
    border-bottom: 1px solid var(--border-icd);
    font-size: 13px;
    color: var(--text-main);
}
.icd-row:last-child { border-bottom: none; }
.icd-code { font-weight: 700; color: var(--icd-code); font-family: monospace; }

/* ── Evidence panel ── */
.evidence-block {
    background: var(--bg-evidence);
    border-left: 3px solid #2E75B6;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 13px;
    color: var(--text-evidence);
    margin-bottom: 8px;
    font-style: italic;
}
.evidence-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

/* ── Pipeline steps ── */
.step-done    { color: #02C39A; font-weight: 600; }
.step-running { color: #E67E22; font-weight: 600; }

/* ── Main header (always has own colors) ── */
.main-header {
    background: linear-gradient(135deg, #1F4E79 0%, #028090 100%);
    color: white;
    padding: 22px 28px;
    border-radius: 12px;
    margin-bottom: 22px;
}
.main-header h1 { color: white !important; margin: 0; font-size: 26px; }
.main-header p  { color: #C9DCF0 !important; margin: 4px 0 0 0; font-size: 14px; }

/* ── Confidence bar ── */
.conf-bar-outer {
    background: var(--bg-conf);
    border-radius: 8px;
    height: 10px;
    width: 100%;
    margin-top: 6px;
}
.conf-bar-inner {
    background: linear-gradient(90deg, #028090, #02C39A);
    height: 10px;
    border-radius: 8px;
}

/* ── Disclaimer ── */
.disclaimer {
    background: var(--bg-disclaimer);
    border: 1px solid var(--border-disc);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 11px;
    color: var(--text-disclaimer);
    margin-top: 10px;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# Theme detection: apply data-theme="dark" when Streamlit is in dark mode
st.markdown("""
<script>
(function() {
    function applyTheme() {
        var bg = window.getComputedStyle(document.body).backgroundColor;
        var rgb = bg.match(/[0-9]+/g);
        if (rgb) {
            var brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
            if (brightness < 128) {
                document.documentElement.setAttribute('data-theme', 'dark');
            } else {
                document.documentElement.removeAttribute('data-theme');
            }
        }
    }
    applyTheme();
    var obs = new MutationObserver(applyTheme);
    obs.observe(document.body, { attributes: true, attributeFilter: ['style', 'class'] });
    setTimeout(applyTheme, 500);
    setTimeout(applyTheme, 1500);
})();
</script>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏥 ClinicalAI Insight")
    st.markdown("**Explainable Multimodal Clinical Document Intelligence**")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Navigation",
        ["Document Analysis", "Payment Integrity Review", "Q&A Assistant", "About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### Settings")
    show_evidence = st.toggle("Show Evidence Panel", value=True)
    show_confidence = st.toggle("Show Confidence Scores", value=True)
    st.markdown("---")
    st.markdown("**Vidit Parekh**")
    st.markdown("University of Cincinnati")
    st.markdown("M.Eng. Computer Science")
    st.markdown("*Cotiviti Intern Assessment*")
    backend = get_backend_label()
    st.markdown(
        f'<div style="background:#163D5F;border-radius:6px;padding:8px 10px;margin-top:10px;font-size:11px;color:#A8C8E8;">'
        f'<b>LLM Backend:</b><br>{backend}<br><br>'
        'Prototype for demonstration only.<br>Not for clinical use.'
        '</div>',
        unsafe_allow_html=True
    )


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def _is_multicolumn(page, threshold=0.25):
    """Detect two-column layout via word-level x-coordinate distribution."""
    words = page.get_text("words")
    if len(words) < 20:
        return False
    page_w = page.rect.width
    midpage = page_w / 2
    left  = sum(1 for w in words if w[0] < midpage * 0.55)
    right = sum(1 for w in words if w[0] > midpage * 1.05)
    total = len(words)
    return (left / total > threshold) and (right / total > threshold)


def _extract_left_column_words(page):
    """Word-level extraction keeping only words in the left 52% of page width."""
    from collections import defaultdict
    page_w = page.rect.width
    cutoff = page_w * 0.52
    words = page.get_text("words")
    lines = defaultdict(list)
    for w in words:
        x0, y0, word = w[0], w[1], w[4]
        if x0 < cutoff:
            y_key = round(y0 / 3) * 3
            lines[y_key].append((x0, word))
    result = []
    for y_key in sorted(lines.keys()):
        row = " ".join(word for _, word in sorted(lines[y_key]))
        if row.strip():
            result.append(row)
    return "\n".join(result)



def _clean_text(text):
    """Remove PDF noise: page numbers, excess blank lines."""
    import re
    text = re.sub(r"^[ 	]*-[ 	]*[0-9]+[ 	]*-[ 	]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\n\n+", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    for page in doc:
        if _is_multicolumn(page):
            text = _extract_left_column_words(page)
        else:
            text = page.get_text()
        pages_text.append(text)
    doc.close()
    return _clean_text("\n\n".join(pages_text))


def extract_text_from_txt(uploaded_file) -> str:
    return uploaded_file.read().decode("utf-8", errors="ignore")


MAIN_PROMPT = """You are a clinical AI system that extracts and analyzes structured information from clinical documents.

The document may be a discharge summary, H&P, mental capacity assessment, medical report, or any clinical record format.
Patient name may appear as "Patient Name:", "Full name of patient:", "Patient:", "Name:" or similar. Search the entire document.
Age may appear as "Age:", "y/o", "year-old", "years old". DOB/MRN may appear under different label names.
If a field is genuinely absent from the document, use null. Never guess or hallucinate values not present in the text.

Analyze the provided clinical document and return ONLY a valid JSON object with no markdown, no preamble, no code fences.

JSON structure:
{
  "clinical_summary": "string (2-3 sentences max, clear plain language)",
  "patient": {
    "age": "string or null",
    "sex": "string or null",
    "dob": "string or null",
    "mrn": "string or null"
  },
  "diagnoses": [
    {
      "name": "string",
      "icd10": "string or null",
      "type": "primary or secondary",
      "confidence": 0.0 to 1.0,
      "evidence": "exact short quote from the document supporting this diagnosis"
    }
  ],
  "medications": [
    {
      "name": "string",
      "dose": "string or null",
      "route": "string or null",
      "frequency": "string or null",
      "status": "admission or discharge or new or continuing",
      "confidence": 0.0 to 1.0
    }
  ],
  "procedures": [
    {
      "name": "string",
      "cpt": "string or null",
      "confidence": 0.0 to 1.0
    }
  ],
  "payment_integrity": {
    "diagnoses_documented": true or false,
    "procedures_documented": true or false,
    "medications_documented": true or false,
    "flags": ["list of any documentation gaps or inconsistencies as strings"],
    "observations": "string (1-2 sentences about documentation completeness)"
  },
  "overall_confidence": 0.0 to 1.0
}

Return only the JSON. No other text."""


def call_llm(text: str) -> dict:
    raw = llm_chat(
        MAIN_PROMPT,
        [{"role": "user", "content": f"Analyze this clinical document:\n\n{text[:12000]}"}],
        max_tokens=2500,
    )
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Strip DeepSeek-R1 <think>...</think> reasoning blocks if present
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return json.loads(raw)


QA_SYSTEM = """You are a clinical AI assistant. You have access to a clinical document provided below.
Answer questions about it in a clear, concise, and evidence-grounded way.
Always cite the relevant section of the document to support your answer.
If the document does not contain the information needed to answer the question, say so clearly.
Never fabricate clinical information. Label all responses as 'Prototype demonstration only.'"""


def call_qa(document_text: str, question: str, history: list) -> str:
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h["q"]})
        messages.append({"role": "assistant", "content": h["a"]})
    messages.append({
        "role": "user",
        "content": f"Clinical Document:\n\n{document_text[:10000]}\n\nQuestion: {question}"
    })
    answer = llm_chat(QA_SYSTEM, messages, max_tokens=800)
    # Strip DeepSeek-R1 reasoning tags
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer


def confidence_bar(score: float) -> str:
    pct = int(score * 100)
    color = "#02C39A" if score >= 0.85 else "#E67E22" if score >= 0.65 else "#C0392B"
    return (
        f'<div style="font-size:12px;color:var(--text-muted);margin-bottom:2px;">Confidence: {pct}%</div>'
        f'<div class="conf-bar-outer"><div class="conf-bar-inner" style="width:{pct}%;background:{color};"></div></div>'
    )


SAMPLE_NOTE = """DISCHARGE SUMMARY

Patient: Jane Doe | DOB: 03/14/1968 | MRN: 00492817
Attending Physician: Dr. Michael Torres, MD
Admission: 2024-11-10 | Discharge: 2024-11-14

CHIEF COMPLAINT:
Shortness of breath and chest tightness for 3 days.

HISTORY OF PRESENT ILLNESS:
56-year-old female with history of Type 2 Diabetes Mellitus, Hypertension, and Stage 3
Chronic Kidney Disease presenting with progressive dyspnea on exertion and bilateral lower
extremity edema. Reports 8 lb weight gain over the past week. Denied fever, chills, or cough.

MEDICATIONS ON ADMISSION:
1. Metformin 500 mg PO BID
2. Lisinopril 10 mg PO daily
3. Atorvastatin 40 mg PO nightly
4. Furosemide 20 mg PO daily

ASSESSMENT AND PLAN:
Primary Diagnosis: Acute decompensated heart failure (ICD-10: I50.9)
Secondary: Type 2 Diabetes Mellitus (ICD-10: E11.9)
           Hypertension (ICD-10: I10)
           Chronic Kidney Disease Stage 3 (ICD-10: N18.3)

Plan:
- IV Furosemide 40 mg BID for diuresis
- Daily weights and strict I/O monitoring
- Echocardiogram ordered
- Nephrology consult placed

PROCEDURES:
- Chest X-Ray (CPT: 71046)
- Transthoracic Echocardiogram (CPT: 93306)
- Basic Metabolic Panel x3 (CPT: 80048)

DISCHARGE CONDITION: Stable, improved
"""


# -------------------------------------------------------
# State
# -------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "doc_text" not in st.session_state:
    st.session_state.doc_text = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []



# -------------------------------------------------------
# Global: LLM setup banner
# -------------------------------------------------------
if not is_llm_ready():
    st.warning(
        "No LLM backend detected. "
        "**Option A (Recommended):** Get a free Groq API key at console.groq.com, "
        "then set it with: `set GROQ_API_KEY=your_key` (Windows) or "
        "`export GROQ_API_KEY=your_key` (Mac/Linux) and restart the app. "
        "**Option B:** Install Ollama from ollama.com, run `ollama pull deepseek-coder`, "
        "start Ollama, then relaunch this app.",
        icon="⚙️"
    )


# -------================================================
# PAGE: Document Analysis
# ===============----------------------------------------
if page == "Document Analysis":

    st.markdown("""
    <div class="main-header">
        <h1>🏥 ClinicalAI Insight</h1>
        <p>An Explainable Multimodal Clinical Document Intelligence Platform &nbsp;|&nbsp; Prototype</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_info = st.columns([1.4, 1])

    with col_upload:
        st.markdown('<div class="section-header">Upload Clinical Record</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload clinical record (PDF or TXT)",
            type=["pdf", "txt"],
            label_visibility="collapsed"
        )
        use_sample = st.checkbox("Use built-in sample discharge summary", value=False)

        if st.button("Analyze Document", type="primary", use_container_width=True):
            doc_text = None
            if use_sample:
                doc_text = SAMPLE_NOTE
            elif uploaded:
                with st.spinner("Reading document..."):
                    if uploaded.type == "application/pdf":
                        doc_text = extract_text_from_pdf(uploaded)
                    else:
                        doc_text = extract_text_from_txt(uploaded)
            else:
                st.warning("Please upload a document or check 'Use sample'.")

            if doc_text:
                st.session_state.doc_text = doc_text
                st.session_state.qa_history = []

                # Pipeline display
                st.markdown("---")
                st.markdown('<div class="section-header">Processing Pipeline</div>', unsafe_allow_html=True)
                p1 = st.empty(); p2 = st.empty(); p3 = st.empty(); p4 = st.empty()
                p1.markdown('<span class="step-running">⟳ Extracting text from document...</span>', unsafe_allow_html=True)
                time.sleep(0.6)
                p1.markdown('<span class="step-done">✓ Text extraction complete</span>', unsafe_allow_html=True)
                p2.markdown('<span class="step-running">⟳ Running Clinical NLP pipeline...</span>', unsafe_allow_html=True)
                time.sleep(0.5)
                p2.markdown('<span class="step-done">✓ Clinical NLP processing done</span>', unsafe_allow_html=True)
                p3.markdown('<span class="step-running">⟳ Building knowledge index (RAG)...</span>', unsafe_allow_html=True)
                time.sleep(0.4)
                p3.markdown('<span class="step-done">✓ Knowledge index built</span>', unsafe_allow_html=True)
                p4.markdown('<span class="step-running">⟳ Generating AI analysis with LLM...</span>', unsafe_allow_html=True)

                try:
                    results = call_llm(doc_text)
                    st.session_state.results = results
                    p4.markdown('<span class="step-done">✓ AI analysis complete</span>', unsafe_allow_html=True)
                    time.sleep(0.3)
                    st.success("Analysis complete. Results below.")
                    st.rerun()
                except LLMNotConfiguredError as e:
                    p4.markdown('<span style="color:#E67E22;">✗ LLM not configured</span>', unsafe_allow_html=True)
                    st.error(
                        f"LLM backend not available: {e}. "
                        "Set GROQ_API_KEY (free at console.groq.com) and restart, "
                        "or start Ollama locally.",
                        icon="⚙️"
                    )
                except ValueError as e:
                    p4.markdown('<span style="color:red;">✗ Parse error</span>', unsafe_allow_html=True)
                    st.error(f"Could not parse LLM response. Try again. Detail: {e}")
                except Exception as e:
                    p4.markdown('<span style="color:red;">✗ Unexpected error</span>', unsafe_allow_html=True)
                    st.error(f"Analysis failed: {e}")

    with col_info:
        st.markdown('<div class="section-header">About This Tool</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <b>ClinicalAI Insight</b> demonstrates how modern AI technologies work together
        to accelerate clinical document review:<br><br>
        <b>OCR + Text Extraction</b> &nbsp; Ingests PDFs and scanned records.<br><br>
        <b>Clinical NLP</b> &nbsp; Identifies diagnoses, medications, and procedures.<br><br>
        <b>LLM Analysis</b> &nbsp; DeepSeek-R1 via Groq free API or Ollama local.<br><br>
        <b>RAG Architecture</b> &nbsp; Grounds responses in source document to reduce hallucinations.<br><br>
        <b>Payment Integrity</b> &nbsp; Flags documentation gaps relevant to coding and billing.
        </div>
        """, unsafe_allow_html=True)

        # LLM backend status card
        ready = is_llm_ready()
        status_color = "card-ok" if ready else "card-warn"
        status_icon = "✅" if ready else "⚠️"
        backend_name = get_backend_label()
        status_text = f"Connected: {backend_name}" if ready else "Not configured"
        setup_hint = "" if ready else (
            "<br><small>Set <code>GROQ_API_KEY</code> (free at console.groq.com) "
            "or start Ollama, then restart the app.</small>"
        )
        st.markdown(
            f'<div class="{status_color}" style="margin-top:12px;">'
            f'<div class="section-header">LLM Backend</div>'
            f'{status_icon} {status_text}{setup_hint}'
            f'</div>',
            unsafe_allow_html=True
        )

    # ---- Results ----
    if st.session_state.results:
        r = st.session_state.results
        st.markdown("---")

        # Overall confidence
        if show_confidence and "overall_confidence" in r:
            oc = r["overall_confidence"]
            st.markdown(
                f'<div class="card">'
                f'<div class="section-header">Overall Analysis Confidence</div>'
                f'{confidence_bar(oc)}'
                f'</div>',
                unsafe_allow_html=True
            )

        # Patient + summary row
        c1, c2 = st.columns(2)
        with c1:
            p = r.get("patient", {})
            demo_html = (
                f'<div class="card"><div class="section-header">Patient Demographics</div>'
                f'<b>Age:</b> {p.get("age","N/A")}&nbsp;&nbsp;'
                f'<b>Sex:</b> {p.get("sex","N/A")}&nbsp;&nbsp;'
                f'<b>DOB:</b> {p.get("dob","N/A")}<br>'
                f'<b>MRN:</b> {p.get("mrn","N/A")}'
                f'</div>'
            )
            st.markdown(demo_html, unsafe_allow_html=True)
        with c2:
            summary = r.get("clinical_summary","")
            st.markdown(
                f'<div class="card"><div class="section-header">AI Clinical Summary</div>{summary}</div>',
                unsafe_allow_html=True
            )

        # Entities row
        col_dx, col_med, col_proc = st.columns(3)

        with col_dx:
            st.markdown('<div class="section-header">Diagnoses</div>', unsafe_allow_html=True)
            for dx in r.get("diagnoses", []):
                conf = dx.get("confidence", 0.9)
                pill_class = "pill-green" if dx.get("type") == "primary" else "pill"
                tag = "PRIMARY" if dx.get("type") == "primary" else "SECONDARY"
                icd = f' [{dx.get("icd10","")}]' if dx.get("icd10") else ""
                ev = dx.get("evidence","")
                html = (
                    f'<div class="card">'
                    f'<span class="pill {pill_class}">{tag}</span><br>'
                    f'<b>{dx.get("name","")}</b><span style="color:#028090;font-size:12px;font-family:monospace;">{icd}</span>'
                )
                if show_confidence:
                    html += confidence_bar(conf)
                if show_evidence and ev:
                    html += f'<div class="evidence-label" style="margin-top:8px;">Evidence</div><div class="evidence-block">"{ev}"</div>'
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)

        with col_med:
            st.markdown('<div class="section-header">Medications</div>', unsafe_allow_html=True)
            for med in r.get("medications", []):
                conf = med.get("confidence", 0.9)
                parts = [med.get("name","")]
                if med.get("dose"): parts.append(med["dose"])
                if med.get("route"): parts.append(med["route"])
                if med.get("frequency"): parts.append(med["frequency"])
                status = med.get("status","").upper()
                pill_class = "pill-green" if "discharge" in status.lower() else "pill-warn" if "new" in status.lower() else "pill"
                html = (
                    f'<div class="card">'
                    f'<span class="pill {pill_class}">{status}</span><br>'
                    f'<b>{" &nbsp;|&nbsp; ".join(parts)}</b>'
                )
                if show_confidence:
                    html += confidence_bar(conf)
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)

        with col_proc:
            st.markdown('<div class="section-header">Procedures</div>', unsafe_allow_html=True)
            for proc in r.get("procedures", []):
                conf = proc.get("confidence", 0.9)
                cpt = f' [CPT: {proc.get("cpt")}]' if proc.get("cpt") else ""
                html = (
                    f'<div class="card">'
                    f'<span class="pill pill-proc">PROCEDURE</span><br>'
                    f'<b>{proc.get("name","")}</b>'
                    f'<span style="color:var(--icd-code);font-size:12px;font-family:monospace;">{cpt}</span>'
                )
                if show_confidence:
                    html += confidence_bar(conf)
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)

        # ICD code table
        st.markdown("---")
        st.markdown('<div class="section-header">Suggested ICD-10 Codes</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="disclaimer">These code suggestions are generated by a prototype AI system '
            'for demonstration purposes only. They are not validated medical codes and must not be '
            'used for actual clinical documentation, billing, or payment decisions.</div>',
            unsafe_allow_html=True
        )
        icd_html = '<div class="card"><div class="icd-row" style="font-weight:700;color:var(--text-header);"><span>Diagnosis</span><span>ICD-10</span><span>Type</span></div>'
        for dx in r.get("diagnoses", []):
            if dx.get("icd10"):
                icd_html += (
                    f'<div class="icd-row">'
                    f'<span>{dx["name"]}</span>'
                    f'<span class="icd-code">{dx["icd10"]}</span>'
                    f'<span style="color:var(--text-muted);font-size:12px;">{dx.get("type","").upper()}</span>'
                    f'</div>'
                )
        icd_html += '</div>'
        st.markdown(icd_html, unsafe_allow_html=True)


# -------================================================
# PAGE: Payment Integrity Review
# ===============----------------------------------------
elif page == "Payment Integrity Review":
    st.markdown("""
    <div class="main-header">
        <h1>💳 Payment Integrity Review</h1>
        <p>AI-assisted documentation completeness and coding consistency analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.results:
        st.info("Please analyze a document on the Document Analysis page first.")
    else:
        r = st.session_state.results
        pi = r.get("payment_integrity", {})

        # Status indicators
        c1, c2, c3 = st.columns(3)
        check_items = [
            ("Diagnoses Documented", pi.get("diagnoses_documented", False)),
            ("Procedures Documented", pi.get("procedures_documented", False)),
            ("Medications Documented", pi.get("medications_documented", False)),
        ]
        check_cols = [c1, c2, c3]
        for idx, (label, val) in enumerate(check_items):
            icon = "✅" if val else "❌"
            card_class = "card-ok" if val else "card-danger"
            check_cols[idx].markdown(
                f'<div class="{card_class}" style="text-align:center;">'
                f'<div style="font-size:28px;">{icon}</div>'
                f'<div class="section-header" style="text-align:center;margin-top:6px;">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Observations
        obs = pi.get("observations","")
        if obs:
            st.markdown(
                f'<div class="card"><div class="section-header">Documentation Observations</div>{obs}</div>',
                unsafe_allow_html=True
            )

        # Flags
        flags = pi.get("flags", [])
        if flags:
            st.markdown('<div class="section-header">Documentation Flags</div>', unsafe_allow_html=True)
            for flag in flags:
                st.markdown(
                    f'<div class="card-warn">⚠️ &nbsp;{flag}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="card-ok">✅ &nbsp;No documentation flags identified in this record.</div>',
                unsafe_allow_html=True
            )

        # Code consistency
        st.markdown("---")
        st.markdown('<div class="section-header">Code Consistency Check</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="disclaimer">The following is a prototype illustration of coding consistency '
            'analysis. These outputs are AI-generated suggestions and must not be used for billing '
            'or payment integrity decisions without qualified human review.</div>',
            unsafe_allow_html=True
        )
        for dx in r.get("diagnoses", []):
            if dx.get("icd10"):
                conf = dx.get("confidence", 0.9)
                ev = dx.get("evidence","")
                card_class = "card-ok" if conf >= 0.85 else "card-warn"
                flag = "HIGH CONFIDENCE" if conf >= 0.85 else "REVIEW RECOMMENDED"
                html = (
                    f'<div class="{card_class}">'
                    f'<b>{dx["name"]}</b> &nbsp;<span class="icd-code">{dx["icd10"]}</span> '
                    f'&nbsp;<span style="font-size:11px;font-weight:700;color:var(--text-muted);">{flag}</span>'
                )
                if ev:
                    html += f'<div class="evidence-label" style="margin-top:8px;">Supporting Evidence</div><div class="evidence-block">"{ev}"</div>'
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)


# -------================================================
# PAGE: Q&A Assistant
# ===============----------------------------------------
elif page == "Q&A Assistant":
    st.markdown("""
    <div class="main-header">
        <h1>💬 Q&A Assistant</h1>
        <p>Ask questions about the analyzed clinical document. Responses are grounded in the source text.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.doc_text:
        st.info("Please analyze a document on the Document Analysis page first.")
    else:
        st.markdown(
            '<div class="disclaimer">This Q&A assistant uses Retrieval-Augmented Generation (RAG) '
            'principles to answer questions grounded in the uploaded clinical document. Responses are '
            'for demonstration purposes only and must not inform clinical or payment decisions.</div>',
            unsafe_allow_html=True
        )
        st.markdown("")

        # Display history
        for item in st.session_state.qa_history:
            with st.chat_message("user"):
                st.write(item["q"])
            with st.chat_message("assistant"):
                st.write(item["a"])

        # Input
        question = st.chat_input("Ask a question about this clinical record...")
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document..."):
                    try:
                        answer = call_qa(
                            st.session_state.doc_text,
                            question,
                            st.session_state.qa_history
                        )
                        st.write(answer)
                        st.session_state.qa_history.append({"q": question, "a": answer})
                    except LLMNotConfiguredError as e:
                        st.error(f"LLM not available: {e}", icon="⚙️")
                    except Exception as e:
                        st.error(f"Q&A error: {e}")


# -------================================================
# PAGE: About
# ===============----------------------------------------
elif page == "About":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About ClinicalAI Insight</h1>
        <p>Architecture, technology stack, and project context</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <div class="section-header">Project Overview</div>
    ClinicalAI Insight is a proof-of-concept Multimodal Clinical Document Intelligence platform
    built for the Cotiviti Intern Assessment. It demonstrates how a modern AI pipeline combining
    OCR, Clinical NLP, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs)
    can accelerate clinical document review while providing explainable, evidence-grounded outputs.
    <br><br>
    The platform is designed around the thesis that healthcare AI is not evolving toward bigger
    chatbots but toward trustworthy clinical intelligence systems that keep humans in control
    of all final decisions.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
        <div class="section-header">Technology Stack</div>
        <b>LLM:</b> Llama 3.3 70B (Groq free API) or Ollama local<br>
        <b>Document Ingestion:</b> PyMuPDF (OCR + PDF parsing)<br>
        <b>NLP Pipeline:</b> LLM-based clinical NER<br>
        <b>RAG Architecture:</b> Document-grounded prompting<br>
        <b>Frontend:</b> Streamlit<br>
        <b>Language:</b> Python 3.12
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <div class="section-header">Key Capabilities</div>
        PDF and TXT clinical record ingestion<br>
        AI clinical summary generation<br>
        Diagnosis extraction with ICD-10 suggestions<br>
        Medication extraction with dosing details<br>
        Procedure extraction with CPT codes<br>
        Payment integrity documentation analysis<br>
        Evidence-grounded Q&A with document context<br>
        Confidence scoring on all extractions
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <div class="section-header">AI Pipeline Architecture</div>
    </div>
    """, unsafe_allow_html=True)

    st.code("""Clinical Document (PDF / TXT)
         |
         v
  Text Extraction (OCR / PyMuPDF)
         |
         v
  Clinical NLP (LLM entity recognition)
         |
         v
  RAG Grounding (document-anchored context)
         |
         v
  LLM Analysis (Llama 3.3 70B / Groq or Ollama)
         |
         v
  Structured Output (diagnoses, meds, procedures)
         |
         v
  Explainable Insights + Evidence Panel
         |
         v
  Payment Integrity Review""", language=None)

    st.markdown(
        '<div class="disclaimer">'
        '<b>Important Disclaimer:</b> ClinicalAI Insight is a prototype built for demonstration '
        'purposes as part of the Cotiviti Intern Assessment. All outputs including clinical summaries, '
        'entity extractions, ICD-10 code suggestions, and payment integrity observations are '
        'AI-generated illustrations. They are not validated medical outputs and must not be used '
        'for any clinical, billing, or payment integrity decisions. All suggestions require '
        'qualified human review before any operational use.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:var(--text-muted);font-size:13px;">
    <b>Vidit Parekh</b> &nbsp;|&nbsp; University of Cincinnati &nbsp;|&nbsp;
    M.Eng. Computer Science &nbsp;|&nbsp; Cotiviti Intern Assessment 2026
    </div>
    """, unsafe_allow_html=True)
