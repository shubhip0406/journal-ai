import json
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st

# ---- GCP credentials from Streamlit Secrets ----
GCP = st.secrets["gcp"]
PROJECT_ID = GCP["project_id"]
LOCATION = GCP.get("location", "us-central1")
VERTEX_MODEL = GCP.get("vertex_model", "gemini-1.5-flash")
SERVICE_ACCOUNT_JSON = GCP["service_account_json"]

from google.oauth2 import service_account
from google.cloud import firestore
from vertexai.generative_models import GenerativeModel
import vertexai

# Build creds from secrets (no file needed)
creds = service_account.Credentials.from_service_account_info(json.loads(SERVICE_ACCOUNT_JSON))
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
db = firestore.Client(project=PROJECT_ID, credentials=creds)

# ---- Constants ----
PROMPTS = [
    "How’s your day going?",
    "What’s been on your mind today?",
    "What gave you energy this week?",
    "What’s one small win you had recently?",
    "What’s been draining your energy lately?"
]
FALLBACK = "Just write whatever’s on your mind."

SYSTEM_PROMPT = (
    "You are a careful mental-health journaling assistant. "
    "Summarize neutrally and supportively without diagnosing."
    " Return STRICT JSON with keys: summary (2-3 sentences), "
    "themes (array of objects with keys name, description), "
    "suggested_prompts (array with 1 short reflective question)."
)

GEN_TEMPLATE = """System:
{system}

User:
JOURNAL:
\"\"\"{text}\"\"\" 

Return strict JSON:
{{
  "summary": "<2-3 sentence recap>",
  "themes": [
    {{"name":"ThemeName1","description":"One sentence."}},
    {{"name":"ThemeName2","description":"One sentence."}}
  ],
  "suggested_prompts": ["One gentle follow-up question"]
}}"""

# ---- Firestore helpers ----
def entries_col(user_id: str):
    return db.collection("users").document(user_id).collection("entries")

def to_title(name: str) -> str:
    return (name or "").strip().title()

def create_entry(user_id: str, text: str, prompt_used: Optional[str]) -> str:
    col = entries_col(user_id)
    doc = col.document()
    doc.set({
        "text": text,
        "prompt_used": prompt_used,
        "created_at": firestore.SERVER_TIMESTAMP,
        "is_shared": False
    })
    return doc.id

def save_summary(user_id: str, entry_id: str, summary: Dict):
    col = entries_col(user_id).document(entry_id).collection("summaries")
    col.add({
        "summary_text": summary.get("summary", ""),
        "themes": summary.get("themes", []),
        "suggested_prompts": summary.get("suggested_prompts", []),
        "model": VERTEX_MODEL,
        "created_at": firestore.SERVER_TIMESTAMP
    })

def fetch_entries(user_id: str, theme_filter: Optional[str] = None) -> List[Dict]:
    col = entries_col(user_id)
    docs = col.order_by("created_at", direction=firestore.Query.DESCENDING).stream()
    out = []
    for d in docs:
        e = d.to_dict()
        e["id"] = d.id
        sdocs = col.document(d.id).collection("summaries") \
                .order_by("created_at", direction=firestore.Query.DESCENDING) \
                .limit(1).stream()
        latest = None
        for s in sdocs:
            latest = s.to_dict()
        e["latest_summary"] = latest
        if theme_filter and latest and "themes" in latest:
            names = {to_title(t.get("name","")) for t in latest["themes"] if isinstance(t, dict)}
            if to_title(theme_filter) not in names:
                continue
        out.append(e)
    return out

def set_share(user_id: str, entry_id: str, is_shared: bool):
    entries_col(user_id).document(entry_id).update({"is_shared": is_shared})

def export_shared(user_id: str) -> Dict:
    col = entries_col(user_id)
    docs = col.where("is_shared", "==", True).order_by("created_at").stream()
    shared = []
    for d in docs:
        e = d.to_dict()
        e["id"] = d.id
        sdocs = col.document(d.id).collection("summaries") \
                .order_by("created_at", direction=firestore.Query.DESCENDING) \
                .limit(1).stream()
        latest = None
        for s in sdocs:
            latest = s.to_dict()
        shared.append({
            "entry_id": d.id,
            "text": e.get("text",""),
            "prompt_used": e.get("prompt_used"),
            "created_at": e.get("created_at").isoformat() if e.get("created_at") else None,
            "summary": None if not latest else latest.get("summary_text"),
            "themes": None if not latest else latest.get("themes")
        })
    return {"user_id": user_id, "shared": shared}

def theme_counts(user_id: str, last_n: int = 10) -> Dict[str, int]:
    col = entries_col(user_id)
    docs = col.order_by("created_at", direction=firestore.Query.DESCENDING).limit(last_n).stream()
    counts = {}
    for d in docs:
        sdocs = col.document(d.id).collection("summaries") \
                .order_by("created_at", direction=firestore.Query.DESCENDING) \
                .limit(1).stream()
        latest = None
        for s in sdocs:
            latest = s.to_dict()
        if latest and "themes" in latest:
            for t in latest["themes"]:
                if isinstance(t, dict):
                    name = to_title(t.get("name",""))
                    if name:
                        counts[name] = counts.get(name, 0) + 1
    return counts

# ---- Model call ----
def summarize_with_gemini(text: str) -> Dict:
    model = GenerativeModel(VERTEX_MODEL, credentials=creds)
    prompt = GEN_TEMPLATE.format(system=SYSTEM_PROMPT, text=text)
    resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    try:
        data = json.loads(resp.text)
    except Exception:
        prompt2 = prompt + "\n\nIMPORTANT: Output MUST be valid JSON only, no commentary."
        resp2 = model.generate_content(prompt2, generation_config={"response_mime_type": "application/json"})
        data = json.loads(resp2.text)
    if "themes" in data and isinstance(data["themes"], list):
        for t in data["themes"]:
            if isinstance(t, dict) and "name" in t:
                t["name"] = to_title(t["name"])
    return data

# ---- UI ----
st.set_page_config(page_title="Journal AI", layout="centered")
st.title("Journal AI")
st.caption("This app supports personal journaling with AI-generated summaries and themes. It is **not therapy** and **not a crisis resource**.")

user_id = st.text_input("User ID", value="me")

# single prompt + shuffle + fallback after 2 refreshes
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = PROMPTS[0]

pc1, pc2 = st.columns([4,1])
with pc1:
    st.caption("Prompt")
    st.subheader(st.session_state.current_prompt)
with pc2:
    if st.button("New prompt"):
        st.session_state.refresh_count += 1
        if st.session_state.refresh_count >= 2:
            st.session_state.current_prompt = FALLBACK
        else:
            import random
            choices = [p for p in PROMPTS if p != st.session_state.current_prompt]
            st.session_state.current_prompt = random.choice(choices)

text = st.text_area("Write your journal entry...", height=160)
sc1, sc2 = st.columns(2)
if sc1.button("Save entry"):
    if text.strip():
        eid = create_entry(user_id, text.strip(), st.session_state.current_prompt)
        st.success(f"Saved entry #{eid}")
        st.session_state.refresh_count = 0
        st.session_state.current_prompt = PROMPTS[0]
    else:
        st.warning("Please write something before saving.")

if sc2.button("Refresh my entries"):
    st.session_state["entries"] = fetch_entries(user_id)

st.divider()
st.subheader("Your entries")
theme_filter = st.text_input("Filter by theme (optional)")
if st.button("Apply theme filter"):
    st.session_state["entries"] = fetch_entries(user_id, theme_filter.strip() or None)

entries = st.session_state.get("entries", [])
for e in entries:
    created = e.get("created_at")
    stamp = created.isoformat() if isinstance(created, datetime) else str(created)
    with st.expander(f"Entry {e['id']} • {stamp} • shared={e.get('is_shared', False)}"):
        st.write(e.get("text",""))
        cc1, cc2 = st.columns(2)
        if cc1.button("Summarize", key=f"summ_{e['id']}"):
            summary = summarize_with_gemini(e.get("text",""))
            save_summary(user_id, e["id"], summary)
            st.session_state[f"summary_{e['id']}"] = summary

            # ---- Agent-like nudge: Try Prompt / Dismiss ----
            counts = theme_counts(user_id, last_n=10)
            hot = next((name for name, c in counts.items() if c >= 3), None)
            if hot:
                st.info(f"I’ve noticed **{hot}** showing up a lot lately. Want a gentle prompt to reflect on it?")
                cta1, cta2 = st.columns(2)
                if cta1.button("Try prompt", key=f"try_{e['id']}"):
                    next_prompt = f"Would you like to explore what's behind your recent {hot.lower()}? What patterns have you noticed?"
                    st.session_state.current_prompt = next_prompt
                    st.session_state.refresh_count = 0
                    st.toast("Prompt loaded above.")
                if cta2.button("Dismiss", key=f"dis_{e['id']}"):
                    st.toast("Got it.")

        shared_now = e.get("is_shared", False)
        new_shared = cc2.toggle("Share", value=shared_now, key=f"share_{e['id']}")
        if new_shared != shared_now:
            set_share(user_id, e["id"], bool(new_shared))
            st.toast("Updated share status.")

        # show latest summary
        summary = st.session_state.get(f"summary_{e['id']}") or e.get("latest_summary")
        if summary:
            st.markdown(f"**Summary:** {summary.get('summary_text') or summary.get('summary','')}")
            themes = summary.get("themes", [])
            if themes:
                st.caption("Themes")
                cols = st.columns(min(5, len(themes)))
                for i, t in enumerate(themes[:5]):
                    with cols[i]:
                        st.button(to_title(t.get("name",""))[:18], key=f"chip_{e['id']}_{i}", disabled=True)

st.divider()
ec1, ec2 = st.columns(2)
if ec1.button("Export shared (JSON)"):
    data = export_shared(user_id)
    st.json(data)

if ec2.button("Theme counts (last 10)"):
    counts = theme_counts(user_id, last_n=10)
    st.json({"user_id": user_id, "last_n": 10, "counts": counts})

