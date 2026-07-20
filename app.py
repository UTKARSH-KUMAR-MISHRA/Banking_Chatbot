import os
import time
import json

import streamlit as st
import pandas as pd

import google.genai as genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

st.set_page_config(page_title="Banking Mitra", page_icon="🏦", layout="wide")

# ---------------------------------------------------------------------------
# DESIGN SYSTEM — "Digital Passbook"
# Palette: navy ledger bg, brass/gold accent, ink-teal secondary
# Type: Fraunces (headings) + Inter (body) + IBM Plex Mono (figures)
# ---------------------------------------------------------------------------
page_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --navy-1: #0b1220;
  --navy-2: #101a2e;
  --navy-3: #16223b;
  --gold: #d4a017;
  --gold-soft: rgba(212,160,23,0.14);
  --ink-teal: #0f766e;
  --ink-teal-soft: rgba(15,118,110,0.16);
  --paper: #f1ede4;
  --paper-dim: #b9c2d6;
  --hairline: rgba(212,160,23,0.22);
}

html, body, .block-container {
  background:
    radial-gradient(circle at 12% -10%, rgba(212,160,23,0.10), transparent 32%),
    radial-gradient(circle at 90% 0%, rgba(15,118,110,0.14), transparent 38%),
    linear-gradient(160deg, var(--navy-1) 0%, var(--navy-2) 45%, var(--navy-1) 100%);
  color: var(--paper);
  font-family: 'Inter', system-ui, sans-serif;
}
.stApp { background: transparent; }

h1, h2, h3, .ledger-title { font-family: 'Fraunces', serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(11,18,32,0.92);
  border-right: 1px solid var(--hairline);
}

/* ---- Ledger hero ---- */
.hero-panel {
  position: relative;
  background: linear-gradient(180deg, rgba(16,26,46,0.96), rgba(11,18,32,0.98));
  border: 1px solid var(--hairline);
  border-radius: 6px;
  padding: 40px 44px;
  box-shadow: 0 30px 70px rgba(0,0,0,0.35);
}
.hero-panel::before {
  content: '';
  position: absolute; inset: 10px;
  border: 1px solid rgba(212,160,23,0.14);
  border-radius: 4px;
  pointer-events: none;
}
.hero-eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  font-size: 0.72rem;
  color: var(--gold);
}
.hero-title {
  font-size: 2.6rem;
  font-weight: 600;
  line-height: 1.12;
  color: var(--paper);
  margin: 10px 0 8px;
}
.hero-desc {
  color: var(--paper-dim);
  font-size: 1.02rem;
  max-width: 640px;
  line-height: 1.6;
}

/* Ledger strip — signature element: looks like passbook entry rows */
.ledger-strip { margin-top: 26px; border-top: 1px solid var(--hairline); }
.ledger-row {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 12px 2px; border-bottom: 1px dashed rgba(212,160,23,0.22);
  font-size: 0.92rem;
}
.ledger-row .label { color: var(--paper-dim); }
.ledger-row .value {
  font-family: 'IBM Plex Mono', monospace;
  color: var(--gold);
  font-weight: 600;
}

/* Feature cards */
.feature-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin-top:24px; }
.feature-card {
  background: rgba(11,18,32,0.55);
  border: 1px solid var(--hairline);
  border-left: 3px solid var(--ink-teal);
  border-radius: 4px;
  padding: 18px 20px;
}
.feature-title { font-weight: 600; color: var(--paper); margin-bottom: 6px; font-size: 0.98rem; }
.feature-text { color: var(--paper-dim); font-size: 0.88rem; line-height: 1.55; }

/* Chat bubbles (now actually wired to the message loop) */
.msg-row { display:flex; margin-bottom:16px; align-items:flex-end; gap:10px; }
.msg-row.user { justify-content:flex-end; }
.msg-row.assistant { justify-content:flex-start; }
.avatar {
  width:32px; height:32px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-size:15px; flex-shrink:0;
}
.avatar.user { background: var(--gold-soft); border:1px solid var(--gold); }
.avatar.assistant { background: var(--ink-teal-soft); border:1px solid var(--ink-teal); }
.chat-bubble-user {
  background: linear-gradient(135deg, rgba(212,160,23,0.20), rgba(212,160,23,0.08));
  color: var(--paper); padding:14px 18px; border-radius:14px 14px 2px 14px;
  border: 1px solid rgba(212,160,23,0.35); max-width: 70%; line-height:1.55;
}
.chat-bubble-assistant {
  background: rgba(255,255,255,0.04); color: var(--paper);
  padding:14px 18px; border-radius:14px 14px 14px 2px;
  border: 1px solid var(--hairline); max-width: 70%; line-height:1.55;
}

/* Typing indicator */
.typing-dots { display:flex; gap:4px; padding:4px 2px; }
.typing-dots span {
  width:6px; height:6px; border-radius:50%; background: var(--ink-teal);
  animation: bounce 1.2s infinite ease-in-out;
}
.typing-dots span:nth-child(2){ animation-delay:.15s; }
.typing-dots span:nth-child(3){ animation-delay:.3s; }
@keyframes bounce { 0%,80%,100%{transform:scale(.6);opacity:.4;} 40%{transform:scale(1);opacity:1;} }

/* Media box */
.media-box {
  background: rgba(16,26,46,0.55); border-radius: 6px;
  border: 1px solid var(--hairline); padding: 22px 24px; margin-top: 18px;
}
.media-box h3 { margin-top:0; font-size:1.05rem; }

/* Sidebar cards */
.sidebar-card {
  background: rgba(16,26,46,0.7); border:1px solid var(--hairline);
  border-radius: 6px; padding: 18px 20px; margin-bottom: 18px;
}
.sidebar-card h2 { font-size:1.15rem; margin:0 0 8px; }
.sidebar-card p { color: var(--paper-dim); font-size:0.88rem; line-height:1.55; margin:0; }
.sidebar-badge { display:flex; flex-wrap:wrap; gap:6px; margin-top:12px; }
.sidebar-badge span {
  font-family:'IBM Plex Mono', monospace; font-size:0.72rem;
  padding:4px 9px; border-radius:3px; background: var(--ink-teal-soft);
  border:1px solid var(--ink-teal); color: var(--paper);
}
.doc-chip { display:flex; align-items:center; gap:8px; padding:6px 4px; font-size:0.86rem; color:var(--paper-dim); }
.doc-chip .dot { color: var(--gold); }

.stButton>button {
  background: linear-gradient(90deg, var(--ink-teal), #115e59) !important;
  color: var(--paper); border: 1px solid rgba(212,160,23,0.3) !important;
  font-weight: 600; border-radius: 4px !important;
}

/* Premium seal badge, top-right of hero */
.hero-seal {
  position: absolute; top: 26px; right: 34px;
  width: 64px; height: 64px; border-radius: 50%;
  border: 2px solid var(--gold);
  display: flex; align-items: center; justify-content: center;
  font-size: 26px;
  background: radial-gradient(circle, rgba(212,160,23,0.18), transparent 70%);
  box-shadow: 0 0 0 4px rgba(212,160,23,0.08);
}

/* Fade-in for new content */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
.msg-row { animation: fadeInUp 0.25s ease; }
.feature-card, .ledger-row { animation: fadeInUp 0.3s ease; }
/* Text-to-speech listen/stop controls reuse the standard .stButton style below */
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "messages": [],
    "last_model": "Waiting for first answer",
    "retrieved_docs": [],
    "audio_bytes": None,
    "camera_bytes": None,
    "audio_mime_type": "audio/wav",
    "camera_enabled_at": 0,
    "microphone_enabled_at": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# API key — ONLY from secrets/env, never hardcoded
# ---------------------------------------------------------------------------
api_key = st.secrets.get("GENAI_API_KEY") if hasattr(st, "secrets") else None
api_key = api_key or os.getenv("GENAI_API_KEY")

if not api_key:
    st.sidebar.error(
        "Google Generative AI API key missing. Add GENAI_API_KEY to Streamlit "
        "secrets (Settings → Secrets on Streamlit Cloud, or .streamlit/secrets.toml locally)."
    )
    st.stop()

client = genai.Client(api_key=api_key)


def build_multimodal_payload(prompt, image_bytes=None, audio_bytes=None, audio_mime_type="audio/wav"):
    parts = []
    if image_bytes is not None:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    if audio_bytes is not None:
        parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime_type))
    prompt_text = (prompt or "Please help me with this banking request.").strip()
    parts.append(
        types.Part.from_text(
            text=(
                f"Question: {prompt_text}\n\n"
                "Answer in Hindi or English using the bank document context and any attached media."
            )
        )
    )
    return parts


def load_knowledge_base():
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": False},
    )
    if os.path.isdir("chroma_db"):
        return Chroma(persist_directory="chroma_db", embedding_function=embedding_function)
    return None


def ensure_knowledge_base_ready():
    """
    chroma_db/ is a build artifact and is intentionally NOT pushed to GitHub
    (it's large and derived from data/). On a fresh deploy (e.g. Streamlit
    Community Cloud), it won't exist yet — so if PDFs are present in data/
    but the index is missing/empty, build it automatically on first boot.
    This only runs once; afterwards chroma_db/ persists on that container.
    """
    chroma_missing_or_empty = (
        not os.path.isdir("chroma_db") or not os.listdir("chroma_db")
    )
    has_pdfs = os.path.isdir("data") and any(
        f.lower().endswith(".pdf") for f in os.listdir("data")
    )
    if chroma_missing_or_empty and has_pdfs:
        from ingest import run_ingestion
        with st.spinner("Pehli baar setup ho raha hai — bank PDFs index kiye ja rahe hain (thoda time lagega)..."):
            run_ingestion()


ensure_knowledge_base_ready()
kb = load_knowledge_base()

data_path = os.path.join(os.getcwd(), "data")
data_files = []
if os.path.isdir(data_path):
    data_files = [f for f in os.listdir(data_path) if f.lower().endswith(".pdf")]

# ---------------------------------------------------------------------------
# FINANCIAL ADVISOR KNOWLEDGE — curated fallback covering common banking
# problems for BOTH rural and urban customers. The model uses PDF context
# first when available, and falls back to this for general/common-knowledge
# banking topics not covered in the uploaded documents.
# ---------------------------------------------------------------------------
FINANCIAL_ADVISOR_KNOWLEDGE = """
BASICS — ACCOUNT & FORMS
- Form kaise bharein: Naam, pata, mobile number, aur signature wahi likhein jo aapke ID proof
  (Aadhaar/PAN) mein hai. Har blank field ko "N/A" ya dash se bhar dein, khali na chhodein.
  Photo aur signature sirf designated box mein lagayein. Submit karne se pehle ek baar sab
  entries dobara padh lein.
- KYC ke liye documents: Identity proof (Aadhaar, PAN, Passport, Voter ID, ya Driving Licence)
  + Address proof (Aadhaar, utility bill, ya Passport) + 2 passport-size photos. PAN card
  zaroori hai agar account mein transactions Rs 50,000 se zyada honi hain.
- Bank account kaise khulega: Nearest branch jaayein ya bank ki app/website se online apply karein
  → Account opening form bharein → KYC documents jama karein → Initial deposit karein (agar
  zero-balance account nahi hai) → Verification ke baad passbook aur debit card milega.
- Nomination facility: Har account mein nominee add karna zaroori hai — account holder ki
  death ki sthiti mein paisa seedha nominee ko milta hai, court process ki zarurat nahi padti.
- Passbook update: Passbook printing machine (branch mein) mein passbook daal kar "Update"
  button dabayein, ya counter par jama karke request karein. Net-banking/app mein bhi
  mini-statement dekha ja sakta hai agar physical update turant nahi chahiye.
- Cheque book/DD: Cheque book branch se, net-banking se, ya SMS/missed-call banking se request
  ho sakti hai. Demand Draft (DD) tab use hota hai jab cheque accept nahi hota (jaise court fees).
- Cheque bounce hone par: Paisa account mein rakhein aur signature/date match karein. Bar-bar
  bounce hone se penalty aur credit score par negative asar padta hai — jitni jaldi ho sake
  dobara clear karayein.

DIGITAL & PAYMENTS
- ATM PIN kaise banega: App mein "Generate Green PIN"/"Set ATM PIN" (registered mobile OTP se),
  ya ATM machine mein "PIN Generation" option chun kar OTP + card details enter karein. Bina
  OTP ke, branch jaakar physical form se bhi set karwaya ja sakta hai.
- Online/net banking kya hai: Website/app se ghar baithe balance check, fund transfer
  (NEFT/RTGS/IMPS), bill payment, statement dekhna. Pehli baar branch se ya app se register
  karna hota hai.
- UPI kaise use karein: Google Pay/PhonePe/bank ka app install karein → mobile number verify
  karein → bank account link karein (debit card ke last 6 digits + expiry se) → UPI PIN set
  karein → ab kisi bhi UPI ID/QR code par payment/receive kar sakte hain.
- Digital fraud se bachna: Bank kabhi phone/SMS/email par OTP, PIN, ya CVV nahi maangta — aisi
  call fraud hoti hai, kabhi share na karein. Kisi bhi unknown link par bank details na daalein.
  Shak hone par turant card/account block karayein aur 1930 (National Cyber Crime helpline)
  par report karein.
- Mobile app/net banking mein dikkat: App update karein, sahi user ID/MPIN use karein, 3 galat
  attempts ke baad ID lock ho sakti hai — unlock ke liye app ke "Forgot password/Reset MPIN"
  ya branch/customer care se help lein.
- Balance check bina smartphone ke: Registered number se bank ke "missed call banking" number
  par missed call dekar balance SMS mil jata hai (number passbook/bank ki website par likha
  hota hai) — smartphone ya internet ki zarurat nahi.

LOANS & CREDIT
- Loan eligibility: Aam taur par stable income/salary slip, minimum credit score (usually
  650+), age 21-60 ke beech, aur kam existing debt-to-income ratio chahiye. Salaried,
  self-employed, aur students alag-alag criteria follow karte hain — exact terms bank policy
  document se confirm karein.
- Credit score kya hai: 300-900 ke beech ka number jo batata hai aap loan/credit card ke
  liye kitne "reliable" hain. Time par EMI/credit card bill bharne se score improve hota hai.
  Free mein CIBIL/Experian report saal mein ek baar check ki ja sakti hai.
- Kisan Credit Card (KCC): Farmers ke liye low-interest short-term credit card, beej/khaad/
  krishi kharche ke liye. Land records + Aadhaar + bank account se apply hota hai, nearest
  rural/semi-urban branch mein.
- Loan against FD: Apni Fixed Deposit ko todhe bina, usi par (FD amount ka ~90% tak) turant
  loan mil sakta hai — emergency ke liye acha option, kam interest rate par.

RURAL BANKING SUPPORT
- Business Correspondent (BC)/Bank Mitra: Jahan branch door hai, wahan gaanv mein hi BC agent
  (chhoti dukaan/kiosk) se Aadhaar-linked fingerprint se paisa jama/nikal sakte hain — poori
  branch tak jaane ki zarurat nahi padti chhote transactions ke liye.
- Jan Dhan Yojana (PMJDY): Zero-balance basic savings account, RuPay debit card aur accident
  insurance ke saath — koi bhi bina existing account ke branch/BC agent se khol sakta hai.
- DBT/Aadhaar seeding: Sarkari subsidy (gas, scholarship, PM Kisan) seedha account mein aane
  ke liye Aadhaar ko bank account se "seed"/link karwana zaroori hai — branch ya net-banking
  se ho sakta hai.
- PM Fasal Bima Yojana: Crop insurance scheme — kharaab mausam/aapda se fasal nuksan hone par
  claim milta hai. Bank ya CSC (Common Service Centre) se enroll hota hai, premium bahut kam
  hota hai.
- Doorstep banking: Senior citizens aur divyang customers ke liye kai banks cash withdrawal/
  deposit, cheque pickup jaisi services ghar par (fees ke saath) provide karte hain — bank ki
  app/toll-free number se book ki ja sakti hain.

PENSION & GOVERNMENT SCHEMES
- Atal Pension Yojana (APY): Unorganized sector workers ke liye retirement pension scheme,
  18-40 age ke beech enroll kar sakte hain.
- Pradhan Mantri Jeevan Jyoti Bima Yojana (PMJJBY): Rs 2 lakh life cover, bahut kam annual
  premium mein.
- Pradhan Mantri Suraksha Bima Yojana (PMSBY): Rs 2 lakh accident cover, sabse sasta premium.
- PM Kisan Samman Nidhi: Farmers ko saal mein Rs 6,000 (3 installments mein) direct account
  mein income support.
  In sab schemes mein branch, net-banking, ya app se enroll kiya ja sakta hai.

PROBLEMS & COMPLAINTS
- ATM se paisa nahi nikla par balance kat gaya: 24 se 48 ghante mein automatically wapas aa
  jata hai (auto-reversal). Agar nahi aata, branch mein written complaint dein ya bank ki
  app/website ke "Failed Transaction" section mein register karein.
- Branch mein staff sahi tarah se help nahi kar raha: Pehle branch manager se baat karein.
  Agar solution na mile, to bank ke toll-free customer care ya app ke "Grievance/Complaint"
  section mein likhit shikayat darj karein — ismein complaint number milta hai jise track
  kiya ja sakta hai.
- Fir bhi solution na mile — RBI Banking Ombudsman: Agar bank 30 din mein jawab na de ya
  jawab santoshjanak na ho, RBI ke Banking Ombudsman ke paas free mein online complaint
  (cms.rbi.org.in) ki ja sakti hai — ye ek independent, sarkari grievance redressal system
  hai.
- Locker facility: Bank branch mein safe deposit locker rent par milta hai gehne/documents
  rakhne ke liye — application + KYC + annual rent + kabhi security deposit/FD lagta hai.
  Availability branch-wise limited hoti hai, waiting list ho sakti hai.
- NRI banking: Videsh mein rehne wale Indians ke liye alag NRE/NRO accounts hote hain — NRE
  mein foreign earning tax-free deposit hoti hai, NRO mein India ki income (rent, dividend)
  rakhi jaati hai. Specialised NRI branch ya net-banking se manage ho sakta hai.
"""


# ---------------------------------------------------------------------------
# QUICK TOPIC LIBRARY — organized by category so users can tap instead of
# type, and it doesn't clutter the screen. Covers common pain points for
# both rural (BC agents, Jan Dhan, Kisan Credit Card) and urban (fraud,
# NRI, locker, digital banking) customers.
# ---------------------------------------------------------------------------
TOPIC_CATEGORIES = {
    "🏦 Account Basics": [
        {"icon": "📝", "title": "Form kaise bharein", "question": "Bank ka form kaise bharte hain, step by step batayein."},
        {"icon": "🪪", "title": "KYC documents", "question": "KYC ke liye kaun-kaun se documents chahiye?"},
        {"icon": "🏦", "title": "Account kholna", "question": "Bank account kaise khulega, poori process batayein."},
        {"icon": "📔", "title": "Passbook update", "question": "Passbook update kahan aur kaise hoti hai?"},
        {"icon": "📄", "title": "Cheque bounce", "question": "Cheque bounce ho jaye to kya karna chahiye?"},
    ],
    "💻 Digital & Payments": [
        {"icon": "🔐", "title": "ATM PIN", "question": "ATM PIN kaise banega ya reset hoga?"},
        {"icon": "💻", "title": "Online banking", "question": "Online banking kya hoti hai aur kaise shuru karein?"},
        {"icon": "📲", "title": "UPI use karna", "question": "UPI kaise use karein, step by step batayein."},
        {"icon": "🛡️", "title": "Fraud se bachna", "question": "Online banking fraud se kaise bachein?"},
        {"icon": "📞", "title": "Bina smartphone balance", "question": "Bina smartphone ke bank balance kaise check karein?"},
    ],
    "💰 Loans & Credit": [
        {"icon": "💰", "title": "Loan eligibility", "question": "Loan ke liye eligibility kya hoti hai?"},
        {"icon": "📊", "title": "Credit score", "question": "Credit score kya hota hai aur kaise improve karein?"},
        {"icon": "🌾", "title": "Kisan Credit Card", "question": "Kisan Credit Card kya hai aur kaise banwayein?"},
        {"icon": "🏠", "title": "Loan against FD", "question": "FD par loan kaise milta hai?"},
    ],
    "🌾 Rural Banking": [
        {"icon": "🧑‍🌾", "title": "BC agent/Bank Mitra", "question": "Business Correspondent (Bank Mitra) se banking kaise karein?"},
        {"icon": "🪙", "title": "Jan Dhan account", "question": "Jan Dhan Yojana account ke fayde kya hain?"},
        {"icon": "🔗", "title": "Aadhaar seeding", "question": "Sarkari subsidy ke liye Aadhaar seeding kaise karayein?"},
        {"icon": "🌦️", "title": "Fasal Bima Yojana", "question": "PM Fasal Bima Yojana kya hai aur kaise apply karein?"},
    ],
    "🏛️ Pension & Schemes": [
        {"icon": "🏛️", "title": "Sarkari yojana", "question": "Pension ya government schemes ke baare mein bataiye."},
        {"icon": "👴", "title": "Doorstep banking", "question": "Senior citizens ke liye doorstep banking kaise book karein?"},
    ],
    "🛠️ Problems & Complaints": [
        {"icon": "🏧", "title": "ATM paisa nahi nikla", "question": "ATM se paisa nahi nikla par balance kat gaya, ab kya karein?"},
        {"icon": "📢", "title": "Shikayat darj karein", "question": "Branch mein problem solve nahi ho rahi, shikayat kaise darj karein?"},
        {"icon": "🔒", "title": "Locker facility", "question": "Bank locker kaise milta hai?"},
        {"icon": "🗣️", "title": "Terms simple mein", "question": "Banking ke mushkil terms simple bhasha mein samjhaiye."},
    ],
}

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div class='sidebar-card'>"
        "<h2>Banking Mitra</h2>"
        "<p>Your intelligent banking assistant answers account, loan, ATM, and KYC "
        "questions using PDF-backed knowledge and optional media attachments.</p>"
        "<div class='sidebar-badge'><span>ENCRYPTED</span><span>DOC-GROUNDED</span><span>HI/EN</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.last_model = "Waiting for first answer"
        st.session_state.retrieved_docs = []

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        "- Type your banking question in the chat box.\n"
        "- The bot uses your PDF documents for context.\n"
        "- Enable camera, microphone, or uploads only when needed.\n"
        "- Use **Reset Conversation** to start fresh."
    )

    st.markdown("---")
    st.metric("Last model", st.session_state.last_model)

    st.markdown("---")
    if kb is None:
        st.warning("No document index found. Run `python ingest.py` after adding PDFs to `data/`.")
    else:
        st.success("Knowledge base loaded. PDF search is active.")

    st.markdown("---")
    st.markdown("### Bank PDFs")
    if data_files:
        for file in data_files:
            st.markdown(f"<div class='doc-chip'><span class='dot'>●</span> {file}</div>", unsafe_allow_html=True)
    else:
        st.info("No bank PDFs found in /data.")

    st.markdown("---")
    st.markdown("### Find a branch (Uttar Pradesh — SBI)")
    st.caption(
        "Ek branch mein bheed ya problem ho to map dekh kar sabse nazdeeki dusri "
        "branch try karein — Lucknow, Kanpur, Varanasi, Agra, Prayagraj, Ghaziabad, "
        "Noida, Meerut, Gorakhpur, Bareilly aur Aligarh ki branches yahan hain."
    )
    branches_path = os.path.join(os.getcwd(), "branches.csv")
    if os.path.isfile(branches_path):
        branches_df = pd.read_csv(branches_path)
        city_options = ["All cities"] + sorted(branches_df["city"].unique().tolist())
        selected_city = st.selectbox("Filter by city", city_options, key="branch_city_filter")
        filtered_df = branches_df if selected_city == "All cities" else branches_df[branches_df["city"] == selected_city]
        st.map(filtered_df, latitude="lat", longitude="lon", size=60)
        st.caption(f"Showing {len(filtered_df)} of {len(branches_df)} branches.")
        for _, row in filtered_df.iterrows():
            st.markdown(f"<div class='doc-chip'><span class='dot'>◆</span> {row['name']} — {row['city']}</div>", unsafe_allow_html=True)
    else:
        st.info("Add a `branches.csv` (name, lat, lon, city) to show a branch locator map.")

    st.markdown("---")
    st.markdown(
        "<div class='sidebar-card'>"
        "<h2 style='font-size:1rem;'>Helpline</h2>"
        "<p>SBI Toll-free: 1800 1234 / 1800 2100<br>"
        "Cyber fraud: 1930 (National Cyber Crime Helpline)<br>"
        "RBI grievance: cms.rbi.org.in</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='hero-panel'>"
    "<div class='hero-seal' title='Trusted assistant'>🏦</div>"
    "<div class='hero-eyebrow'>Digital Passbook · AI Assistant</div>"
    "<div class='hero-title'>Banking Mitra</div>"
    "<div class='hero-desc'>A financial-advisor-style assistant for everyday banking — forms, KYC, "
    "account opening, passbook, ATM PIN, UPI, loans, and government schemes — explained in simple "
    "Hindi/English, so you don't need to wait in line to understand it.</div>"
    "<div class='feature-grid'>"
    "<div class='feature-card'><div class='feature-title'>Document-grounded</div>"
    "<div class='feature-text'>Answers are drawn from your indexed bank PDFs, with sources shown.</div></div>"
    "<div class='feature-card'><div class='feature-title'>Bilingual by default</div>"
    "<div class='feature-text'>Responds naturally in Hindi, English, or Hinglish.</div></div>"
    "<div class='feature-card'><div class='feature-title'>Privacy-first media</div>"
    "<div class='feature-text'>Camera and mic stay hidden and auto-disable after 30 seconds.</div></div>"
    "</div>"
    "<div class='ledger-strip'>"
    f"<div class='ledger-row'><span class='label'>Chat messages</span><span class='value'>{len(st.session_state.messages)}</span></div>"
    f"<div class='ledger-row'><span class='label'>Source documents indexed</span><span class='value'>{len(data_files)}</span></div>"
    f"<div class='ledger-row'><span class='label'>Retrieval status</span><span class='value'>{'ACTIVE' if kb else 'OFFLINE'}</span></div>"
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)

main_col, side_col = st.columns([3, 1], gap="large")

# ---------------------------------------------------------------------------
# MAIN CHAT COLUMN
# ---------------------------------------------------------------------------
with main_col:

    def render_bubble(role: str, content: str):
        avatar = "🧑" if role == "user" else "🏦"
        bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-assistant"
        # WhatsApp/Facebook-style layout: user message on the right, assistant
        # reply on the left. Auto-scroll is handled separately by
        # scroll_to_bottom() so this stays plain custom HTML.
        if role == "user":
            html = (
                f"<div class='msg-row user'>"
                f"<div class='{bubble_class}'>{content}</div>"
                f"<div class='avatar user'>{avatar}</div></div>"
            )
        else:
            html = (
                f"<div class='msg-row assistant'>"
                f"<div class='avatar assistant'>{avatar}</div>"
                f"<div class='{bubble_class}'>{content}</div></div>"
            )
        st.markdown(html, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # TEXT-TO-SPEECH — anyone can listen to an answer instead of reading it.
    # Splits the answer into sentences/lines and reads them one after another
    # (natural pauses, and avoids browsers cutting off very long single
    # utterances). Auto-detects Hindi (Devanagari) vs English pronunciation.
    # Uses real st.button widgets (not raw HTML) so layout stays stable and
    # the stop control is always clickable.
    # -----------------------------------------------------------------------
    _TTS_SPLIT_JS = (
        "var hasDev=/[\\u0900-\\u097F]/.test(t);"
        "var lines=t.split(/(?<=[.?!\\u0964])\\s+/).filter(function(l){return l.trim().length>0;});"
        "lines.forEach(function(line){"
        "var u=new SpeechSynthesisUtterance(line);"
        "u.lang=hasDev?'hi-IN':'en-IN';"
        "u.rate=0.92;"
        "window.speechSynthesis.speak(u);"
        "});"
    )

    def speak_now(text: str):
        safe_json = json.dumps(text)
        script = (
            "<script>(function(){"
            f"var t={safe_json};"
            "window.speechSynthesis.cancel();" + _TTS_SPLIT_JS +
            "})();</script>"
        )
        st.components.v1.html(script, height=0)

    def stop_speaking():
        st.components.v1.html("<script>window.speechSynthesis.cancel();</script>", height=0)

    def render_listen_controls(content: str, msg_key: str):
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("🔊 Sunein", key=f"speak_{msg_key}"):
                speak_now(content)
        with c2:
            if st.button("⏹ Roken", key=f"stop_{msg_key}"):
                stop_speaking()

    def scroll_to_bottom():
        # Anchor lives in the real page DOM (via st.markdown), the script below
        # (rendered in an iframe by components.html) reaches into the parent
        # page to scroll that anchor into view after every new message.
        st.markdown("<div id='chat-bottom-anchor'></div>", unsafe_allow_html=True)
        st.components.v1.html(
            """
            <script>
              const anchor = window.parent.document.getElementById('chat-bottom-anchor');
              if (anchor) { anchor.scrollIntoView({behavior: 'smooth', block: 'end'}); }
            </script>
            """,
            height=0,
        )

    st.markdown("<h3 style='margin-bottom:6px;'>Common questions</h3>", unsafe_allow_html=True)
    st.caption("Tap a topic for an instant answer — no need to type or wait in a branch line.")

    topic_tabs = st.tabs(list(TOPIC_CATEGORIES.keys()))
    for tab, (category, topics) in zip(topic_tabs, TOPIC_CATEGORIES.items()):
        with tab:
            cols = st.columns(len(topics))
            for col, topic in zip(cols, topics):
                with col:
                    if st.button(f"{topic['icon']} {topic['title']}", key=f"topic_{category}_{topic['title']}", use_container_width=True):
                        st.session_state["queued_topic_prompt"] = topic["question"]

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    if st.session_state.messages:
        for idx, msg in enumerate(st.session_state.messages):
            render_bubble(msg["role"], msg["content"])
            if msg["role"] == "assistant":
                render_listen_controls(msg["content"], msg_key=f"hist_{idx}")
    else:
        st.info(
            "Ask about loans, KYC, ATM rules, account opening, or document requirements. "
            "You can also attach a voice note or photo, or tap a topic above for an instant answer."
        )

    scroll_to_bottom()

    with st.container():
        st.markdown("<div class='media-box'><h3>Media & Devices</h3>", unsafe_allow_html=True)
        st.caption("Enable only the feature you want to use — camera and microphone stay hidden by default.")
        st.info("Camera and microphone access automatically turn off after 30 seconds of inactivity.")

        auto_read = st.checkbox(
            "🔊 Auto-read answers aloud",
            key="auto_read",
        )

        enable_col1, enable_col2, enable_col3 = st.columns(3)
        with enable_col1:
            enable_mic = st.checkbox("Enable Microphone (record)", key="enable_mic")
        with enable_col2:
            enable_cam = st.checkbox("Enable Camera (capture)", key="enable_cam")
        with enable_col3:
            enable_upload = st.checkbox("Enable File Uploads", key="enable_upload")

        if enable_cam:
            if st.session_state.camera_enabled_at == 0:
                st.session_state.camera_enabled_at = time.time()
            elapsed = time.time() - st.session_state.camera_enabled_at
            if elapsed > 30:
                st.session_state.enable_cam = False
                st.session_state.camera_enabled_at = 0
                st.warning("Camera access disabled after 30 seconds. Re-enable to capture a photo.")
            else:
                st.info(f"Camera enabled for {int(30 - elapsed)} more seconds.")
                captured_image = st.camera_input("Take a photo", key="camera_input")
                if captured_image is not None:
                    st.session_state.camera_bytes = (
                        captured_image.getvalue() if hasattr(captured_image, "getvalue") else captured_image
                    )
                    st.success("Photo captured")
        else:
            st.session_state.camera_enabled_at = 0

        if enable_mic:
            if st.session_state.microphone_enabled_at == 0:
                st.session_state.microphone_enabled_at = time.time()
            mic_elapsed = time.time() - st.session_state.microphone_enabled_at
            if mic_elapsed > 30:
                st.session_state.enable_mic = False
                st.session_state.microphone_enabled_at = 0
                st.warning("Microphone access disabled after 30 seconds. Re-enable to record audio.")
            else:
                st.info(f"Microphone enabled for {int(30 - mic_elapsed)} more seconds.")
                recorded_audio = st.audio_input("Record a voice note", key="voice_input")
                if recorded_audio is not None:
                    st.session_state.audio_bytes = (
                        recorded_audio.getvalue() if hasattr(recorded_audio, "getvalue") else recorded_audio
                    )
                    st.session_state.audio_mime_type = "audio/wav"
                    st.success("Voice note captured")
        else:
            st.session_state.microphone_enabled_at = 0

        if enable_upload:
            st.write("**Upload options**")
            uploaded_audio = st.file_uploader(
                "Upload a voice note (optional)", type=["wav", "mp3", "m4a", "ogg"], key="audio_upload"
            )
            if uploaded_audio is not None:
                st.session_state.audio_bytes = uploaded_audio.getvalue()
                st.session_state.audio_mime_type = {
                    "wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4", "ogg": "audio/ogg",
                }.get(uploaded_audio.name.split(".")[-1].lower(), "audio/wav")
                st.success("Audio file uploaded")

            uploaded_image = st.file_uploader("Or upload a photo", type=["jpg", "jpeg", "png"], key="image_upload")
            if uploaded_image is not None:
                st.session_state.camera_bytes = uploaded_image.getvalue()
                st.success("Image uploaded")

        if not enable_cam and not enable_mic and not enable_upload:
            st.info("Activate a device or upload option to begin.")
        st.markdown("</div>", unsafe_allow_html=True)

    typed_prompt = st.chat_input("Ask your banking question here...")
    send_message = st.button("Send message", type="primary")
    queued_topic_prompt = st.session_state.pop("queued_topic_prompt", None)
    prompt = typed_prompt or queued_topic_prompt

    pending_audio_bytes = st.session_state.audio_bytes
    pending_camera_bytes = st.session_state.camera_bytes
    pending_audio_mime_type = st.session_state.audio_mime_type

    if pending_audio_bytes is not None:
        st.warning("Voice note ready. Click Send message to submit it with your question.")

    if prompt or send_message or pending_audio_bytes or pending_camera_bytes:
        user_text = (prompt or "Shared voice or photo input").strip()
        st.session_state.messages.append({"role": "user", "content": user_text})
        render_bubble("user", user_text)
        if pending_audio_bytes is not None:
            st.audio(pending_audio_bytes, format=pending_audio_mime_type)
        if pending_camera_bytes is not None:
            st.image(pending_camera_bytes)
        scroll_to_bottom()

        if kb is not None:
            docs = kb.similarity_search(user_text, k=3)
            context = "\n\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
            source_docs = [doc.metadata.get("source", "Unknown") for doc in docs]
        else:
            context = ""
            source_docs = []

        response = None
        models_to_try = [
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
            "gemini-pro-latest",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ]
        error_message = None

        def should_continue_for_error(exc: Exception) -> bool:
            if isinstance(exc, ServerError):
                return True
            if isinstance(exc, ClientError):
                return exc.code in {404, 429, 503}
            return False

        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            "<div class='msg-row assistant'><div class='avatar assistant'>🏦</div>"
            "<div class='chat-bubble-assistant'><div class='typing-dots'><span></span><span></span><span></span></div></div></div>",
            unsafe_allow_html=True,
        )

        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=(
                        build_multimodal_payload(
                            user_text,
                            image_bytes=pending_camera_bytes,
                            audio_bytes=pending_audio_bytes,
                            audio_mime_type=pending_audio_mime_type,
                        )
                        if (pending_audio_bytes is not None or pending_camera_bytes is not None)
                        else (
                            f"Bank document context (use this first if relevant): {context}\n\n"
                            f"General banking reference knowledge (use when the document context "
                            f"above doesn't cover the topic): {FINANCIAL_ADVISOR_KNOWLEDGE}\n\n"
                            f"Question: {user_text}\n\n"
                            "You are a friendly banking assistant acting like a helpful financial "
                            "advisor for someone who may not be familiar with banking or English "
                            "terms. Rules:\n"
                            "1. Prefer the bank document context; if it doesn't cover the question, "
                            "use the general reference knowledge instead of saying you don't know.\n"
                            "2. Answer in simple Hindi/Hinglish (or English if the user wrote in "
                            "English), avoiding jargon. If a banking term is unavoidable, explain it "
                            "in brackets the first time, e.g. 'NEFT (bank se bank paisa transfer)'.\n"
                            "3. For 'how to' questions, give clear numbered steps.\n"
                            "4. Keep the answer focused and practical — the goal is that the person "
                            "does not need to visit the branch or wait in line just to understand this.\n"
                            "5. If something depends on the specific bank's rules and isn't in the "
                            "context, say so honestly and suggest confirming at the branch or on the "
                            "bank's official app/website.\n"
                            "6. Be mindful that the user could be from a rural area (may not have "
                            "smartphone/internet access, may prefer BC agent/Bank Mitra or missed-call "
                            "banking) or an urban area (may prefer app/UPI-based solutions) — offer the "
                            "option that best fits what they're asking, and mention the alternative "
                            "briefly if relevant."
                        )
                    ),
                )
                st.session_state.last_model = model_name
                error_message = None
                break
            except Exception as e:
                error_message = f"{type(e).__name__}: {e}"
                if should_continue_for_error(e):
                    continue
                break

        typing_placeholder.empty()

        if response is None:
            st.error(error_message or "Unable to generate an answer. Please try again later.")
        else:
            answer = ""
            if getattr(response, "text", None):
                answer = response.text
            elif getattr(response, "parts", None):
                answer = "".join(str(getattr(part, "text", part)) for part in response.parts)
            else:
                answer = str(response)

            render_bubble("assistant", answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            render_listen_controls(answer, msg_key=f"live_{len(st.session_state.messages)}")
            st.session_state.retrieved_docs = source_docs
            st.session_state.audio_bytes = None
            st.session_state.camera_bytes = None
            st.session_state.audio_mime_type = "audio/wav"
            if st.session_state.get("auto_read"):
                speak_now(answer)
            scroll_to_bottom()

# ---------------------------------------------------------------------------
# SIDE COLUMN — Chat Insights
# ---------------------------------------------------------------------------
with side_col:
    st.markdown("### Chat Insights")
    st.metric("Model active", st.session_state.last_model)

    if kb is not None:
        st.success("Document retrieval is enabled")
    else:
        st.warning("No retrieval index available")

    if st.session_state.retrieved_docs:
        st.markdown("**Recent sources:**")
        for source in st.session_state.retrieved_docs:
            st.write(f"• {source}")
    else:
        st.write("No source documents retrieved yet.")

    st.markdown("---")
    st.markdown("### Why this demo works")
    st.write(
        "• Clean visual layout for presentation.\n"
        "• Bilingual answers for Hindi and English.\n"
        "• PDF-backed knowledge retrieval.\n"
        "• Clear guidance for users and reviewers."
    )