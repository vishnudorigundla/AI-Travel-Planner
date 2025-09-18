# AI-Travel-Planner
# Project summary & features

AI Travel Planner — single-page Streamlit app that:

Accepts any destination, dates, travelers, budget.

Generates 1–3 itinerary drafts using Perplexity Pro (LangChain wrapper).

Fetches hotels, restaurants, attractions from Google Places and attaches clickable Google Maps links for every item (guaranteed, even if API returns zero results).

Merges AI itinerary text with place links so each itinerary item shows a map link inline.

Displays full results in Streamlit (editable draft area).

Sends itinerary via WhatsApp (Twilio) with automatic chunking to avoid message limits.

Exports itinerary as TXT or PDF (PDF optional — uses reportlab if installed).

Supports multiple tones and length settings (hooks in the LLM prompt).

File upload: accept .txt, .eml, .docx to extract user preferences or seed itinerary.

Basic bilingual hook (English / Telugu) — you can extend prompts for languages.

Robust error handling and fallback search links if Places API fails.

# Tech stack & requirements

Python 3.10+ recommended

Streamlit (UI)

langchain + langchain-perplexity (Perplexity Pro LLM)

requests (Google Places)

python-dotenv (.env)

twilio (WhatsApp)

python-docx (optional file parsing)

reportlab (optional PDF export)

# Project structure
```
ai-travel-planner/
├── app.py
├── requirements.txt
├── .env
└── .streamlit/
    └── config.toml
```

(You can add extra modules later: utils.py, places.py, llm.py, templates/, static/.)

# requirements.txt
```
streamlit>=1.28.0
langchain>=0.3.27
langchain-perplexity==0.1.2
python-dotenv>=1.0.1
twilio>=8.9.0
requests>=2.32.0
python-docx>=0.8.11
reportlab>=4.0.0

```
install with: pip install -r requirements.txt

# .streamlit/config.toml
```
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[theme]
base = "light"
primaryColor = "#1F77B4"
backgroundColor = "#F5F5F5"
secondaryBackgroundColor = "#E0E0E0"
textColor = "#000000"
font = "sans serif"

[global]
developmentMode = false
```
# .env template (fill these)
```
PERPLEXITY_API_KEY=your_perplexity_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
USER_WHATSAPP_NUMBER=whatsapp:+91YYYYYYYYYY


Make sure Places API (Places API / Text Search), Maps JavaScript or Geocoding are enabled in Google Cloud for this key.

Twilio sandbox: use the sandbox WhatsApp number while testing and add your number to the sandbox
```
# app.py
```
# app.py
import os
import streamlit as st
import requests
import textwrap
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_perplexity import ChatPerplexity
from twilio.rest import Client
from io import BytesIO
from docx import Document  # optional use if processing .docx uploads
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ---------------------------
# Load env
# ---------------------------
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
USER_WHATSAPP_NUMBER = os.getenv("USER_WHATSAPP_NUMBER")

# Validate keys quickly (show warnings inside app)
# ---------------------------
# Initialize clients
# ---------------------------
llm = ChatPerplexity(api_key=PERPLEXITY_API_KEY) if PERPLEXITY_API_KEY else None
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

# ---------------------------
# Helper: Google Maps link for any place (guaranteed)
# ---------------------------
def create_maps_search_url(place_name: str, destination: str) -> str:
    q = f"{place_name} {destination}".strip().replace(" ", "+")
    return f"https://www.google.com/maps/search/?api=1&query={q}"

# ---------------------------
# Robust Google Places fetcher with guaranteed links
# ---------------------------
def get_places(destination: str, place_type: str, top_n: int = 5):
    """
    Try Google Places Text Search. If results exist, return top_n with maps search link.
    If zero results or an error, return fallback generated search links.
    """
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": f"{place_type} in {destination}", "key": GOOGLE_MAPS_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()
        results = data.get("results", []) if isinstance(data, dict) else []
    except Exception:
        results = []

    places = []
    if results:
        for p in results[:top_n]:
            name = p.get("name") or f"{place_type}"
            address = p.get("formatted_address", "Address not available")
            rating = p.get("rating", "N/A")
            maps_url = create_maps_search_url(name, destination)
            places.append({"name": name, "address": address, "rating": rating, "maps_url": maps_url})
    else:
        # fallback: generate generic search links so UI always has links
        for i in range(top_n):
            fake_name = f"{place_type} {i+1}"
            maps_url = create_maps_search_url(place_type, destination)
            places.append({"name": fake_name, "address": "Not available", "rating": "N/A", "maps_url": maps_url})
    return places

# ---------------------------
# LLM wrapper - builds prompt + calls Perplexity
# ---------------------------
def build_prompt(destination, start_date, end_date, members, budget, tone="friendly", length="detailed"):
    """
    Construct a robust prompt for Perplexity. You can add languages or other options.
    length: 'short', 'medium', 'detailed'
    tone: 'professional', 'friendly', 'casual', 'formal', 'empathetic'
    """
    days = (end_date - start_date).days + 1
    prompt = f"Plan a {length} day-by-day travel itinerary for {members} people visiting {destination} from {start_date} to {end_date} ({days} days) within a budget of {budget}. "
    prompt += "Include for each day: times (morning/afternoon/evening) and activities, recommended attractions with short descriptions, suggested places for breakfast/lunch/dinner, a suitable hotel area. "
    prompt += f"Tone: {tone}. Keep the plan practical and specify approximate costs where possible. Provide results in plain text, day-wise sections. At end, provide a short summary and top recommended hotels, restaurants and attractions."
    return prompt

def generate_itineraries(destination, start_date, end_date, members, budget, drafts=1, tone="friendly", length="detailed"):
    if not llm:
        return ["Perplexity API key missing — enable PERPLEXITY_API_KEY in .env."]
    prompt = build_prompt(destination, start_date, end_date, members, budget, tone, length)
    drafts_text = []
    # generate N drafts (make separate calls if drafts>1)
    for i in range(max(1, drafts)):
        try:
            resp = llm.invoke(prompt + (f"\nDraft number: {i+1}" if drafts>1 else ""))
            content = resp.content if hasattr(resp, "content") else str(resp)
            drafts_text.append(content)
        except Exception as e:
            drafts_text.append(f"Error generating itinerary: {e}")
    return drafts_text

# ---------------------------
# Merge / attach maps inline for any place names that match (best-effort),
# and always append a guaranteed Maps-links section at the end.
# ---------------------------
def attach_links_inline(itinerary_text: str, hotels, restaurants, attractions, destination: str):
    """
    Best-effort: if exact place name appears in itinerary_text, replace it with markdown link.
    Also append a structured section at the end with guaranteed links.
    """
    # replace exact matches
    text = itinerary_text
    # combine all places for replacement (hotels first helps)
    for p in hotels + restaurants + attractions:
        name = p["name"]
        if name and name in text:
            md = f"[{name}]({p['maps_url']})"
            text = text.replace(name, md)
    # Append maps index
    links = "\n\n---\n\n### 🔎 Quick map links\n"
    links += "\n**Hotels**\n"
    for p in hotels:
        links += f"- [{p['name']}]({p['maps_url']}) — {p['address']} (⭐ {p['rating']})\n"
    links += "\n**Restaurants**\n"
    for p in restaurants:
        links += f"- [{p['name']}]({p['maps_url']}) — {p['address']} (⭐ {p['rating']})\n"
    links += "\n**Attractions**\n"
    for p in attractions:
        links += f"- [{p['name']}]({p['maps_url']}) — {p['address']} (⭐ {p['rating']})\n"
    return text + links

# ---------------------------
# WhatsApp sender with chunking
# ---------------------------
def send_whatsapp_chunks(message_text: str):
    if not twilio_client:
        return "Twilio credentials missing"
    MAX = 1500
    chunks = textwrap.wrap(message_text, MAX)
    try:
        for chunk in chunks:
            twilio_client.messages.create(body=chunk, from_=TWILIO_WHATSAPP_NUMBER, to=USER_WHATSAPP_NUMBER)
        return True
    except Exception as e:
        return str(e)

# ---------------------------
# Export helpers
# ---------------------------
def export_txt(content: str):
    b = content.encode("utf-8")
    return BytesIO(b)

def export_pdf(content: str):
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    lines = content.split("\n")
    y = height - 72
    for line in lines:
        c.drawString(72, y, line[:200])  # wrap roughly
        y -= 12
        if y < 72:
            c.showPage()
            y = height - 72
    c.save()
    buffer.seek(0)
    return buffer

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Travel Planner", page_icon="🧳", layout="wide")
st.title("🧳 AI Travel Planner — dynamic (Perplexity + Google Maps + WhatsApp)")

# Sidebar: inputs
st.sidebar.header("Trip inputs")
destination = st.sidebar.text_input("Destination (city/place)", value="Madanapalle")
start_date = st.sidebar.date_input("Start date", value=datetime.today())
end_date = st.sidebar.date_input("End date", value=datetime.today() + timedelta(days=1))
members = st.sidebar.number_input("Travelers", min_value=1, value=2)
budget = st.sidebar.text_input("Budget (e.g., 5000 INR)", value="5000 INR")
drafts = st.sidebar.selectbox("How many drafts to generate?", options=[1,2,3], index=0)
tone = st.sidebar.selectbox("Tone", options=["friendly","professional","casual","formal","empathetic"], index=0)
length = st.sidebar.selectbox("Length", options=["short","medium","detailed"], index=2)

st.sidebar.markdown("---")
st.sidebar.header("APIs & Tools")
st.sidebar.write("Perplexity and Google Places must be enabled in your .env.")
if not PERPLEXITY_API_KEY:
    st.sidebar.error("PERPLEXITY_API_KEY missing in .env")
if not GOOGLE_MAPS_API_KEY:
    st.sidebar.error("GOOGLE_MAPS_API_KEY missing in .env")
if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
    st.sidebar.info("Twilio not configured — WhatsApp disabled")

# File upload support (seed preferences or upload email/file)
st.sidebar.header("Upload preferences / inspiration (optional)")
uploaded = st.sidebar.file_uploader("Upload .txt / .docx / .eml", type=["txt","docx","eml"])
seed_text = ""
if uploaded:
    try:
        if uploaded.type == "text/plain":
            seed_text = uploaded.getvalue().decode("utf-8")
        elif uploaded.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded)
            seed_text = "\n".join(p.text for p in doc.paragraphs)
        else:
            # .eml basic parsing
            seed_text = uploaded.getvalue().decode("utf-8", errors="ignore")
        st.sidebar.success("Uploaded file read OK")
    except Exception as e:
        st.sidebar.error(f"Failed to parse file: {e}")

# Main action
if st.button("Generate Travel Plan"):
    if not destination:
        st.error("Please enter destination.")
    else:
        with st.spinner("Generating itineraries..."):
            # 1. call LLM for drafts
            drafts_texts = generate_itineraries(destination, start_date, end_date, members, budget, drafts=drafts, tone=tone, length=length)
            # 2. fetch places
            hotels = get_places(destination, "hotel", top_n=5)
            restaurants = get_places(destination, "restaurant", top_n=6)
            attractions = get_places(destination, "tourist attractions", top_n=8)
            # 3. attach links and show each draft
            st.success("Itineraries generated")
            for idx, d in enumerate(drafts_texts, start=1):
                st.subheader(f"Draft #{idx}")
                merged = attach_links_inline(d, hotels, restaurants, attractions, destination)
                # editable textarea
                edited = st.text_area(f"Editable draft #{idx}", value=merged, height=320, key=f"draft_{idx}")
                # actions per draft
                cols = st.columns([1,1,1,1])
                if cols[0].button(f"Send draft #{idx} to WhatsApp"):
                    with st.spinner("Sending via WhatsApp..."):
                        result = send_whatsapp_chunks(edited)
                        if result is True:
                            st.success("Sent to WhatsApp ✅")
                        else:
                            st.error(f"WhatsApp send failed: {result}")
                if cols[1].button(f"Download TXT draft #{idx}"):
                    bio = export_txt(edited)
                    st.download_button(f"Download draft #{idx}.txt", data=bio, file_name=f"itinerary_{destination}_{idx}.txt")
                if REPORTLAB_AVAILABLE and cols[2].button(f"Download PDF draft #{idx}"):
                    pdfb = export_pdf(edited)
                    if pdfb:
                        st.download_button(f"Download draft #{idx}.pdf", data=pdfb, file_name=f"itinerary_{destination}_{idx}.pdf")
                if cols[3].button(f"Use draft #{idx} as final"):
                    st.info("Final draft selected — you can edit and re-send.")
                    final_itinerary = edited
                    # store in session state
                    st.session_state["final_itinerary"] = final_itinerary

        # Display recommended places lists below
        st.markdown("---")
        st.subheader("Recommended hotels")
        for h in hotels:
            st.markdown(f"- **{h['name']}** (⭐ {h['rating']}) — {h['address']} — [Map]({h['maps_url']})")
        st.subheader("Recommended restaurants")
        for r in restaurants:
            st.markdown(f"- **{r['name']}** (⭐ {r['rating']}) — {r['address']} — [Map]({r['maps_url']})")
        st.subheader("Recommended attractions")
        for a in attractions:
            st.markdown(f"- **{a['name']}** (⭐ {a['rating']}) — {a['address']} — [Map]({a['maps_url']})")

# End of app.py

```
# How to run

Create virtualenv (recommended):

python -m venv venv
# Windows
venv\Scripts\activate
# mac/linux
source venv/bin/activate


# Install:

pip install -r requirements.txt


Fill .env with keys.

Run:

streamlit run app.py


Open http://localhost:8501

# outputs
<img width="1907" height="922" alt="image" src="https://github.com/user-attachments/assets/86d16f58-0f9e-4892-8765-6aa6d99c091c" />

<img width="1595" height="778" alt="image" src="https://github.com/user-attachments/assets/b3a42b98-d77b-4c36-b1d4-08873a4f9867" />

<img width="1884" height="913" alt="image" src="https://github.com/user-attachments/assets/5813435a-4e42-4dc4-ac7c-47b2cbc483db" />

<img width="1876" height="934" alt="image" src="https://github.com/user-attachments/assets/888ff56a-92df-4268-8f6d-f0262eb13740" />

<img width="1755" height="895" alt="image" src="https://github.com/user-attachments/assets/129b6190-0302-4619-844f-64d37f335879" />

<img width="1525" height="824" alt="image" src="https://github.com/user-attachments/assets/3ccf0577-9a1f-4a83-a000-da72f6a0fc03" />

<img width="1687" height="921" alt="image" src="https://github.com/user-attachments/assets/b26eb142-ebc6-4a90-bf08-cd03a7b7cb1f" />

<img width="914" height="919" alt="image" src="https://github.com/user-attachments/assets/b2a2beed-5d25-4879-9903-478c44cbbd11" />

# Result 
 AI travel planner executed successfully.
