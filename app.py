import os
import streamlit as st
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_perplexity import ChatPerplexity
from twilio.rest import Client
import textwrap
from urllib.parse import quote_plus

# ==========================
# Load environment variables
# ==========================
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
USER_WHATSAPP_NUMBER = os.getenv("USER_WHATSAPP_NUMBER")

# ==========================
# Initialize Clients
# ==========================
llm = ChatPerplexity(api_key=PERPLEXITY_API_KEY, model="sonar-pro")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ==========================
# Helper Functions
# ==========================

def get_maps_url(name, address=None):
    # Prioritize address when available
    query = quote_plus(f"{name} {address}" if address else name)
    return f"https://www.google.com/maps/search/?api=1&query={query}"

def generate_itinerary(destination, start_date, end_date, members, budget, places_dict):
    """Use Perplexity to generate a detailed day-by-day travel plan WITH MAP LINKS."""
    prompt = f"""
    Create a detailed travel itinerary for {members} people visiting {destination}
    from {start_date} to {end_date} with a budget of {budget}.
    Include:
    - Day-wise breakdown with times
    - Top attractions with short descriptions
    - Suggested restaurants nearby (breakfast, lunch, dinner)
    - Hotel recommendations
    - For every hotel/restaurant/attraction, ADD its Google Maps link after name (use this format: [Google Maps](url)), using info from: {places_dict}
    """
    response = llm.invoke(prompt)
    return response.content

def get_places(destination, place_type, top_n=5):
    """Fetch hotels/restaurants/attractions with guaranteed Maps links."""
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{place_type} in {destination}",
        "key": GOOGLE_MAPS_API_KEY
    }
    r = requests.get(url, params=params)
    results = r.json().get("results", [])

    places = []

    if results:
        for place in results[:top_n]:
            name = place.get("name")
            address = place.get("formatted_address", "No address available")
            rating = place.get("rating", "N/A")
            maps_url = get_maps_url(name, address)
            places.append({
                "name": name,
                "address": address,
                "rating": rating,
                "maps_url": maps_url
            })
    else:
        # Fallback if no results returned
        for i in range(top_n):
            name = f"{place_type} {i+1}"
            address = "Not available"
            rating = "N/A"
            maps_url = get_maps_url(name, destination)
            places.append({
                "name": name,
                "address": address,
                "rating": rating,
                "maps_url": maps_url
            })

    return places

def send_whatsapp(message):
    """Send WhatsApp message via Twilio; split long messages if needed."""
    try:
        max_len = 1500  # Twilio WhatsApp limit
        for chunk in textwrap.wrap(message, max_len):
            twilio_client.messages.create(
                body=chunk,
                from_=TWILIO_WHATSAPP_NUMBER,
                to=USER_WHATSAPP_NUMBER
            )
        return True
    except Exception as e:
        return str(e)

def format_places(title, places):
    """Format hotel/restaurant/attraction list for Streamlit and WhatsApp."""
    formatted = f"### {title}\n"
    for p in places:
        formatted += f"- **{p['name']}** (⭐ {p['rating']}) [Google Maps]({p['maps_url']})\n  - {p['address']}\n"
    return formatted

def places_dict(hotels, restaurants, attractions):
    # Build a dict for the itinerary LLM prompt, so it can use correct links
    out = {
        "hotels": {p['name']: p['maps_url'] for p in hotels},
        "restaurants": {p['name']: p['maps_url'] for p in restaurants},
        "attractions": {p['name']: p['maps_url'] for p in attractions}
    }
    return out

# ==========================
# Streamlit App
# ==========================
st.set_page_config(page_title="AI Travel Planner", page_icon="🧳", layout="wide")
st.title("🧳 AI Travel Planner with Perplexity Pro + WhatsApp")

# Sidebar Inputs
st.sidebar.header("✈️ Trip Details")
destination = st.sidebar.text_input("Destination")
start_date = st.sidebar.date_input("Start Date", min_value=datetime.today())
end_date = st.sidebar.date_input("End Date", min_value=datetime.today() + timedelta(days=1))
members = st.sidebar.number_input("Number of Travelers", min_value=1, value=2)
budget = st.sidebar.text_input("Budget (e.g., 50,000 INR)")

if st.sidebar.button("Generate Travel Plan"):
    if destination:
        with st.spinner("🔮 Generating your AI itinerary..."):
            hotels = get_places(destination, "hotels")
            restaurants = get_places(destination, "restaurants")
            attractions = get_places(destination, "tourist attractions")
            places_links = places_dict(hotels, restaurants, attractions)
            itinerary = generate_itinerary(destination, start_date, end_date, members, budget, places_links)

        st.subheader("📅 AI-Generated Travel Plan")
        st.write(itinerary)

        # Display in Streamlit
        st.subheader("🏨 Recommended Hotels")
        for h in hotels:
            st.markdown(f"**{h['name']}** (⭐ {h['rating']}) [Google Maps]({h['maps_url']})\n📍 {h['address']}\n")

        st.subheader("🍽️ Recommended Restaurants")
        for r in restaurants:
            st.markdown(f"**{r['name']}** (⭐ {r['rating']}) [Google Maps]({r['maps_url']})\n📍 {r['address']}\n")

        st.subheader("🌍 Must-See Attractions")
        for a in attractions:
            st.markdown(f"**{a['name']}** (⭐ {a['rating']}) [Google Maps]({a['maps_url']})\n📍 {a['address']}\n")

        # Prepare WhatsApp message
        whatsapp_message = f"Your AI Travel Plan for {destination}:\n\n{itinerary}\n\n"
        whatsapp_message += format_places("Hotels", hotels) + "\n"
        whatsapp_message += format_places("Restaurants", restaurants) + "\n"
        whatsapp_message += format_places("Attractions", attractions)

        result = send_whatsapp(whatsapp_message)
        if result is True:
            st.success("📲 Full travel plan sent to your WhatsApp!")
        else:
            st.error(f"Failed to send WhatsApp message: {result}")
    else:
        st.error("Please enter a destination.")
