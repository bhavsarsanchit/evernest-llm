"""Small page for trying the listing agent. The API does the work."""

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

API = os.environ.get("APP_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Listing agent", layout="centered")
st.title("Listing agent")
st.caption("Questions go to the API. Model and sampling override agent.yaml for this call.")

try:
    spec = requests.get(f"{API}/spec", timeout=3).json()
except requests.RequestException:
    spec = {"model": "", "temperature": 0.1, "max_tokens": 700, "top_k": 6}
    st.error(f"API is not running at {API}. Start it with the command in the README.")

with st.sidebar:
    st.header("Call")
    agent_id = st.text_input("Agent", "A-01")
    listing_id = st.text_input("Listing", "L-1001")
    st.header("Model")
    model = st.text_input("Model name", spec.get("model") or "")
    temperature = st.slider("Temperature", 0.0, 1.5, float(spec.get("temperature", 0.1)), 0.05)
    max_tokens = st.number_input("Max tokens", 50, 4000, int(spec.get("max_tokens", 700)), 50)
    top_k = st.slider("Pages to retrieve", 1, 12, int(spec.get("top_k", 6)))
    if st.button("Reindex documents"):
        with st.spinner("Indexing PDFs into Chroma"):
            try:
                result = requests.post(f"{API}/index", timeout=120).json()
                st.success(f"Indexed {result.get('indexed_pages', 0)} pages")
            except requests.RequestException as exc:
                st.error(str(exc))

question = st.text_area("Question", "Wie groß ist die Wohnfläche?")
if st.button("Ask", type="primary"):
    with st.spinner("Asking"):
        try:
            response = requests.post(
                f"{API}/ask",
                json={
                    "agent_id": agent_id,
                    "listing_id": listing_id,
                    "question": question,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": int(max_tokens),
                    "top_k": int(top_k),
                },
                timeout=120,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            st.error(str(exc))
        else:
            st.write(body["answer"])
            access = body["access"]
            st.caption(
                f"{body['model'] or 'no model call'} · listing {access['can_read_listing']} · "
                f"documents {access['can_read_documents']} · {access['reason']}"
            )
            if body["sources"]:
                st.subheader("Sources")
                for source in body["sources"]:
                    st.write(f"{source['document_id']} page {source['page']} · {source['sub_category']}")
