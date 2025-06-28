import sys
import os
from dotenv import load_dotenv
from groq import Groq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as sl
from rag_pipeline.retrieve import retrieve_quotes

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

def summarize_quotes_groq(quotes, query):
    prompt = f"""The user searched for: '{query}'.
Here are the top quotes:\n\n""" + "\n".join([f'"{q["quote"]}" - {q["author"]}' for q in quotes])

    prompt += "\n\nSummarize the overall theme of these quotes in 2-3 sentences."

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes a list of quotes."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from Groq API: {e}"

sl.set_page_config(page_title="Semantic Quote Finder", layout="centered")
sl.title("📚 Semantic Quote Finder")
sl.markdown("Search quotes using natural language queries like *'funny quotes about life'* or *'quotes by Einstein on success'*.")

query = sl.text_input("🔍 Enter your quote search:", placeholder="e.g. inspirational quotes about failure")
top_k = sl.slider("How many quotes to return?", min_value=1, max_value=10, value=3)

if sl.button("🔎 Search Quotes") and query:
    with sl.spinner("Searching..."):
        results = retrieve_quotes(query, top_k=top_k)
        sl.session_state.results = results
        sl.session_state.query = query

if "results" in sl.session_state:
    results = sl.session_state.results
    for i, res in enumerate(results, 1):
        sl.markdown(f"### #{i}: \"{res['quote']}\"")
        sl.markdown(f"**Author:** {res['author'] or 'Unknown'}")
        sl.markdown(f"**Tags:** `{res['tags']}`")
        sl.markdown("---")

    if sl.button("🧠 Summarize These Quotes (Groq LLaMA3)"):
        with sl.spinner("Summarizing via LLaMA..."):
            summary = summarize_quotes_groq(results, query)
            sl.success("📜 Summary:")
            sl.markdown(summary)

print("PYTHONPATH: ", sys.path)