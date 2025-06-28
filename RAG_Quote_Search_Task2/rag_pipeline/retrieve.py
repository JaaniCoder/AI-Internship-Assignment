from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "vector_db", "faiss_index.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "..", "vector_db", "quote_metadata.pkl")

index = faiss.read_index(os.path.abspath(INDEX_PATH))

with open(os.path.abspath(METADATA_PATH), "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_quotes(query, top_k=5):
    query_vector = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    result = []
    for id in indices[0]:
        quote_data = metadata[id]
        result.append({
            "quote": quote_data['quote'],
            "author": quote_data.get('author', ''),
            "tags": quote_data.get('tags', '')
        })

    return result

if __name__ == "__main__":
    query = "quotes about success by famous people"
    result = retrieve_quotes(query, top_k=3)

    for i, r in enumerate(result, 1):
        print(f"\n#{i}")
        print(f"Quote: {r['quote']}")
        print(f"Author: {r['author']}")
        print(f"Tags: {r['tags']}")