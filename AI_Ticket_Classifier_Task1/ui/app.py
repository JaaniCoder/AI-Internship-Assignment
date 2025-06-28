import gradio as gr
import pickle
import numpy as np
import re
from scipy.sparse import hstack

with open('../models/issue_type_model.pkl', 'rb') as f:
    issue_model = pickle.load(f)

with open('../models/urgency_level_model.pkl', 'rb') as f:
    urgency_model = pickle.load(f)

with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('../models/issue_encoder.pkl', 'rb') as f:
    issue_encoder = pickle.load(f)

with open('../models/urgency_encoder.pkl', 'rb') as f:
    urgency_encoder = pickle.load(f)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemma.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def extract_entities(text):
    text = text.lower()
    return {
        "products": re.findall(r"\b(?:laptop|phone|charger|tablet)\b", text),
        "dates": re.findall(r"\b(?:\d{1,2}/\d{1,2}/\d{2,4})\b", text),
        "complaint_keywords": [w for w in ['broken', 'late', 'error', 'refund'] if w in text]
    }

def detect_urgency(text):
    text = str(text).lower()
    keywords = ['urgent', 'immediately', 'asap', 'now', 'today', 'soon', 'as soon as possible', 'quickly']
    return int(any(word in text for word in keywords))


def predict_ticket(text):
    try:
        clean = preprocess_text(text)
        tfidf_feat = tfidf.transform([clean])
        extra_feat = np.array([[len(clean.split()), int(any(w in clean for w in ['broken', 'error', 'late', 'refund'])), detect_urgency(text), len(clean.split())]])
        final_feat = hstack([tfidf_feat, extra_feat])

        issue_pred_label = issue_model.predict(final_feat)[0]
        urgency_pred_label = urgency_model.predict(final_feat)[0]

        issue = issue_encoder.inverse_transform([issue_pred_label])[0]
        urgency = urgency_encoder.inverse_transform([urgency_pred_label])[0]
        entities = extract_entities(text)

        return issue, urgency, entities

    except Exception as e:
        return "Error", "Error", {"error": str(e)}

# Gradio Interface
with gr.Blocks(title="Support Ticket Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📬 Support Ticket Classifier")
    gr.Markdown("Enter a customer support ticket message to classify issue type, urgency level, and extract key entities.")

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(placeholder="Paste the support ticket text here...", label="Ticket Text", lines=6)
            submit_btn = gr.Button("Analyze Ticket", variant="primary")

        with gr.Column(scale=1):
            issue_out = gr.Textbox(label="Predicted Issue Type", interactive=False)
            urgency_out = gr.Textbox(label="Predicted Urgency Level", interactive=False)
            entities_out = gr.JSON(label="Extracted Entities")

    submit_btn.click(fn=predict_ticket, inputs=input_text, outputs=[issue_out, urgency_out, entities_out])

demo.launch()
