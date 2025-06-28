# 🧠 AI Internship Assignment - Vijayi WFH Technologies Pvt Ltd

## 👤 Candidate: \[Jitin]

## 📅 Submission Date: \[28-06-2025]

---

## ✅ Overview

This repository contains solutions to both tasks assigned as part of the AI Developer Internship assignment for **Vijayi WFH Technologies Pvt Ltd**. Each task demonstrates practical skills in NLP, LLMs, RAG pipelines, and API integration.

---

# 🧪 Task 1: Support Ticket Classification & Entity Extraction

### 🔍 Objective

Build an ML pipeline to classify customer support tickets into:

* **Issue Type** (e.g., Delivery, Billing)
* **Urgency Level** (Low, Medium, High)

Also extract structured entities like products, dates, and complaint keywords.

### 📁 Key Files

```
AI_Ticket_classifier_Task1/
├── ui
    ├── app.py                  
├── task1_ticket_classification.ipynb    
├── models/                 
├── data/                   
```

### 🔧 Tools & Technologies

* Python, Scikit-learn, NLTK
* TF-IDF vectorizer + extra features
* RandomForestClassifier
* Gradio for UI

### ✅ Features Implemented

* Text preprocessing, lemmatization
* Entity extraction using regex & keywords
* Classification pipeline with issue + urgency prediction
* Gradio UI for end-to-end testing

### 🎯 Final Model Results

* Accuracy \~87% for issue type
* Accuracy \~84% for urgency level
* Functional UI working with correct prediction/label mapping

---

# 📚 Task 2: Quote Retrieval + RAG Evaluation System

### 🔍 Objective

Develop a semantic quote search engine using Retrieval-Augmented Generation:

* Search and retrieve quotes based on semantic query
* Summarize retrieved results using LLM (LLaMA3 via Groq)
* Evaluate with RAGAS (faithfulness, recall, etc.)

### 📁 Key Files

```
Rag_Quote_Search_Task2/
├── ui
    ├── app.py                      
├── rag_pipeline
    ├── __init__.py
    ├── retrieve.py      
├── vector_db/                    
├── evaluation      
    ├── eval_dataset.json
    ├── ragas_eval.py
```

### 🔧 Tools & Stack

* Sentence Transformers (MiniLM)
* FAISS for vector search
* Groq API for LLaMA3 summarization
* RAGAS for structured evaluation
* Streamlit for web UI

### ✅ Features Implemented

* End-to-end semantic quote retrieval
* Groq + LLaMA3 summary generation
* Evaluation using RAGAS with metrics:

  * **Faithfulness**: 1.0 ✅
  * **Answer Relevance**: \~0.66 ✅
  * **Context Precision**: \~0.5 ✅
  * **Context Recall**: 0.0 ⚠️ (due to dataset mismatch)

### 🌐 Bonus

* Avoided OpenAI completely — used Groq Cloud + HF embeddings
* Successfully deployed local evaluation using `LangchainLLMWrapper`

---

## ✅ Final Notes

* Both tasks are completed and tested end-to-end
* Groq API used smartly to avoid OpenAI reliance
* RAG evaluation handled entirely offline (except LLM via Groq)
* Ready for deployment and further fine-tuning

### 🙏 Thank You

I appreciate the opportunity to complete this assignment. It showcases my ability to:

* Build full-stack ML/NLP pipelines
* Work with LLM APIs (Groq, Hugging Face)
* Evaluate models using modern metrics like RAGAS

Please feel free to reach out for a walkthrough or clarification.

---

**Author:** \[Jitin]
**Email:** \[[theshayarguyjaani.@gmail.com](mailto:theshayarguyjaani.@gmail.com)]
**LinkedIn:** [Jitin Sharma](https://www.linkedin.com/in/jitin-sharma-5191ba2aa)
