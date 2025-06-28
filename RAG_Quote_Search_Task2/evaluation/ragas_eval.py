from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.evaluation import evaluate
from datasets import Dataset
import json

groq_llm = ChatGroq(model_name="llama3-70b-8192")
ragas_llm = LangchainLLMWrapper(groq_llm)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

from ragas import evaluate as ragas_evaluate
ragas_evaluate.llm = ragas_llm

with open("evaluation/eval_dataset.json", "r") as f:
    raw = json.load(f)

formatted = Dataset.from_list([
    {
        "question": item["query"],
        "answer": item["answer"],
        "contexts": item["ground_truths"],
        "ground_truths": item["ground_truths"],
        "reference": item["reference"]
    }
    for item in raw
])

result = evaluate(
    formatted,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)
print("📊 RAG Evaluation Results:")
print(result)