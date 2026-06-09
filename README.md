# 🧠 NeuralScholar
**Machine Learning Research Literature Intelligence**

[![Live Application](https://img.shields.io/badge/Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://neuralscholar-vapssaw6clm9s7jkzpwgjh.streamlit.app/)
[![API Docs](https://img.shields.io/badge/API_Docs-FastAPI_Swagger-009688?style=for-the-badge&logo=fastapi)](https://neuralscholar-api.onrender.com/docs)

NeuralScholar is an advanced, end-to-end Retrieval-Augmented Generation (RAG) engine designed specifically for navigating dense Machine Learning research literature. It bridges the gap between semantic understanding and exact keyword matching by leveraging a sophisticated hybrid-retrieval pipeline, local embeddings, cross-encoder reranking, and large language model generation.

## 🚀 Key Features

* **Hybrid Search Retrieval:** Combines the semantic depth of dense vector search (Pinecone & FAISS) with the exact-keyword precision of sparse retrieval (BM25Okapi).
* **Cross-Encoder Reranking:** Implements a two-stage retrieval pipeline, refining initial candidate documents using a `ms-marco-MiniLM-L-6-v2` cross-encoder for maximal contextual relevance.
* **Hypothetical Document Embeddings (HyDE):** Optionally generates hallucinated answers to map queries closer to target document embeddings in latent space.
* **Automated Data Ingestion:** Natively integrates with the ArXiv API to dynamically fetch, chunk, embed, and index research papers on the fly.
* **Streaming Generation:** Real-time token streaming to the frontend using OpenAI's GPT-4o.

---

## 🏗️ System Architecture & Workflow

### 1. Data Ingestion Pipeline
1. **Extraction:** Papers are fetched via the ArXiv API matching specific IDs or ML queries.
2. **Chunking:** Documents are parsed and split using LangChain's `RecursiveCharacterTextSplitter` (Size: 1000, Overlap: 150) to preserve paragraph contexts.
3. **Embedding:** Chunks are embedded locally using `all-MiniLM-L6-v2` (384 dimensions) for cloud-optimized memory usage.
4. **Indexing:** Vectors are concurrently upserted to a cloud **Pinecone** index, a local **FAISS** index, and a **BM25** sparse index.

### 2. Query & Retrieval Pipeline
1. **HyDE Generation (Optional):** The user query is expanded into a hypothetical ideal document.
2. **Hybrid Search:** The query/HyDE text is embedded and searched against Pinecone/FAISS (Cosine Similarity via Inner Product) and BM25 (TF-IDF probabilistic scoring).
3. **Reciprocal Rank Fusion (RRF):** Results from dense and sparse retrievers are mathematically fused to mitigate distinct retrieval biases.
4. **Reranking:** The top fused candidates are paired with the user query and scored by a local Cross-Encoder to guarantee semantic alignment.

### 3. Answer Generation
1. The highest-ranked context chunks are injected into a strict, anti-hallucination system prompt.
2. The payload is sent to the LLM (GPT-4o) which synthesizes a grounded answer and streams the response back to the Streamlit UI.

---

## 🛠️ Technology Stack

| Component | Technology | Cloud Host |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Streamlit Community Cloud |
| **Backend API** | FastAPI, Uvicorn | Render |
| **Vector Databases** | Pinecone (Cloud), FAISS (Local) | Pinecone |
| **Sparse Retrieval** | rank_bm25 (Okapi) | Render |
| **Embedding Model** | SentenceTransformers (`all-MiniLM-L6-v2`) | Render |
| **Reranker** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | Render |
| **LLM / Generation** | OpenAI API (GPT-4o) | OpenAI |

---

## ⚙️ Installation & Setup (Local Development)

### 1. Prerequisites
* Python 3.11+
* An active OpenAI API Key
* An active Pinecone API Key & Index (Dimension: 384, Metric: cosine)

### 2. Environment Setup
Clone the repository and spin up a virtual environment:

```bash
git clone [https://github.com/Sahilll10/NeuralScholar.git](https://github.com/Sahilll10/NeuralScholar.git)
cd NeuralScholar
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements-backend.txt
