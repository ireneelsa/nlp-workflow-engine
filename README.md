---
title: NLP Workflow Engine
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# NLP Workflow Creation & Execution Engine

An LLM-powered agent system that takes a plain English goal and automatically
plans, executes, and reflects on a custom NLP analysis workflow — with full
multimodal RAG support for documents, scanned PDFs, and images.

## What it does

You give it a goal and a piece of text:
- **Goal:** `"Extract all people mentioned and summarize this article"`
- **Text:** any news article, email, document, or report

It figures out which NLP tools to run, executes them in order, reflects on
each result, and returns structured output — all automatically.

Separately, you can upload documents or images and ask questions about them
using a contextual RAG pipeline with topic-aware retrieval.

## Architecture

**NLP Workflow**

```
User Goal + Text
      ↓
  Planner Agent        ← LLM decides which tools to run and in what order
      ↓
Execution Engine       ← runs each step, handles retries, tracks state
      ↓
Reflection Agent       ← evaluates each step, decides to continue or stop
      ↓
Structured Output      ← entities, intent, summary, trace, timing
```

**Multimodal RAG Pipeline**

```
Document / Image Upload
      ↓
File Type Detection    ← image / normal PDF / scanned PDF
      ↓
PaddleOCR + Vision     ← extracts text + describes visual content (Groq vision)
      ↓
Document Summary       ← Gemini generates a 2–4 line summary
      ↓
Topic Classification   ← Gemini tags the document's category
      ↓
Markdown Chunking      ← splits by structure, not word count
      ↓
Contextual Embedding   ← summary prepended to every chunk before embedding
      ↓
ChromaDB Vector Store

User Question
      ↓
Semantic Search (+ optional topic filter)
      ↓
Gemini answers using only retrieved chunks
```

## NLP Tools

| Tool | What it does |
|---|---|
| Summarizer | Condenses text into 1–3 sentences |
| Intent Classifier | Labels the type of content (news, complaint, question…) |
| NER | Extracts people, organizations, locations, dates — dynamically filterable by type |
| RAG | Answers questions about uploaded documents/images using contextual retrieval |

## Multimodal Document Pipeline

| Case | Input | Pipeline |
|---|---|---|
| Image | `.jpg`, `.png` | PaddleOCR (text) + Groq vision (visual description) |
| Normal PDF | `.pdf` with selectable text | PyMuPDF text extraction + embedded image handling via the image pipeline |
| Scanned PDF | `.pdf` with no selectable text | pdf2image → OpenCV preprocessing (deskew/denoise) → PaddleOCR |

All three feed into the same contextual embedding and markdown chunking pipeline before storage in ChromaDB.

## Tech Stack

| Layer | Technology |
|---|---|
| LLM — planning, NLP tools, vision | Groq — `llama-3.3-70b-versatile`, `llama-4-scout` (vision) |
| LLM — summarization, topic classification, RAG answering | Gemini — `gemini-2.5-flash-lite` |
| OCR | PaddleOCR |
| PDF processing | PyMuPDF, pdf2image |
| Image preprocessing | OpenCV, Pillow |
| Vector DB | ChromaDB (local, persistent) |
| Embeddings | ONNX MiniLM (384d) |
| Backend | FastAPI + Python 3.11 |
| Agent framework | Custom-built (no LangChain) |
| Validation | Pydantic v2 |
| Logging | Loguru |
| Deployment | Docker, Hugging Face Spaces |

All free tier — no credit card required for any service used.

## Setup

```bash
git clone https://github.com/yourusername/nlp-workflow-engine
cd nlp-workflow-engine

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

cp .env.example .env
# Add your GROQ_API_KEY and GEMINI_API_KEY to .env

uvicorn src.api:app
```

Then open `http://localhost:8000` in your browser.

## Run Tests

```bash
python tests/test_eval.py          # NLP quality benchmarks
python tests/test_rag_eval.py      # RAG accuracy evaluation
python tests/test_adversarial.py   # Edge case handling
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Browser UI |
| `/health` | GET | Health check |
| `/tools` | GET | List available NLP tools |
| `/run` | POST | Run an NLP workflow |
| `/workflows` | GET | List saved workflows |
| `/workflows/{id}` | GET | Get a specific workflow |
| `/upload` | POST | Upload and index a document or image for RAG |
| `/ask` | POST | Ask a question about an indexed document |
| `/collections` | GET | List all indexed document collections |
| `/collections/{name}` | DELETE | Remove an indexed collection |

## Project Structure

```
nlp-workflow-engine/
├── src/
│   ├── models.py          # Pydantic data models
│   ├── registry.py        # Tool registry
│   ├── planner.py         # LLM planner agent
│   ├── executor.py        # Workflow execution engine
│   ├── reflection.py      # Reflection agent
│   ├── indexer.py         # Multimodal document indexer
│   ├── workflow_store.py  # Save/load workflows
│   └── api.py             # FastAPI routes
├── src/tools/
│   ├── base.py            # BaseTool interface
│   ├── summarizer.py
│   ├── classifier.py
│   ├── ner.py
│   └── rag.py
├── tests/
│   ├── test_eval.py
│   ├── test_rag_eval.py
│   └── test_adversarial.py
├── workflows/             # Saved workflow JSON files
├── chroma_store/          # Vector DB (gitignored)
├── Dockerfile
├── .dockerignore
├── .env.example
├── requirements.txt
└── README.md
```

## Deployment

Deployed via Docker on Hugging Face Spaces. See [`Dockerfile`](Dockerfile) for build configuration.
Set `GROQ_API_KEY` and `GEMINI_API_KEY` as Space secrets in the HF dashboard — do not include them in the image.
