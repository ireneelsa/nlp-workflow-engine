# NLP Workflow Creation & Execution Engine

An LLM-powered agent system that takes a plain English goal and automatically
plans, executes, and reflects on a custom NLP analysis workflow.

## What it does

You give it a goal and a piece of text:
- Goal: `"Extract all people mentioned and summarize this article"`
- Text: any news article, email, document, or report

It figures out which NLP tools to run, executes them in order, reflects on
each result, and returns structured output — all automatically.

## Architecture

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

## NLP Tools

| Tool | What it does |
|---|---|
| Summarizer | Condenses text into 1-3 sentences |
| Intent Classifier | Labels the type of content (news, complaint, question…) |
| NER | Extracts people, organizations, locations, dates |

## Tech Stack

- **LLM**: Groq (free, no credit card) — llama-3.3-70b-versatile
- **Backend**: FastAPI + Python 3.11
- **Agent framework**: Custom-built (no LangChain)
- **Validation**: Pydantic v2
- **Logging**: Loguru

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/nlp-workflow-engine
cd nlp-workflow-engine

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key (free at console.groq.com — no card needed)
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Start the server
uvicorn src.api:app
```

Then open `http://localhost:8000` in your browser.

## Run tests

```bash
# Unit tests
python tests/test_smoke.py
python tests/test_tools.py

# Integration tests
python tests/test_phase3.py
python tests/test_phase4.py
python tests/test_phase5.py

# Eval harness (benchmarks NLP quality)
python tests/test_eval.py

# Adversarial tests (edge cases)
python tests/test_adversarial.py
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Browser UI |
| `/health` | GET | Health check |
| `/tools` | GET | List available NLP tools |
| `/run` | POST | Run a workflow |
| `/workflows` | GET | List saved workflows |
| `/workflows/{id}` | GET | Get a specific workflow |

### Example request

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Extract all people and summarize this article",
    "text": "Elon Musk announced Tesla will open a factory in Chennai, India..."
  }'
```

## Project Structure

```
nlp-workflow-engine/
├── src/
│   ├── models.py          # Pydantic data models
│   ├── registry.py        # Tool registry
│   ├── planner.py         # LLM planner agent
│   ├── executor.py        # Workflow execution engine
│   ├── reflection.py      # Reflection agent
│   ├── workflow_store.py  # Save/load workflows
│   └── api.py             # FastAPI routes
├── tools/
│   ├── base.py            # BaseTool interface
│   ├── summarizer.py
│   ├── classifier.py
│   └── ner.py
├── tests/
│   ├── test_smoke.py
│   ├── test_tools.py
│   ├── test_phase3.py
│   ├── test_phase4.py
│   ├── test_phase5.py
│   ├── test_eval.py
│   └── test_adversarial.py
├── workflows/             # Saved workflow JSON files
├── .env.example
├── requirements.txt
└── README.md
```