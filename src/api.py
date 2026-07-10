import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import json
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

from src.agents import RouterAgent
from src.models import AgentState
from src.workflow_store import save_trace, list_workflows, load_workflow

from src.indexer import DocumentIndexer
from src.paper_store import upload_paper, delete_paper, list_papers, collection_name_for_filename
from fastapi import UploadFile, File
import tempfile

app = FastAPI(
    title="NLP Workflow Engine",
    description="Give it a goal and text — it plans and runs NLP tools automatically.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

router = RouterAgent()
from src.registry import build_default_registry
registry = build_default_registry()


# ── Request / Response models ──────────────────────────────────────────

class RunRequest(BaseModel):
    goal: str = ""
    text: str = ""
    max_sentences: Optional[int] = 3

class StepResult(BaseModel):
    step_name: str
    tool_name: str
    status: str
    output: Optional[object] = None
    duration_ms: float = 0.0

class ReflectionOut(BaseModel):
    step_id: str
    quality: str
    reason: str
    should_continue: bool

class RunResponse(BaseModel):
    workflow_id: str
    workflow_name: str
    goal: str
    status: str
    steps: list[StepResult]
    reflections: list[ReflectionOut]
    total_duration_ms: float


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    with open("src/ui.html", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "message": "NLP Workflow Engine is running"}


@app.get("/tools")
def get_tools():
    from src.registry import build_default_registry
    registry = build_default_registry()
    return {"tools": registry.list_tools()}

#!!!!!!!!!!!!!!!!#

@app.post("/run", response_model=RunResponse)
def run_workflow(req: RunRequest):
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="goal cannot be empty")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    logger.info(f"[api] /run → goal: {req.goal[:60]}")

    try:
        agent_state = AgentState(
            memory={"input": {"text": req.text, "max_sentences": req.max_sentences}}
        )
        trace = router.route_and_run(agent_state, req.goal)
        save_trace(trace)

        steps_out = [
            StepResult(
                step_name=t["step_name"],
                tool_name=t["tool_name"],
                status=t["status"],
                output=t.get("output"),
                duration_ms=t.get("duration_ms", 0.0)
            )
            for t in trace.steps_trace
        ]

        reflections_out = [
            ReflectionOut(
                step_id=r.step_id,
                quality=r.quality,
                reason=r.reason,
                should_continue=r.should_continue
            )
            for r in trace.reflections
        ]

        return RunResponse(
            workflow_id=trace.workflow_id,
            workflow_name=trace.workflow_name,
            goal=req.goal,
            status=trace.final_status,
            steps=steps_out,
            reflections=reflections_out,
            total_duration_ms=trace.total_duration_ms
        )

    except Exception as e:
        logger.error(f"[api] /run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run/stream")
def run_workflow_stream(req: RunRequest):
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="goal cannot be empty")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    logger.info(f"[api] /run/stream → goal: {req.goal[:60]}")

    try:
        agent_state = AgentState(
            memory={"input": {"text": req.text, "max_sentences": req.max_sentences}}
        )

        def event_stream():
            for event in router.route_and_stream(agent_state, req.goal):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"[api] /run/stream failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows")
def get_workflows():
    ids = list_workflows()
    return {"count": len(ids), "workflow_ids": ids}


@app.get("/workflows/{workflow_id}")
def get_workflow(workflow_id: str):
    try:
        wf = load_workflow(workflow_id)
        return wf.model_dump()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")

# ── RAG Routes ─────────────────────────────────────────────────────────

class FileIndexResult(BaseModel):
    filename: str
    collection_name: str
    file_type: str
    chunks: int

class UploadResponse(BaseModel):
    results: list[FileIndexResult]
    total_files: int
    total_chunks: int

class QuestionRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None
    topic_filter: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    chunks_used: int
    collection: str
    chunk_types_used: list[str]
    sources: list[str]
    duration_ms: float



@app.post("/upload", response_model=UploadResponse)
async def upload_document(files: List[UploadFile] = File(...)):
    """Upload one or more .txt, .pdf, or image files and index them for RAG."""
    allowed_exts = [".txt", ".pdf", ".jpg", ".jpeg", ".png"]
    max_total_size = 50 * 1024 * 1024
    file_payloads = []
    total_size = 0

    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    try:
        for file in files:
            ext = os.path.splitext(file.filename or "")[1].lower()
            if ext not in allowed_exts:
                raise HTTPException(status_code=400, detail=f"Only {', '.join(allowed_exts)} files supported")

            content = await file.read()
            total_size += len(content)
            if total_size > max_total_size:
                raise HTTPException(status_code=400, detail="Total upload size must be 50MB or less")

            file_payloads.append((file.filename, ext, content))

        indexer = DocumentIndexer()
        upload_results = []

        for filename, ext, content in file_payloads:
            logger.info(f"[api] Uploading file: {filename}")
            tmp_path = None

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                base = os.path.basename(filename)
                result = indexer.index_file_to_research(tmp_path, base)
                upload_results.append(FileIndexResult(
                    filename=filename,
                    collection_name=collection_name_for_filename(base),
                    file_type=result["file_type"],
                    chunks=result["chunks"]
                ))
                if ext == ".pdf":
                    ok = upload_paper(tmp_path, filename)
                    if not ok:
                        logger.warning(f"[api] HF Dataset upload skipped or failed for '{filename}'")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.warning(f"Could not delete temp file: {e}")

        return UploadResponse(
            results=upload_results,
            total_files=len(upload_results),
            total_chunks=sum(result.chunks for result in upload_results)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[api] Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QuestionResponse)
def ask_question(req: QuestionRequest):
    """Ask a question about an indexed document."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    logger.info(f"[api] /ask → {req.question[:60]}")

    result = registry.run_tool("rag", {
        "question": req.question,
        "collection_name": req.collection_name or "research_papers",
        "topic_filter": req.topic_filter
    })

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    return QuestionResponse(
        answer=result.output["answer"],
        chunks_used=result.output["chunks_used"],
        collection=result.output["collection"],
        chunk_types_used=result.output.get("chunk_types_used", []),
        sources=result.output.get("sources", []),
        duration_ms=result.duration_ms
    )


@app.get("/collections")
def get_collections():
    """List unique source filenames indexed in the research_papers collection."""
    try:
        indexer = DocumentIndexer()
        collection = indexer.chroma.get_collection("research_papers")
        results = collection.get(include=["metadatas"])
        sources = list(set(
            m.get("source", "")
            for m in results["metadatas"]
            if m and m.get("source")
        ))
        return {"collections": sources, "count": len(sources)}
    except Exception:
        return {"collections": [], "count": 0}


@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str):
    """Delete all chunks for a paper from the research_papers collection."""
    try:
        indexer = DocumentIndexer()
        deleted_count = indexer.delete_paper_chunks(collection_name)
        logger.info(f"[api] Deleted {deleted_count} chunks for '{collection_name}'")
        try:
            papers = list_papers()
            for filename in papers:
                if collection_name_for_filename(filename) == collection_name:
                    ok = delete_paper(filename)
                    if not ok:
                        logger.warning(f"[api] HF Dataset delete failed for '{filename}'")
                    break
            else:
                logger.info(f"[api] No matching PDF in HF Dataset for collection '{collection_name}'")
        except Exception as e:
            logger.warning(f"[api] HF Dataset delete lookup failed: {e}")
        return {"deleted": collection_name, "success": True}
    except Exception as e:
        logger.error(f"[api] Delete collection failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))

# ── Run the server ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.api:app", host="0.0.0.0", port=port)
