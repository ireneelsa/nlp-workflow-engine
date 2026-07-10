import os
import json
from dotenv import load_dotenv
load_dotenv(override=True)
from loguru import logger

_vertex_initialized = False
_use_vertex = False


def _init_vertex() -> bool:
    global _vertex_initialized, _use_vertex
    if _vertex_initialized:
        return _use_vertex

    creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    project_id = os.getenv("VERTEX_PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION", "us-central1")

    if creds_json and project_id:
        try:
            import vertexai
            from google.oauth2 import service_account

            # Accept either a file path or inline JSON string
            if os.path.isfile(creds_json):
                with open(creds_json) as f:
                    creds_data = json.load(f)
            else:
                creds_data = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(
                creds_data,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            vertexai.init(project=project_id, location=location, credentials=credentials)
            _use_vertex = True
            logger.info("[vertex_client] Vertex AI initialized")
        except Exception as e:
            logger.warning(f"[vertex_client] Vertex AI init failed: {e} — falling back to google.generativeai")
            _use_vertex = False
    else:
        logger.info("[vertex_client] GOOGLE_CREDENTIALS_JSON not set — using google.generativeai fallback")
        _use_vertex = False

    _vertex_initialized = True
    return _use_vertex


class VertexWithFallback:
    """Wraps a Vertex GenerativeModel and falls back to google.generativeai on 404/403."""

    def __init__(self, vertex_model, model_name: str):
        self._vertex_model = vertex_model
        self._model_name = model_name

    def generate_content(self, prompt):
        global _use_vertex
        try:
            return self._vertex_model.generate_content(prompt)
        except Exception as e:
            err = str(e)
            if any(code in err for code in ("404", "403", "NOT_FOUND", "PERMISSION_DENIED")):
                logger.warning(f"[vertex_client] Vertex model call failed ({e}) — switching to google.generativeai")
                _use_vertex = False
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                return genai.GenerativeModel(self._model_name).generate_content(prompt)
            raise


def get_gemini_model(model_name: str):
    """Return a GenerativeModel via Vertex AI (with fallback), or google.generativeai directly."""
    if _init_vertex():
        from vertexai.generative_models import GenerativeModel
        clean_name = model_name.removeprefix("models/")
        return VertexWithFallback(GenerativeModel(clean_name), model_name)
    else:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return genai.GenerativeModel(model_name)


def get_embedding_model():
    """Return the raw Vertex TextEmbeddingModel (text-embedding-004, 768d), or None on fallback."""
    if _init_vertex():
        from vertexai.language_models import TextEmbeddingModel
        return TextEmbeddingModel.from_pretrained("text-embedding-004")
    return None


class _VertexEmbeddingFn:
    """ChromaDB-compatible embedding function backed by Vertex AI text-embedding-004 (768d)."""

    def __init__(self):
        self._model = None
        self.name = "vertex_embedding"

    def _get_model(self):
        if self._model is None:
            from vertexai.language_models import TextEmbeddingModel
            self._model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        return self._model

    def __call__(self, input: list) -> list:
        embeddings = self._get_model().get_embeddings(input)
        return [e.values for e in embeddings]


def get_chroma_embedding_fn():
    """Return a ChromaDB-compatible embedding function (Vertex 768d, or ONNX 384d fallback)."""
    if _init_vertex():
        return _VertexEmbeddingFn()
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    return ONNXMiniLM_L6_V2()
