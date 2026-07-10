import os
from loguru import logger
from huggingface_hub import HfApi

_FOLDER = "papers"


def _creds() -> tuple[str | None, str | None]:
    return os.getenv("HF_TOKEN"), os.getenv("HF_DATASET_REPO")


def collection_name_for_filename(filename: str) -> str:
    from src.indexer import clean_collection_name
    return clean_collection_name(filename)


def upload_paper(file_path: str, filename: str) -> bool:
    token, repo = _creds()
    if not token or not repo:
        logger.warning("[paper_store] HF_TOKEN or HF_DATASET_REPO not set — skipping upload")
        return False
    try:
        HfApi(token=token).upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"{_FOLDER}/{filename}",
            repo_id=repo,
            repo_type="dataset",
        )
        logger.info(f"[paper_store] Uploaded '{filename}' to {repo}")
        return True
    except Exception as e:
        logger.error(f"[paper_store] Upload failed for '{filename}': {e}")
        return False


def delete_paper(filename: str) -> bool:
    token, repo = _creds()
    if not token or not repo:
        logger.warning("[paper_store] HF_TOKEN or HF_DATASET_REPO not set — skipping delete")
        return False
    try:
        HfApi(token=token).delete_file(
            path_in_repo=f"{_FOLDER}/{filename}",
            repo_id=repo,
            repo_type="dataset",
        )
        logger.info(f"[paper_store] Deleted '{filename}' from {repo}")
        return True
    except Exception as e:
        logger.error(f"[paper_store] Delete failed for '{filename}': {e}")
        return False


def list_papers() -> list[str]:
    token, repo = _creds()
    if not token or not repo:
        logger.warning("[paper_store] HF_TOKEN or HF_DATASET_REPO not set — cannot list papers")
        return []
    try:
        prefix = f"{_FOLDER}/"
        return [
            f[len(prefix):]
            for f in HfApi(token=token).list_repo_files(repo_id=repo, repo_type="dataset")
            if f.startswith(prefix) and f.lower().endswith(".pdf")
        ]
    except Exception as e:
        logger.error(f"[paper_store] list_papers failed: {e}")
        return []


def download_all_papers(local_dir: str = "./papers") -> list[str]:
    token, repo = _creds()
    if not token or not repo:
        logger.warning("[paper_store] HF_TOKEN or HF_DATASET_REPO not set — skipping download")
        return []

    abs_local_dir = os.path.abspath(local_dir)
    parent_dir = os.path.dirname(abs_local_dir)
    os.makedirs(abs_local_dir, exist_ok=True)

    filenames = list_papers()
    if not filenames:
        logger.info("[paper_store] No papers found in HF Dataset")
        return []

    from huggingface_hub import hf_hub_download

    downloaded = []
    for filename in filenames:
        local_path = os.path.join(abs_local_dir, filename)
        if os.path.exists(local_path):
            logger.info(f"[paper_store] '{filename}' already exists locally, skipping")
            continue
        try:
            hf_hub_download(
                repo_id=repo,
                filename=f"{_FOLDER}/{filename}",
                repo_type="dataset",
                token=token,
                local_dir=parent_dir,
            )
            logger.info(f"[paper_store] Downloaded '{filename}'")
            downloaded.append(local_path)
        except Exception as e:
            logger.error(f"[paper_store] Failed to download '{filename}': {e}")

    return downloaded
