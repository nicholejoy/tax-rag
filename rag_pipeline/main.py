import logging
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from .config import get_settings
from .api import app
from .vector_store import initialize_vector_store, refresh_vector_store

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def init_pipeline():
    settings = get_settings()

    docs_path = Path(settings.docs_path)
    index_path = Path(settings.faiss_index_path)

    index_path_arg = str(index_path) if index_path.parent.exists() else None

    try:
        store = initialize_vector_store(index_path=index_path_arg)

        if (store.index is None or store.index.ntotal == 0) and docs_path.exists():
            logger.info("Empty store with source documents available — refreshing from source")
            refresh_vector_store(docs_path, index_path_arg)
            logger.info("Pipeline initialized and refreshed from source")
        elif store.index is not None:
            logger.info(f"Pipeline initialized with existing index ({store.index.ntotal} vectors)")
        else:
            logger.warning(f"Documents file not found: {docs_path}. Pipeline will start empty.")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


def main():
    settings = get_settings()
    init_pipeline()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()