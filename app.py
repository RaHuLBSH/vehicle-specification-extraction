"""Small Streamlit interface for the persisted vehicle-specification RAG app.

Start with ``streamlit run app.py`` after ingesting a manual.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from main import load_pipeline
from src.pipeline import PipelineExtractionError, PipelineRetrievalError, RAGPipeline
from src.vector_store import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH, VectorStoreError


@st.cache_resource(show_spinner="Loading the indexed manual and embedding model…")
def get_pipeline(top_k: int) -> RAGPipeline:
    """Cache expensive query-time resources across Streamlit reruns."""
    return load_pipeline(
        top_k=top_k,
        index_path=Path(DEFAULT_INDEX_PATH),
        metadata_path=Path(DEFAULT_METADATA_PATH),
    )


def main() -> None:
    st.set_page_config(page_title="Vehicle Specification Extraction", page_icon="🚘")
    st.title("Vehicle Specification Extraction")
    st.caption("Answers are extracted only from retrieved service-manual passages.")

    top_k = int(st.sidebar.number_input("Retrieved chunks", min_value=1, max_value=10, value=5))
    try:
        pipeline = get_pipeline(top_k)
    except VectorStoreError as exc:
        st.error(f"Index unavailable: {exc}. Run `python ingest.py <manual.pdf>` first.")
        return
    except Exception as exc:
        st.error(f"Unable to start the query service: {exc}")
        return

    query = st.text_input(
        "Ask about a vehicle specification",
        placeholder="e.g. What is the wheel nut torque specification?",
    )
    if not st.button("Extract specification", type="primary"):
        return
    if not query.strip():
        st.warning("Enter a specification question first.")
        return

    try:
        with st.spinner("Retrieving manual evidence and extracting the specification…"):
            result = pipeline.run(query)
    except PipelineRetrievalError as exc:
        st.error(f"Retrieval failed: {exc}")
        return
    except PipelineExtractionError as exc:
        st.error(f"Extraction failed: {exc}")
        return
    except Exception as exc:
        # Keep unexpected errors concise; Streamlit only shows tracebacks for
        # unhandled exceptions, so users never receive an implementation trace.
        st.error(f"Unable to answer this query: {exc}")
        return

    st.json(result.to_dict())


if __name__ == "__main__":
    main()
