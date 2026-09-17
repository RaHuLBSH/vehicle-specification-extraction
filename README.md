# Vehicle Specification Extraction

## Objective

This project answers service-manual specification questions with grounded, structured JSON. It ingests a PDF once, persists searchable chunks in FAISS, retrieves relevant evidence for each query, and asks Gemini to extract only from that evidence. Each answer is validated with Pydantic and includes source-page evidence.

## Architecture

```text
PDF → PyMuPDF → cleaning → section-aware chunks → MiniLM embeddings → FAISS + metadata
Query → MiniLM embedding → top-k chunks → Gemini → Pydantic validation → JSON
```

Ingestion and querying are intentionally separate: querying loads the persisted index and never reparses or re-embeds the PDF.

## Tech Stack

- Python 3.10+, PyMuPDF, sentence-transformers, FAISS, Google Gen AI SDK, Pydantic, and pytest.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment Variables

```dotenv
GEMINI_API_KEY=your_key_here
# Optional; defaults to gemini-3.6-flash
GEMINI_MODEL=gemini-3.6-flash
```

## Ingestion

Run once for a manual (or whenever it changes):

```bash
python ingest.py data/service_manual.pdf
```

This writes `storage/manual.faiss` and `storage/metadata.json`, then prints page, chunk, embedding, and output-path counts.

## Querying

```bash
python main.py
```

The process loads the FAISS index, metadata, and embedding model once. Enter queries until `quit`; each query embeds only the query text, retrieves top-k chunks, calls Gemini, and prints validated JSON.

### Streamlit UI

After installing dependencies and ingesting a manual, start the browser UI:

```bash
streamlit run app.py
```

The UI uses the same query pipeline as `main.py`, caches loaded resources between interactions, and shows concise startup, retrieval, or extraction errors rather than a traceback.

## Example Queries

```text
What is the wheel nut torque specification?
What is the minimum front brake disc thickness?
What is the front axle lubricant fill capacity?
What is the part number for the specified front axle lubricant?
```

## Example Output

```json
{
  "found": true,
  "specifications": [{
    "component": "wheel nuts",
    "spec_type": "torque",
    "value": 204.0,
    "unit": "Nm",
    "alt_value": 150.0,
    "alt_unit": "lb-ft",
    "page": 199,
    "source_text": "Tighten to 204 Nm (150 lb-ft)."
  }],
  "reason": ""
}
```

## Chunking Strategy

Text is cleaned conservatively so measurements, identifiers, grades, and units are retained. The chunker prefers headings and paragraph boundaries, targets roughly 650 word-like tokens, caps chunks at 800, and carries about 100 tokens of overlap. Each chunk retains its section and page range so retrieved evidence remains traceable.

## Embedding Strategy

`all-MiniLM-L6-v2` was selected because it is compact, fast on CPU, broadly effective for semantic retrieval, and produces a convenient 384-dimensional representation. Document and query vectors are L2-normalized, so inner product corresponds to cosine similarity.

## FAISS Design

FAISS `IndexFlatIP` provides exact inner-product nearest-neighbor search with minimal operational complexity for this manual-sized corpus. The FAISS file stores vectors; `metadata.json` stores the parallel chunk text, page, section, and range fields. Retaining source metadata lets Gemini cite evidence and lets evaluation measure whether the correct manual page was retrieved.

## Gemini Extraction

Gemini is used after retrieval rather than against the entire PDF. This reduces prompt size and cost, focuses the model on relevant evidence, and makes the instruction “use only retrieved context” enforceable. Gemini returns JSON constrained by a schema; Pydantic then validates required fields, found/not-found consistency, and value/unit formatting locally. Combined measurements such as `32 mm (1.259 in)` are normalized into `value=32`, `unit="mm"`, `alt_value=1.259`, and `alt_unit="in"`.

## Evaluation

`data/evaluation.json` contains 20 manually verified questions from the supplied service manual across torque, fluid capacity, part numbers, dimensions, pressure, and oil/fluid specifications. Every case includes query, expected component, value, unit, and supporting page.

Run retrieval-only evaluation without Gemini calls:

```bash
python -m src.evaluate
```

It reports Top-1, Top-3, and Top-5 page-retrieval accuracy. To also score Gemini extraction (component, value, unit, and exact specification match), explicitly opt in:

```bash
python -m src.evaluate --with-extraction
```

Run tests without external LLM calls:

```bash
pytest -m "not integration"
```

## Limitations

The current parser only extracts embedded PDF text; scanned pages need OCR. Chunking and section detection are heuristic, tables may lose their original layout, and exact FAISS search is intended for a modest corpus. Gemini output is grounded by prompt and validation but still requires evaluation and human review for safety-critical use.

## Production Considerations

Version the PDF, chunks, embedding model, index, and evaluation set together. Add API retries, timeouts, structured logs, access controls, telemetry, and regression gates. Keep retrieval evidence in audit records, monitor retrieval/extraction quality separately, and use a scalable vector database when index size or update volume exceeds a single FAISS deployment.

## Future Improvements

- Hybrid BM25 + vector retrieval and a learned reranker
- OCR for scanned manuals and table-aware extraction
- Better section detection and document-layout preservation
- Canonical unit normalization and conversion policy
- Confidence scoring with abstention thresholds
- Larger evaluation suites, per-category dashboards, and error analysis
- Scalable vector databases and asynchronous ingestion for production-scale deployments
