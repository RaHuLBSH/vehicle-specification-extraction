# Vehicle Specification Extraction

## Objective

This project answers automotive service-manual specification questions
with grounded, structured JSON.

A service-manual PDF is ingested once and converted into section-aware
chunks. At query time, the system combines semantic retrieval with FAISS
and lexical retrieval with BM25, fuses both rankings using Reciprocal
Rank Fusion (RRF), and supplies the highest-ranked evidence to Gemini.
Gemini extracts specifications only from the retrieved context, and
Pydantic validates the structured output.

Each extracted specification includes source-page evidence for
traceability.

## Architecture

``` text
PDF
 ↓
PyMuPDF extraction
 ↓
Conservative text cleaning
 ↓
Section-aware chunking
 ↓
MiniLM embeddings
 ↓
FAISS index + chunk metadata

Query
 ├── MiniLM embedding → FAISS Top-20 ──┐
 └───────────────────→ BM25 Top-20 ────┤
                                        ↓
                               Reciprocal Rank Fusion
                                        ↓
                                     Top-5
                                        ↓
                                      Gemini
                                        ↓
                               Pydantic validation
                                        ↓
                                Structured JSON
```

Ingestion and querying are intentionally separate. Querying loads the
persisted FAISS index and metadata and never reparses or re-embeds the
PDF.

## Tech Stack

-   Python 3.10+
-   PyMuPDF
-   `sentence-transformers/all-MiniLM-L6-v2`
-   FAISS
-   BM25 (`rank-bm25`)
-   Google Gen AI SDK
-   Pydantic
-   pytest
-   Streamlit

## Setup

``` bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment Variables

``` dotenv
GEMINI_API_KEY=your_key_here

# Optional; use the model configured for the project.
GEMINI_MODEL=your_model_name
```

## Ingestion

Run ingestion once for a manual, or whenever the source document
changes:

``` bash
python ingest.py data/service_manual.pdf
```

This creates:

``` text
storage/manual.faiss
storage/metadata.json
```

The persisted metadata contains the chunk text and source information
required for retrieval and evidence tracing.

## Querying

Start the interactive query pipeline:

``` bash
python main.py
```

For each query, the system:

1.  Embeds the query with MiniLM.
2.  Retrieves semantic candidates from FAISS.
3.  Retrieves lexical candidates using BM25.
4.  Combines both ranked lists with Reciprocal Rank Fusion.
5.  Sends the highest-ranked evidence chunks to Gemini.
6.  Validates the extracted specification with Pydantic.
7.  Returns structured JSON with source evidence.

### Retrieval Debugging

The retrieval layer can also be queried independently from Gemini:

``` bash
python -m src.retriever "What is the wheel nut torque specification?" --top-k 5
```

This is useful for inspecting FAISS, BM25, RRF, pages, sections, and
retrieved text without making an LLM request.

### Streamlit UI

After installing dependencies and ingesting a manual, start the browser
UI:

``` bash
streamlit run app.py
```

The UI uses the same retrieval and extraction pipeline as the CLI and
caches loaded resources between interactions.

## Example Queries

``` text
What is the wheel nut torque specification?
What is the minimum front brake disc thickness?
What is the front axle lubricant fill capacity?
What is the part number for the specified front axle lubricant?
```

## Example Output

``` json
{
  "found": true,
  "specifications": [
    {
      "component": "wheel nuts",
      "spec_type": "torque",
      "value": 204.0,
      "unit": "Nm",
      "alt_value": 150.0,
      "alt_unit": "lb-ft",
      "page": 199,
      "source_text": "Tighten to 204 Nm (150 lb-ft)."
    }
  ],
  "reason": ""
}
```

## Chunking Strategy

Text is cleaned conservatively so measurements, identifiers, grades,
part numbers, and units are preserved.

The chunker:

-   prefers headings and paragraph boundaries,
-   targets roughly 650 word-like tokens,
-   caps chunks at approximately 800 tokens,
-   uses approximately 100 tokens of overlap, and
-   retains section and page metadata.

This keeps retrieved evidence traceable while preserving enough local
context for specification extraction.

## Embedding Strategy

`all-MiniLM-L6-v2` is used for dense semantic retrieval because it is
compact, fast on CPU, and produces 384-dimensional embeddings.

Document and query vectors are L2-normalized. FAISS `IndexFlatIP` is
therefore used as an exact inner-product search index, with inner
product corresponding to cosine similarity for normalized vectors.

## Hybrid Retrieval Strategy

The final retrieval pipeline combines semantic and lexical search.

### FAISS

FAISS retrieves chunks that are semantically similar to the query using
normalized MiniLM embeddings.

This is useful when the wording of the user's question differs from the
wording used in the service manual.

### BM25

BM25 provides lexical retrieval over the chunk corpus.

It complements dense retrieval for technical queries containing exact
automotive terminology such as:

-   component names,
-   lubricant grades,
-   measurements,
-   part identifiers, and
-   specification-related wording.

### Reciprocal Rank Fusion

FAISS similarity scores and BM25 scores are not directly comparable.

Reciprocal Rank Fusion combines the two ranked candidate lists using
their ranks rather than their raw scores:

``` text
RRF(d) = Σ 1 / (k + rank(d))
```

The fused candidates are sorted by RRF score and the highest-ranked
chunks are supplied to Gemini.

A learned CrossEncoder reranker was also evaluated during development.
It did not improve the measured hybrid Answer-in-Context @1, @3, or @5
results across the final evaluation sets, so it was removed from the
final retrieval pipeline.

Detailed retrieval experiments and ablation results are documented in
[`results.md`](results.md).

## Gemini Extraction

Gemini is used after retrieval rather than against the complete PDF.

This:

-   reduces prompt size,
-   reduces unnecessary LLM processing,
-   focuses extraction on relevant evidence, and
-   makes the instruction to use only retrieved context easier to
    enforce.

Gemini returns JSON constrained by the extraction schema. Pydantic then
validates required fields, found/not-found consistency, and value/unit
formatting locally.

Combined measurements such as:

``` text
32 mm (1.259 in)
```

can be represented as:

``` text
value=32
unit="mm"
alt_value=1.259
alt_unit="in"
```

## Evaluation

`data/evaluation.json` contains **58 answerable specification
questions** covering categories such as:

-   torque,
-   fluid and lubricant capacity,
-   part numbers,
-   dimensions,
-   clearances,
-   pressure, and
-   oil/fluid specifications.

The final evaluation set combines the frozen test cases used during
retrieval evaluation.

### Retrieval Metrics

Retrieval is evaluated independently from Gemini using:

-   Answer-in-Context Recall@1
-   Answer-in-Context Recall@3
-   Answer-in-Context Recall@5
-   supporting-page hit rate

**Answer-in-Context is treated as the primary retrieval metric.**

A specification may occur on multiple pages of a service manual, and
document/page metadata can differ from the manually selected supporting
page. Therefore, whether the retrieved evidence actually contains the
expected answer is more representative of retrieval usefulness than
exact page matching alone.

Run retrieval-only evaluation:

``` bash
python -m src.evaluate
```

Run retrieval plus Gemini extraction evaluation:

``` bash
python -m src.evaluate --with-extraction
```

Detailed baseline, hybrid-retrieval, and reranking experiments are
available in:

``` text
results.md
```

### Final Retrieval Result

Across the three evaluation datasets, the final **FAISS + BM25 + RRF**
architecture improved Answer-in-Context retrieval over the FAISS-only
baseline.

The hybrid system achieved approximately **85% weighted
Answer-in-Context Recall@5 across 58 answerable cases**, compared with
approximately **76% for FAISS-only retrieval**.

CrossEncoder reranking produced no measurable improvement to the
reported hybrid @1, @3, or @5 metrics and was therefore excluded from
the final architecture.

## Testing

Run tests that do not require external LLM calls:

``` bash
pytest -m "not integration"
```

## Limitations

-   The current parser extracts embedded PDF text; scanned documents
    require OCR.
-   Section detection and chunking are heuristic.
-   Complex tables may lose some of their original layout during PDF
    text extraction.
-   Exact FAISS search is appropriate for this manual-sized corpus but
    is not intended as a large distributed vector-search system.
-   Exact supporting-page evaluation can be affected when the same
    specification appears on multiple pages.
-   Gemini is grounded using retrieved context and schema validation,
    but extraction quality should still be evaluated before
    safety-critical use.

## Production Considerations

For a production deployment:

-   version the source PDF, chunks, embedding model, index, and
    evaluation set together,
-   add API retries and timeouts,
-   add structured logging and telemetry,
-   preserve retrieved evidence in audit records,
-   monitor retrieval and extraction quality separately,
-   add access controls where documents contain restricted information,
-   use regression evaluation before deploying retrieval changes, and
-   move to scalable search/storage infrastructure when corpus size or
    update frequency exceeds a single-process FAISS deployment.

## Future Improvements

-   OCR for scanned manuals
-   table-aware extraction
-   improved section and document-layout detection
-   canonical unit normalization and conversion
-   confidence scoring and abstention thresholds
-   per-category evaluation dashboards and automated error analysis
-   larger evaluation suites using additional manuals
-   scalable vector/search infrastructure for multi-document production
    deployments
