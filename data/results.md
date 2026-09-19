# Retrieval Evaluation Results

## Overview

This document summarizes the retrieval experiments performed for the
Vehicle Specification Extraction project.

The final retrieval architecture was selected through ablation testing
across three frozen evaluation datasets containing **58 answerable
questions** in total.

The main retrieval metric is **Answer-in-Context Recall@K**: whether at
least one of the top-K retrieved chunks contains the expected
specification. Supporting-page hit rate is also reported, but is treated
as a secondary metric because the same specification can appear on
multiple pages and PDF/page metadata may differ between sections.

## Systems Evaluated

Four retrieval configurations were evaluated:

1.  **FAISS** --- dense semantic retrieval using normalized
    `all-MiniLM-L6-v2` embeddings.
2.  **FAISS + CrossEncoder** --- FAISS candidates reranked with
    `cross-encoder/ms-marco-MiniLM-L6-v2`.
3.  **FAISS + BM25 + RRF** --- dense FAISS retrieval combined with
    lexical BM25 retrieval using Reciprocal Rank Fusion.
4.  **FAISS + BM25 + RRF + CrossEncoder** --- hybrid RRF candidates
    followed by CrossEncoder reranking.

Unless otherwise noted, retrieval returns the final Top-5 chunks.

## Answer-in-Context Results

### Dataset 1 --- Default Evaluation Set

20 answerable cases.

  Retrieval Method                           @1        @3        @5
  ----------------------------------- --------- --------- ---------
  FAISS                                     45%       75%       80%
  FAISS + CrossEncoder                      60%       80%       80%
  **FAISS + BM25 + RRF**                **60%**   **90%**   **95%**
  FAISS + BM25 + RRF + CrossEncoder         60%       90%       95%

### Dataset 2 --- Additional Evaluation Set

20 answerable cases.

  Retrieval Method                           @1        @3        @5
  ----------------------------------- --------- --------- ---------
  FAISS                                     45%       60%       75%
  FAISS + CrossEncoder                      50%       65%       75%
  **FAISS + BM25 + RRF**                **50%**   **70%**   **80%**
  FAISS + BM25 + RRF + CrossEncoder         50%       70%       80%

### Dataset 3 --- Independent Evaluation Set

18 answerable cases.

  Retrieval Method                           @1        @3        @5
  ----------------------------------- --------- --------- ---------
  FAISS                                     50%       61%       72%
  FAISS + CrossEncoder                      56%       61%       72%
  **FAISS + BM25 + RRF**                **56%**   **67%**   **78%**
  FAISS + BM25 + RRF + CrossEncoder         56%       67%       78%

## Final Hybrid Retrieval Page-Hit Results

The final **FAISS + BM25 + RRF** configuration produced the following
supporting-page hit rates.

  Dataset       Cases   Page Hit @1   Page Hit @3   Page Hit @5
  ----------- ------- ------------- ------------- -------------
  Dataset 1        20           40%           60%           80%
  Dataset 2        20            5%           15%           20%
  Dataset 3        18           17%           22%           22%

Page-hit accuracy is intentionally interpreted as a secondary
diagnostic. Answer-in-Context is more representative of retrieval
usefulness because correct specifications may occur on multiple manual
pages even when the expected supporting page is not retrieved.

## Key Findings

### 1. Hybrid retrieval improved over FAISS-only retrieval

The hybrid approach consistently improved Answer-in-Context results
across all three evaluation sets.

At Recall@5:

  Dataset       FAISS   Hybrid RRF   Absolute Change
  ----------- ------- ------------ -----------------
  Dataset 1       80%          95%            +15 pp
  Dataset 2       75%          80%             +5 pp
  Dataset 3       72%          78%             +6 pp

Across the 58 answerable cases, the reported percentages correspond to
approximately **76% weighted Answer@5 for FAISS-only retrieval** and
approximately **85% for hybrid RRF retrieval**, an improvement of
roughly **9 percentage points**.

### 2. BM25 complements dense semantic retrieval

FAISS is effective for semantic similarity, while BM25 helps recover
chunks containing exact technical terminology such as component names,
lubricant grades, identifiers, measurements, and specification-related
wording.

A representative failure analysis involved the query:

> How much SAE 75W-140 synthetic lubricant is required to fill the Ford
> 8.8-inch rear axle?

The expected page was absent from the FAISS Top-20 candidates, while
BM25 recovered it at rank 7. This demonstrated that lexical retrieval
can recover relevant evidence missed by dense retrieval.

### 3. RRF provides a simple way to combine both retrievers

FAISS similarity scores and BM25 scores are not directly comparable.
Reciprocal Rank Fusion combines their rankings instead of adding raw
scores.

The final retrieval flow is:

``` text
                         ┌── FAISS Top-20
Query ──────────────────┤
                         └── BM25 Top-20
                                  ↓
                         Reciprocal Rank Fusion
                                  ↓
                               Top-5
```

### 4. CrossEncoder did not improve the final hybrid metrics

Adding CrossEncoder reranking after FAISS + BM25 + RRF produced
**identical Answer-in-Context @1, @3, and @5 results** across all three
evaluation datasets.

It also produced identical reported page-hit metrics for the hybrid
runs.

Therefore, the CrossEncoder layer was removed from the final retrieval
architecture. This preserves the measured retrieval quality while
reducing:

-   model loading and download requirements,
-   inference latency,
-   memory usage,
-   implementation complexity, and
-   an additional runtime dependency.

## Final Retrieval Architecture

The selected retrieval pipeline is:

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

## Conclusion

The evaluation supports **FAISS + BM25 + Reciprocal Rank Fusion** as the
final retrieval strategy for this project.

Hybrid retrieval improved Answer-in-Context recall over the FAISS-only
baseline across all three evaluation datasets. CrossEncoder reranking
did not improve the measured hybrid retrieval metrics, so it was
excluded from the final system in favor of a simpler and lower-latency
pipeline.

The evaluation datasets are kept fixed for final reporting to avoid
tuning the retrieval system against the test cases.
