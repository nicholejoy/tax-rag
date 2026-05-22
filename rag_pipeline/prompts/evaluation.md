Build a production-ready RAG evaluation framework using the RAGAS library.

Context:

* I already have a working RAG pipeline (implementation can be found in rag_pipeline):

  * FAISS for vector storage
  * HuggingFace embeddings
  * FastAPI endpoint: /query
* Evaluation dataset is in evaluation_data/input.jsonl

Dataset format:
{
"question": "...",
"ground_truth": "...",
"context": "...",
"chunk_id": "...",
"url": "...",
"source_title": "..."
}

GOAL:
Do NOT reimplement RAG evaluation metrics manually.
Use RAGAS as the primary evaluation engine and build a clean wrapper around it.

---

Architecture requirement:

Create module:
evaluation/

Runner should:

* load dataset
* batch process evaluation
* print summary metrics at end

---

7. Testing:
   Add pytest suite:
   tests/
   test_ragas_integration.py
   test_retrieval_metrics.py
   test_runner_smoke.py

Tests should:

* validate RAGAS outputs are returned
* validate dataset loader
* run a small end-to-end smoke test with 2–3 samples

---

8. Docker:
   Add Dockerfile to run evaluation locally.

---

IMPORTANT CONSTRAINTS:

* Do NOT implement custom faithfulness / relevancy scoring logic
* Do NOT duplicate RAGAS metrics
* Only wrap RAGAS cleanly and standardize input/output formats
* Keep retrieval metrics separate and custom

---

DESIGN GOAL:
The system should act as a thin evaluation orchestration layer over RAGAS + FAISS retrieval metrics, not a reimplementation of evaluation logic.
