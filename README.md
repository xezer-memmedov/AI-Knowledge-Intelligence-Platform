#  AI Knowledge Intelligence Platform (Azerbaijani Verifiable RAG)

An enterprise-grade, **hallucination-free Retrieval-Augmented Generation (RAG)** platform specifically engineered for analyzing Azerbaijani corporate and regulatory PDF documents (e.g., Code of Ethics, internal policies).

Unlike standard generative RAG pipelines that risk paraphrasing errors or fabricating facts, this platform implements a **JSON-constrained Extractive Candidate Selection** architecture with an automated **Top-K Fallback Retry** mechanism.

---

## Key Engineering Features

* **Zero-Hallucination Guarantee:** Instead of letting the LLM freely generate text, retrieved chunks are segmented into atomic Azerbaijani sentences using custom regex rules (`(?<=[.!?])\s+(?=[A-ZƏÖÜĞÇŞİ])`). The LLM acts strictly as a semantic judge, outputting structured JSON (`{"decision": "answerable", "id": <int>}`) to return exact verbatim document excerpts.
* **Hardware-Optimized Hybrid Execution:**
  * **LLM (`Qwen/Qwen2.5-7B-Instruct`):** Quantized to **4-bit NF4** (`double_quant=True`, `float16` compute) via `bitsandbytes` to fit comfortably on a single T4 GPU.
  * **Retriever (`intfloat/multilingual-e5-small`):** Explicitly offloaded to **CPU** to preserve 100% of GPU VRAM for LLM inference.
* **Layout-Aware PDF Ingestion:** Uses `pypdf` in `layout` extraction mode alongside regex normalization to preserve numbered clauses (e.g., `2.11.1`) and clean out formatting artifacts.
* **Adaptive Top-K Fallback Retry:** If multi-chunk context (`top_k=3`) introduces noise and results in an `unknown` decision, the pipeline automatically retries with `top_k=1` to isolate the highest-scoring passage.
* **Prompt Injection Immunity:** System prompts enforce strict data-vs-instruction boundaries, ensuring malicious instructions inside PDFs are never executed.
* **Deterministic Checkpointing:** Exports parsed pages, word-overlapped chunks, normalized E5 embeddings (`.npy`), and model metadata with UTC timestamps for instant reload.

---

##  System Architecture

1. **Ingestion & Chunking:** PDF -> Layout Text Extraction -> Sliding Window Chunking (`120` words, `25` words overlap).
2. **Dense Retrieval:** Prefixed E5 encoding (`passage: ` / `query: `) -> L2 Normalization -> Exact Inner Product Search (`@`).
3. **Candidate Filtering:** Sentence boundary detection + trailing clause number stripping (`2.11.1`).
4. **LLM Verification:** `Qwen2.5-7B-Instruct` evaluates candidates -> Returns JSON ID -> Verifies existence -> Fallback retry (`top_k=1`) or flags `needs_review`.

---

## Installation & Dependencies

```bash
pip install torch transformers accelerate bitsandbytes sentence-transformers pypdf numpy

