## Step 1: Import Libraries and Setup Environment
import os
import json
import pickle
import textwrap
from pathlib import Path

import fitz  # PyMuPDF — for PDF parsing
import docx  # python-docx — for .docx parsing
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from openai import OpenAI  # OpenRouter uses the OpenAI-compatible SDK

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR        = Path("mini_rag/data")          # Folder with your uploaded docs
INDEX_PATH      = Path("mini_rag/faiss.index")   # Saved FAISS index
CHUNKS_PATH     = Path("mini_rag/chunks.pkl")    # Saved chunk metadata

EMBED_MODEL     = "all-MiniLM-L6-v2"            # Fast, free, great quality
CHUNK_SIZE      = 500                            # Characters per chunk
CHUNK_OVERLAP   = 100                            # Overlap between chunks
TOP_K           = 4                              # Number of chunks to retrieve

# OpenRouter settings (get a free key at https://openrouter.ai)
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"  # ← Replace this
LLM_MODEL          = "mistralai/mistral-7b-instruct:free"

print("✅ Config ready.")
print(f"   Data dir  : {DATA_DIR}")
print(f"   Embed model: {EMBED_MODEL}")
print(f"   LLM model  : {LLM_MODEL}")


## Step 2: Load Documents

from pathlib import Path
import fitz
import docx


def load_pdf(path: Path) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(str(path))
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def load_docx(path: Path) -> str:
    """Extract all text from a DOCX file."""
    doc = docx.Document(str(path))
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def load_txt(path: Path) -> str:
    """Load a plain text file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_md(path: Path) -> str:
    """Load a markdown file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(data_dir: Path) -> list[dict]:
    """Load all supported documents from data_dir."""
    documents = []

    supported = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
        ".md": load_md,   
    }

    files = list(data_dir.glob("**/*"))

    if not files:
        print(f"No files found in {data_dir}. Please add your documents there.")
        return documents

    for file in files:
        if file.suffix.lower() in supported:
            loader = supported[file.suffix.lower()]
            try:
                text = loader(file)
                documents.append({"filename": file.name, "text": text})
                print(f"Loaded: {file.name} ({len(text)} chars)")
            except Exception as e:
                print(f"Failed to load {file.name}: {e}")
        else:
            print(f"Skipped unsupported file: {file.name}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents



DATA_DIR = Path("mini_rag/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

documents = load_documents(DATA_DIR)

## Step 3: Chunk Documents
import textwrap

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_chunks(documents: list[dict]) -> list[dict]:
    """Chunk all documents and attach metadata."""
    all_chunks = []

    for doc in documents:
        raw_chunks = chunk_text(doc["text"])

        for i, chunk in enumerate(raw_chunks):
            all_chunks.append({
                "chunk_id": len(all_chunks),
                "filename": doc["filename"],
                "chunk_index": i,
                "text": chunk,
            })

        print(f"{doc['filename']}: {len(raw_chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    return all_chunks


# Run
chunks = build_chunks(documents)

# Preview
if chunks:
    print("\nSample chunk:\n")
    print(textwrap.fill(chunks[0]["text"][:300], width=80))

## Step 4: Load Embedding Model
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"

print(f"Loading embedding model: {EMBED_MODEL} ...")
embedder = SentenceTransformer(EMBED_MODEL)
print("Model loaded.")


def embed_chunks(chunks: list[dict], batch_size: int = 64) -> np.ndarray:
    """Generate embeddings for all chunks."""
    
    if not chunks:
        raise ValueError("No chunks found. Please run chunking step first.")

    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")

    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # Enables cosine similarity via dot product
    )

    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings.astype("float32")


# Run
embeddings = embed_chunks(chunks)



## Step 5: Generate Embeddings

import faiss
import pickle
from pathlib import Path


INDEX_PATH = Path("mini_rag/faiss.index")
CHUNKS_PATH = Path("mini_rag/chunks.pkl")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS inner-product index (cosine similarity for normalized vectors)."""

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Embeddings are empty. Run embedding step first.")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine similarity (since normalized)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors (dim={dim})")

    return index


# Build index
index = build_faiss_index(embeddings)


# Save index and chunks
Path("mini_rag").mkdir(exist_ok=True)

faiss.write_index(index, str(INDEX_PATH))

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)


print(f"Index saved to: {INDEX_PATH}")
print(f"Chunks saved to: {CHUNKS_PATH}")


## Step 6: Build FAISS Index

import textwrap


def load_index_and_chunks():
    """Load saved FAISS index and chunks from disk."""
    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def retrieve(query: str, index, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a query."""

    if not query.strip():
        raise ValueError("Query is empty.")

    if index is None or not chunks:
        raise ValueError("Index or chunks not loaded properly.")

    # Embed query
    query_vec = embedder.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    # Search
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        result = chunks[idx].copy()
        result["similarity_score"] = round(float(score), 4)
        results.append(result)

    return results


# Test retrieval
if chunks:
    test_query = "What factors affect construction project delays?"
    retrieved = retrieve(test_query, index, chunks)

    print(f"Query: {test_query}\n")

    for i, r in enumerate(retrieved, 1):
        print(f"Chunk #{i} | Source: {r['filename']} | Score: {r['similarity_score']}")
        print(textwrap.fill(r["text"][:300], width=80))
        print()


## Step 7: Retrieve Relevant Chunks

import os
from openai import OpenAI

# Set your API key (replace with your NEW key safely)
os.environ["OPENROUTER_API_KEY"] = "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"

# Initialize OpenRouter client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Use a valid working model
LLM_MODEL = "openai/gpt-4o-mini"


SYSTEM_PROMPT = """You are an AI assistant for a construction marketplace.

Answer the question using ONLY the provided context.

You are allowed to:
- Summarize information from multiple context sections
- Combine related points into a clear answer

Do NOT:
- Add information not present in the context
- Use outside knowledge

If the answer is clearly not present, say:
"I could not find an answer in the provided documents."

When answering:
- Cite the source document names (e.g., doc1.md, doc3.md)
- Do NOT refer to context numbers like "Context 1"

Be clear and concise.
"""


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Build the user prompt with retrieved context injected."""

    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"[Context {i} — Source: {chunk['filename']}]\n{chunk['text']}"
        )

    context_str = "\n\n".join(context_blocks)

    return f"""Use the following retrieved document excerpts to answer the question.

{context_str}

Question: {query}

Answer (based strictly on the context above):
"""


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """Call the LLM to generate a grounded answer."""

    if not query.strip():
        raise ValueError("Query is empty.")

    if not retrieved_chunks:
        return "No relevant context found in the documents."

    prompt = build_prompt(query, retrieved_chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()


print("LLM generation function ready.")


## Step 8: Full RAG Pipeline

import textwrap


def rag_query(query: str, verbose: bool = True) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks
    2. Generate grounded answer
    3. Return context + answer (transparent)
    """

    if not query.strip():
        raise ValueError("Query is empty.")

    # Step 1: Retrieve
    retrieved = retrieve(query, index, chunks)

    # Step 2: Generate
    answer = generate_answer(query, retrieved)

    result = {
        "query": query,
        "retrieved_chunks": retrieved,
        "answer": answer,
    }

    if verbose:
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        print("\nRETRIEVED CONTEXT:")
        if not retrieved:
            print("No relevant chunks found.")
        else:
            for i, chunk in enumerate(retrieved, 1):
                score = chunk.get("similarity_score", "N/A")
                print(f"\n[{i}] Source: {chunk['filename']} | Score: {score}")
                print(textwrap.fill(chunk["text"][:400], width=78))

        print("\n" + "-" * 70)

        print("\nGENERATED ANSWER:")
        print(textwrap.fill(answer, width=70))

        print("=" * 70)

    return result


# Test
if chunks:
    result = rag_query("What factors affect construction project delays?")

## Step 9: Chatbot Interface (Gradio)

import gradio as gr


def chat_fn(user_message, history):
    if history is None:
        history = []

    if not user_message.strip():
        return history, ""

    # Add user message
    history.append({
        "role": "user",
        "content": user_message
    })

    # Run RAG
    result = rag_query(user_message, verbose=False)
    answer = result["answer"]

    # Build sources
    sources = "\n\nSources Used:\n"
    seen = set()

    for chunk in result["retrieved_chunks"]:
        key = (chunk["filename"], chunk["chunk_index"])
        if key not in seen:
            score = chunk.get("similarity_score", "N/A")
            sources += f"- {chunk['filename']} (chunk {chunk['chunk_index']}, score: {score})\n"
            seen.add(key)

    full_response = answer + sources

    # Add assistant response
    history.append({
        "role": "assistant",
        "content": full_response
    })

    return history, ""


with gr.Blocks(title="Construction RAG Assistant") as demo:

    gr.Markdown("""
    # Construction Marketplace AI Assistant

    Ask questions about construction policies, FAQs, and specifications.
    Answers are grounded strictly in the provided internal documents.
    """)

    chatbot = gr.Chatbot(label="Chat", height=450)

    msg_box = gr.Textbox(
        placeholder="Ask a question about construction...",
        label="Your Question"
    )

    send_btn = gr.Button("Send")

    # FAQ Section
    with gr.Column():
        gr.Markdown("### Frequently Asked Questions")

        faq_questions = [
            "What factors affect construction project delays?",
            "What quality assurance measures are followed?",
            "What is the payment safety system?",
            "What maintenance services are provided after construction?",
        ]

        for q in faq_questions:
            btn = gr.Button(q)

            btn.click(
                lambda x=q: x,
                inputs=[],
                outputs=msg_box,
            ).then(
                None,
                None,
                None,
                js=f"""
                () => {{
                    navigator.clipboard.writeText("{q}");
                }}
                """
            )

    clear_btn = gr.Button("Clear Chat")

    # Events
    send_btn.click(chat_fn, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])
    msg_box.submit(chat_fn, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])


print("Launching Gradio app...")
demo.launch(share=True)