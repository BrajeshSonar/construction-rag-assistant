# Construction Marketplace AI Assistant (RAG-based)

This project implements a **Retrieval-Augmented Generation (RAG)** system to answer construction-related queries using internal documents.  
The system retrieves relevant information from documents and generates **accurate, grounded, and explainable responses** using a Large Language Model.

---

## Features

- Semantic search using FAISS
- Context-aware answer generation using LLM
- Strict grounding (no hallucination)
- Source citation for transparency
- Interactive chatbot UI using Gradio
- FAQ quick access with clipboard copy

---

## System Architecture

User Query  
↓  
Embedding (Sentence Transformers)  
↓  
FAISS Vector Search (Top-K Retrieval)  
↓  
Context Injection  
↓  
LLM (OpenRouter - GPT-4o-mini)  
↓  
Final Answer + Sources  

---

## Project Structure


construction-rag-assistant/
│
├── app.py # Main application (RAG pipeline + UI)
├── notebook.ipynb # Development notebook
├── requirements.txt
├── README.md
│
├── mini_rag/
│ ├── data/ # Input documents
│ │ ├── doc1.md
│ │ ├── doc2.md
│ │ ├── doc3.md
│ ├── faiss.index # Saved FAISS index (optional)
│ └── chunks.pkl # Stored chunks (optional)


---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository

git clone https://github.com/BrajeshSonar/construction-rag-assistant.git

cd construction-rag-assistant


### 2. Create Virtual Environment (Recommended)

py -3.10 -m venv venv
venv\Scripts\activate


### 3. Install Dependencies

pip install -r requirements.txt


---

## 🔑 API Setup

1. Go to https://openrouter.ai  
2. Generate an API key  
3. Set it in your environment:

```python
import os
os.environ["OPENROUTER_API_KEY"] = "your-api-key"
▶️ Run the Application
python app.py

Then open:

http://127.0.0.1:7860
🧠 Methodology
1. Embedding Model and LLM Selection
Embedding Model
Model Used: all-MiniLM-L6-v2
Reason:
Lightweight and fast
Good semantic understanding
Works efficiently on CPU
LLM
Model Used: openai/gpt-4o-mini via OpenRouter
Reason:
Low latency and cost-effective
Strong performance for QA tasks
Works well with structured prompts
2. Document Chunking and Retrieval
Chunking Strategy
Fixed chunk size (e.g., 500 characters)
Overlap between chunks (e.g., 100 characters)

Benefits:

Preserves context
Improves retrieval accuracy
Prevents information loss
Embedding & Storage
Each chunk is converted into embeddings
Stored in a FAISS vector database
Retrieval Process
Query is embedded
FAISS performs cosine similarity search
Top-K relevant chunks are retrieved
3. Grounding to Retrieved Context

To ensure reliable responses:

Only retrieved context is passed to the LLM
The system prompt enforces:
No external knowledge usage
Strict reliance on provided context

If no answer is found:

"I could not find an answer in the provided documents."

Benefits:

Prevents hallucination
Ensures transparency
Provides traceable answers
💬 Chatbot Interface

Built using Gradio with:

Interactive chat UI
Answer + source display
FAQ buttons
Clipboard copy functionality
🧪 Example Query

Input:

What factors affect construction project delays?

Output:

Factors include lack of real-time tracking, failure to detect deviations,
and absence of accountability mechanisms (doc3.md).
⚠️ Limitations
Dependent on document quality
Chunking may split sentences
Requires API key for LLM
🔮 Future Improvements
Smarter chunking (sentence-based)
Improved ranking strategies
Better UI enhancements
Deployment optimization
👨‍💻 Author

Brajesh
Student

📜 License

This project is for academic and educational purposes.

⭐ Conclusion

This project demonstrates a complete RAG pipeline combining:

Retrieval (FAISS)
Generation (LLM)
Grounding (context-only answers)
Usability (chat interface)
