# Construction Marketplace AI Assistant (RAG-based)

This project implements a **Retrieval-Augmented Generation (RAG)** system to answer construction-related queries using internal documents.  
The system retrieves relevant information from documents and generates **accurate, grounded responses** using a Large Language Model.

---

## 🚀 Features

- Semantic search using FAISS
- Context-aware answer generation using LLM
- Strict grounding (no hallucination)
- Source citation for transparency
- Interactive chatbot UI using Gradio
- FAQ quick access with clipboard copy

---

## 🧱 System Architecture

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

## 📂 Project Structure


construction-rag-assistant/
│
├── app.py # Main application (RAG + UI)
├── notebook.ipynb # Development notebook
├── requirements.txt
├── README.md
│
├── mini_rag/
│ ├── data/ # Input documents
│ ├── faiss.index # Saved FAISS index
│ └── chunks.pkl # Stored chunks


---

## ⚙️ Installation

### 1. Clone Repository

git clone https://github.com/BrajeshSonar/construction-rag-assistant.git

cd construction-rag-assistant


### 2. Install Dependencies

pip install -r requirements.txt


---

## 🔑 API Setup

1. Go to https://openrouter.ai  
2. Generate an API key  
3. Set it in your code:

```python
import os
os.environ["OPENROUTER_API_KEY"] = "your-api-key"
📥 Data Preparation

Place your documents inside:

mini_rag/data/

Supported formats:

.md
.txt
.pdf
.docx
▶️ Running the Application

Run the chatbot:

python app.py

Then open the link shown in terminal:

http://127.0.0.1:7860
🧠 Model Details
Embedding Model
all-MiniLM-L6-v2
Lightweight and fast
Good semantic understanding
LLM
openai/gpt-4o-mini (via OpenRouter)
Low cost and fast
Reliable for grounded responses
✂️ Chunking Strategy
Fixed chunk size (e.g., 500 characters)
Overlap between chunks (e.g., 100 characters)

Benefits:

Preserves context
Improves retrieval accuracy
🔍 Retrieval Mechanism
Embeddings stored in FAISS index
Cosine similarity used for matching
Top-K relevant chunks retrieved
🛡️ Grounding Strategy

The system enforces grounding by:

Passing only retrieved context to the LLM
Restricting the model from using external knowledge
Returning "not found" if context is insufficient

This ensures:

No hallucination
Reliable answers
Source transparency
💬 Chatbot Interface

Built using Gradio with features:

Interactive chat UI
Answer + source display
FAQ quick buttons
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
Better ranking strategies
UI enhancements
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
Transparency (source citation)
Usability (chat interface)

It ensures accurate and explainable AI responses for construction-related queries.
