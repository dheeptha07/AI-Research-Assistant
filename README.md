
# 🤖 AI Research Assistant — RAG-Based PDF Question Answering

An AI-powered research assistant that allows users to upload research papers in PDF format, ask questions in natural language, and receive context-grounded answers with source-page references.

Built using Retrieval-Augmented Generation (RAG), semantic embeddings, FAISS vector search, LangChain, and Streamlit.

## ✨ Features

- **PDF Upload and Processing:** Extract text from uploaded research papers.
- **Semantic Search:** Retrieve relevant passages using Hugging Face embeddings and FAISS similarity search.
- **Conversational Question Answering:** Ask questions and follow-up questions about the uploaded document.
- **Query Rewriting:** Reformulate questions using conversation history to improve retrieval.
- **Source-Page References:** Display retrieved excerpts with document names and page numbers.
- **Document Information:** View the uploaded filename, page count, and number of text chunks.
- **Context-Grounded Responses:** Instruct the language model to avoid inventing information absent from the retrieved context.
- **Session Controls:** Clear the conversation or active document.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| User Interface | Streamlit |
| LLM Integration | LangChain, ChatGroq |
| Language Model | OpenAI GPT-OSS 20B via Groq |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| PDF Extraction | PyPDFLoader, pypdf |
| Text Chunking | RecursiveCharacterTextSplitter |
| Configuration | python-dotenv |

## 🏗️ Architecture

1. **Upload:** The user uploads a research PDF through the Streamlit sidebar.
2. **Extract:** PyPDFLoader extracts text and page metadata.
3. **Chunk:** RecursiveCharacterTextSplitter divides text into chunks of 1,000 characters with 200 characters of overlap.
4. **Embed:** The Hugging Face embedding model converts chunks into vector representations.
5. **Index:** FAISS creates a vector index for similarity-based retrieval.
6. **Rewrite:** The language model reformulates the question using conversation history.
7. **Retrieve:** The retriever selects the five most relevant chunks.
8. **Generate:** The language model generates an answer using the retrieved context.
9. **Display:** The application presents the answer and expandable source excerpts with page references.

## 📂 Project Structure

```text
AI-Research-Assistant/
├── app.py
├── ingest.py
├── requirements.txt
├── README.md
└── .gitignore
```

`app.py` contains the main application, including PDF ingestion, chunking, embeddings, vector indexing, retrieval, answer generation, and source display.

`ingest.py` is currently empty and is not required by the application's existing execution flow.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/dheeptha07/AI-Research-Assistant.git
cd AI-Research-Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API Key

Create a file named `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own API key from the Groq Console.

**Security:** Never commit your real API key or `.env` file to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## 💬 Example Questions

After uploading a research paper, try asking:

- What is this paper about?
- What methodology did the authors use?
- What are the main findings?
- What did the authors propose?
- How many pages does this PDF contain?
- How many text chunks were created?
- Can you explain the methodology in simpler terms?

## 🧠 Retrieval and Answer Generation

The application uses a RAG pipeline to retrieve relevant document passages before generating an answer.

The embedding model supports semantic retrieval, while the language model performs query rewriting and answer generation. Retrieved passages are supplied as context, and the interface displays source excerpts associated with their PDF page numbers.

The application also handles basic document metadata questions, such as filename, page count, and chunk count, without requiring a generated answer.

## ⚠️ Limitations

- Answers depend on the quality and readability of the uploaded PDF.
- Scanned PDFs without extractable text may not be processed successfully.
- Retrieval may miss relevant information if the required passage is not among the selected chunks.
- Source excerpts provide supporting context but do not guarantee that every generated statement is fully supported by the cited pages.
- The application requires a valid Groq API key, network access, model availability, and applicable API limits.
- Conversation history and the active document are maintained in Streamlit session state; persistent conversation storage is not implemented.
- The current implementation processes one active document at a time.
- Formal retrieval-quality and answer-accuracy benchmarks have not yet been reported.

## 🚀 Future Improvements

- Evaluate retrieval relevance and answer faithfulness using a test dataset.
- Support multiple research papers in one session.
- Improve source attribution by linking individual claims to supporting passages.
- Add automated tests for document processing, retrieval, and answer generation.
- Add deployment instructions and screenshots of the working application.

## 👩‍💻 Author

**Dheeptha M**

- GitHub: [@dheeptha07](https://github.com/dheeptha07)
- Project Repository: [AI Research Assistant](https://github.com/dheeptha07/AI-Research-Assistant)

## 📄 License

No license has been specified yet. Add a license file if you intend to distribute or permit reuse of this project.
