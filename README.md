# Agentic Document Intelligence System

An internship-ready Agentic RAG application that works like a document-aware assistant for internal company files. Users can upload PDFs, PowerPoint decks, and text files, index them into a vector database, ask natural-language questions, and export generated answers as template-based PPTX, PDF, or Excel reports.

## What It Does

- Extracts text from PDF, PPTX, and TXT documents.
- Splits long documents into smaller searchable chunks.
- Creates Gemini embeddings and stores them in ChromaDB.
- Uses a LangGraph workflow to retrieve local evidence first.
- Falls back to Tavily web search when local evidence is weak or unavailable.
- Generates cited answers through Gemini, OpenAI, or Groq from a single UI.
- Supports model tuning controls for temperature, max tokens, and top-p.
- Provides a browser UI for upload, indexing, querying, source review, and report export.
- Generates template-based PPTX decks with references separated into a dedicated slide.

## Architecture

```text
User Interface
    |
    v
FastAPI Backend
    |
    +--> Document Processor
    |       |
    |       +--> PDF / PPTX / TXT extraction
    |       +--> Chunking
    |
    +--> Vector Store Manager
    |       |
    |       +--> Gemini embeddings
    |       +--> ChromaDB persistent memory
    |
    +--> LangGraph Agent Workflow
            |
            +--> Retrieval Agent
            +--> Router Agent
            +--> Web Search Agent
            +--> Configurable Gemini / OpenAI / Groq Response Generator
            +--> PPTX / PDF / Excel Formatter
```

## Folder Structure

```text
agentic-rag-system/
├── agent_graph.py          # LangGraph agent workflow
├── document_processor.py   # File extraction and chunking
├── vector_store.py         # ChromaDB and embedding operations
├── main.py                 # FastAPI app, upload, query, reports, static UI
├── static/                 # Browser dashboard
├── data/                   # Uploaded source documents
├── db/                     # Persistent ChromaDB files
├── outputs/                # Generated reports
└── requirements.txt
```

## Setup

### Option 1: Run With Docker

This is the recommended way to share the project with a mentor because Docker creates the same runtime environment on any machine.

1. Create the environment file:

```bash
cp .env.example .env
```

2. Add your API keys inside `.env`:

```bash
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

`GOOGLE_API_KEY` is required for embeddings and document indexing. `OPENAI_API_KEY`, `GROQ_API_KEY`, and `TAVILY_API_KEY` are only required when using those providers/features.

3. Build and start the app:

```bash
docker compose up --build
```

4. Open the UI:

```text
http://localhost:8000
```

Docker mounts these folders so data remains available after the container restarts:

```text
data/      uploaded documents
db/        persistent ChromaDB vector database
outputs/   generated PPTX, PDF, and Excel reports
```

To stop the app:

```bash
docker compose down
```

### Option 2: Run Locally With Python

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file:

```bash
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

`OPENAI_API_KEY` is only required when selecting OpenAI in the UI. `GROQ_API_KEY` is only required when selecting Groq. `TAVILY_API_KEY` is optional. Without it, the system still works for local document Q&A.

3. Run the application:

```bash
uvicorn main:app --reload
```

4. Open the UI:

```text
http://127.0.0.1:8000
```

## How To Use

1. Upload internal documents from the sidebar.
2. Click `Index` to convert documents into searchable vector memory.
3. Select Gemini, OpenAI, or Groq, choose the model name, and adjust temperature, max tokens, and top-p if needed.
4. Ask a question in the main workspace.
5. Review the generated answer and supporting sources.
6. Choose `PPT`, `PDF`, or `Excel` to download a formatted report.

## API Endpoints

- `GET /health` checks if the backend is running.
- `GET /api/status` returns document, embedding, and source status.
- `POST /api/upload` uploads PDF, PPTX, or TXT files.
- `POST /api/ingest` indexes uploaded files into ChromaDB.
- `POST /api/ask` asks the agent and returns JSON, PPTX, PDF, or Excel output.
- `GET /api/sources` lists indexed source documents.

## Mentor Review Notes

This project demonstrates a complete RAG pipeline rather than a basic chatbot. It includes document ingestion, semantic search, agentic routing, web fallback, source-aware generation, persistent memory, API endpoints, template-based PPTX generation, report exports, and an interactive UI.
