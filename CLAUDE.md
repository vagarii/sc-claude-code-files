# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Dependency Management
```bash
# Install/sync dependencies
uv sync

# Add new dependency
uv add <package-name>
```

### Code Quality
```bash
# Format code with black
./scripts/format.sh

# Check code formatting
./scripts/check.sh

# Manual formatting commands
uv run black backend/ main.py          # Format all Python files
uv run black backend/ main.py --check  # Check formatting without changes
```

### Environment Setup
Create `.env` file in root with:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials using Anthropic's Claude AI. The system follows a modular architecture with clear separation between document processing, vector storage, AI generation, and search capabilities.

### Core Architecture Pattern

The system uses an **agent-based approach** where Claude autonomously decides whether to search course materials or answer from general knowledge. This is implemented through:

1. **Tool-based AI System**: Claude has access to search tools and decides when to use them
2. **Multi-layered Processing**: Frontend → FastAPI → RAG System → AI Generator → Search Tools → Vector Store
3. **Session-based Conversations**: Maintains context across multiple queries using session management

### Key Components

- **RAG System (`rag_system.py`)**: Main orchestrator that coordinates between all components
- **AI Generator (`ai_generator.py`)**: Handles Anthropic Claude API interactions with tool integration
- **Vector Store (`vector_store.py`)**: ChromaDB wrapper for semantic search with sentence transformers
- **Document Processor (`document_processor.py`)**: Handles course document parsing and text chunking
- **Search Tools (`search_tools.py`)**: Implements search capabilities as tools for Claude
- **Session Manager (`session_manager.py`)**: Manages conversation history and context

### Document Processing Pipeline

Documents must follow this specific format:
```
Course Title: [title]
Course Link: [url] 
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [lesson_url]
[lesson content...]
```

The processor:
1. Extracts metadata from the first 3 lines using regex patterns
2. Parses lessons using "Lesson X:" markers
3. Chunks text using sentence-based splitting with configurable overlap
4. Adds contextual prefixes: "Course [title] Lesson [X] content: [chunk]"
5. Stores in ChromaDB with embeddings for semantic search

### Data Models

The system uses three main Pydantic models:
- **Course**: Represents complete course with lessons list
- **Lesson**: Individual lesson with number, title, and optional link
- **CourseChunk**: Text chunk with course/lesson metadata for vector storage

### Configuration

All settings are centralized in `config.py`:
- **Chunk Settings**: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`
- **AI Model**: `claude-sonnet-4-20250514`
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Search Limits**: `MAX_RESULTS=5`, `MAX_HISTORY=2`

### Query Processing Flow

1. **Frontend**: User input → POST `/api/query` with session management
2. **FastAPI**: Validates request → calls `rag_system.query()`
3. **RAG System**: Builds prompt → retrieves session history → calls AI generator
4. **AI Generator**: Claude decides autonomously whether to search or use knowledge
5. **Search Tools**: If search needed → semantic search in ChromaDB
6. **Vector Store**: Embedding-based similarity search → returns ranked chunks
7. **Response**: AI synthesizes results → sources tracked → session updated

### Frontend Architecture

Simple HTML/CSS/JS interface with:
- Real-time chat interface with markdown rendering
- Loading animations and error handling
- Course statistics sidebar
- Session-based conversation history

### Key Technical Decisions

- **Agent-based AI**: Claude makes autonomous decisions about when to search vs. use general knowledge
- **Sentence-based chunking**: Preserves semantic integrity across chunk boundaries
- **Contextual chunk prefixes**: Improves search relevance by including course/lesson context
- **Session management**: Maintains conversation context without persistent storage
- **Tool-based search**: Cleaner separation between AI reasoning and search execution

### Adding New Course Documents

Place `.txt` files in `/docs` folder following the expected format. The system automatically loads documents on startup and avoids re-processing existing courses.