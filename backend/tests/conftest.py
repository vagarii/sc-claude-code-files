"""Shared test fixtures and configuration for the RAG system tests."""

import os
import tempfile
import shutil
from typing import Generator, Dict, Any
from unittest.mock import Mock, AsyncMock

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from config import Config
from models import Course, Lesson, CourseChunk


@pytest.fixture(scope="session")
def test_config() -> Config:
    """Create a test configuration with isolated settings."""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.VECTOR_DB_PATH = ":memory:"  # Use in-memory ChromaDB for tests
    config.CHUNK_SIZE = 100
    config.CHUNK_OVERLAP = 20
    config.MAX_RESULTS = 3
    return config


@pytest.fixture
def temp_docs_dir() -> Generator[str, None, None]:
    """Create a temporary directory with test course documents."""
    temp_dir = tempfile.mkdtemp()
    
    # Create test course files
    course1_content = """Course Title: Test Python Programming
Course Link: https://example.com/python
Course Instructor: Test Instructor

Lesson 0: Introduction
Lesson Link: https://example.com/python/intro
Welcome to Python programming. This is an introduction to the basics.

Lesson 1: Variables and Data Types
Variables in Python are containers for storing data values.
Python has various data types including integers, floats, and strings."""
    
    course2_content = """Course Title: Test Web Development
Course Link: https://example.com/web
Course Instructor: Web Instructor

Lesson 0: HTML Basics
HTML is the standard markup language for creating web pages.

Lesson 1: CSS Styling
CSS is used for describing the presentation of a document written in HTML."""
    
    with open(os.path.join(temp_dir, "course1.txt"), "w") as f:
        f.write(course1_content)
    
    with open(os.path.join(temp_dir, "course2.txt"), "w") as f:
        f.write(course2_content)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_course() -> Course:
    """Create a sample course for testing."""
    return Course(
        title="Test Course",
        link="https://example.com/test",
        instructor="Test Instructor",
        lessons=[
            Lesson(number=0, title="Introduction", link="https://example.com/test/intro"),
            Lesson(number=1, title="Advanced Topics")
        ]
    )


@pytest.fixture
def sample_course_chunks() -> list[CourseChunk]:
    """Create sample course chunks for testing."""
    return [
        CourseChunk(
            text="Course Test Course Lesson 0 content: This is the introduction to the test course.",
            course_title="Test Course",
            lesson_number=0,
            lesson_title="Introduction"
        ),
        CourseChunk(
            text="Course Test Course Lesson 1 content: This covers advanced topics in the course.",
            course_title="Test Course", 
            lesson_number=1,
            lesson_title="Advanced Topics"
        )
    ]


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing AI generation."""
    mock_client = Mock()
    
    # Mock successful response
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response from Claude.")]
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = Mock()
    mock_store.add_chunks.return_value = None
    mock_store.search.return_value = [
        ("Course Test Course Lesson 0 content: Sample content.", 0.9),
        ("Course Test Course Lesson 1 content: More content.", 0.8)
    ]
    mock_store.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }
    return mock_store


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing."""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test-session-123"
    mock_manager.get_session_history.return_value = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ]
    mock_manager.add_to_session.return_value = None
    return mock_manager


@pytest.fixture
def test_app() -> FastAPI:
    """Create a test FastAPI app without static file mounting."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create test app without static file mounting issues
    app = FastAPI(title="Test Course Materials RAG System")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define the same Pydantic models as the main app
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system for testing
    mock_rag = Mock()
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    mock_rag.query.return_value = ("Test answer", ["Source 1", "Source 2"])
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}
    
    # Attach mock for test access
    app.state.mock_rag = mock_rag
    
    return app


@pytest.fixture
def test_client(test_app: FastAPI) -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(test_app)


@pytest.fixture
def sample_query_request() -> Dict[str, Any]:
    """Sample query request data for testing."""
    return {
        "query": "What is Python programming?",
        "session_id": None
    }


@pytest.fixture
def sample_query_request_with_session() -> Dict[str, Any]:
    """Sample query request with session ID for testing."""
    return {
        "query": "Tell me more about variables",
        "session_id": "existing-session-123"
    }