"""Mock objects for testing RAG system components"""

from typing import List, Dict, Any, Optional
from unittest.mock import Mock, MagicMock
from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


class MockVectorStore:
    """Mock vector store for testing"""
    
    def __init__(self, search_results: Optional[SearchResults] = None, should_raise: bool = False):
        self.search_results = search_results
        self.should_raise = should_raise
        self.search_calls = []
        
    def search(self, query: str, course_name: Optional[str] = None, 
               lesson_number: Optional[int] = None, limit: Optional[int] = None) -> SearchResults:
        """Mock search method that records calls and returns predefined results"""
        self.search_calls.append({
            'query': query,
            'course_name': course_name,
            'lesson_number': lesson_number,
            'limit': limit
        })
        
        if self.should_raise:
            raise Exception("Mock search error")
            
        if self.search_results:
            return self.search_results
        
        # Default empty results
        return SearchResults(documents=[], metadata=[], distances=[])
    
    def get_lesson_link(self, course_title: str, lesson_num: int) -> Optional[str]:
        """Mock lesson link getter"""
        return f"https://example.com/{course_title}/lesson-{lesson_num}"


class MockToolManager:
    """Mock tool manager for testing"""
    
    def __init__(self, tool_response: str = "Mock tool response", should_raise: bool = False):
        self.tool_response = tool_response
        self.should_raise = should_raise
        self.execute_calls = []
        self.tools = {}
        
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Mock tool execution"""
        self.execute_calls.append({
            'tool_name': tool_name,
            'kwargs': kwargs
        })
        
        if self.should_raise:
            raise Exception("Mock tool execution error")
            
        return self.tool_response
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Mock tool definitions"""
        return [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "course_name": {"type": "string"},
                        "lesson_number": {"type": "integer"}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def get_last_sources(self) -> List[Dict[str, Any]]:
        """Mock sources getter"""
        return [{"text": "Mock Course - Lesson 1", "link": "https://example.com"}]
    
    def reset_sources(self):
        """Mock sources reset"""
        pass


class MockAnthropicClient:
    """Mock Anthropic client for testing"""
    
    def __init__(self, response_text: str = "Mock AI response", should_use_tool: bool = False, should_raise: bool = False):
        self.response_text = response_text
        self.should_use_tool = should_use_tool
        self.should_raise = should_raise
        self.messages_calls = []
        self.call_count = 0
        
    def create(self, **kwargs):
        """Mock message creation"""
        self.messages_calls.append(kwargs)
        self.call_count += 1
        
        if self.should_raise:
            raise Exception("Mock API error")
        
        # Create mock response
        mock_response = Mock()
        
        # For tool use tests, first call should be tool use, second should be text
        if self.should_use_tool and self.call_count == 1:
            # Mock tool use response
            mock_response.stop_reason = "tool_use"
            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.input = {"query": "test query"}
            mock_tool_block.id = "tool_call_123"
            mock_response.content = [mock_tool_block]
        else:
            # Mock direct text response
            mock_response.stop_reason = "end_turn"
            mock_text_block = Mock()
            mock_text_block.text = self.response_text
            mock_response.content = [mock_text_block]
            
        return mock_response


class MockSessionManager:
    """Mock session manager for testing"""
    
    def __init__(self):
        self.sessions = {}
        
    def create_session(self) -> str:
        """Mock session creation"""
        return "test_session_123"
    
    def get_conversation_history(self, session_id: str) -> Optional[str]:
        """Mock conversation history"""
        return self.sessions.get(session_id, "Previous conversation context")
    
    def add_exchange(self, session_id: str, query: str, response: str):
        """Mock adding conversation exchange"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ""
        self.sessions[session_id] += f"User: {query}\nAssistant: {response}\n"


def create_sample_search_results(with_error: bool = False, empty: bool = False) -> SearchResults:
    """Create sample search results for testing"""
    if with_error:
        return SearchResults.empty("Search error occurred")
    
    if empty:
        return SearchResults(documents=[], metadata=[], distances=[])
    
    return SearchResults(
        documents=[
            "Course Introduction to Python Lesson 1 content: This is the first lesson about Python basics.",
            "Course Introduction to Python Lesson 2 content: This lesson covers variables and data types."
        ],
        metadata=[
            {"course_title": "Introduction to Python", "lesson_number": 1},
            {"course_title": "Introduction to Python", "lesson_number": 2}
        ],
        distances=[0.2, 0.3]
    )


def create_sample_course() -> Course:
    """Create sample course for testing"""
    return Course(
        title="Introduction to Python",
        instructor="John Doe",
        course_link="https://example.com/python-course",
        lessons=[
            Lesson(lesson_number=1, title="Python Basics", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Variables and Data Types", lesson_link="https://example.com/lesson2")
        ]
    )


def create_sample_course_chunks() -> List[CourseChunk]:
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Course Introduction to Python Lesson 1 content: This is the first lesson about Python basics.",
            course_title="Introduction to Python",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Course Introduction to Python Lesson 2 content: This lesson covers variables and data types.",
            course_title="Introduction to Python",
            lesson_number=2,
            chunk_index=1
        )
    ]