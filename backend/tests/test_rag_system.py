"""End-to-end tests for RAG system content-query handling"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the backend directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from test_mocks import (
    MockVectorStore, MockAnthropicClient, MockToolManager, MockSessionManager,
    create_sample_search_results, create_sample_course, create_sample_course_chunks
)


class TestRAGSystemQuery(unittest.TestCase):
    """Test RAG system query handling end-to-end"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock config
        self.mock_config = Mock()
        self.mock_config.CHUNK_SIZE = 800
        self.mock_config.CHUNK_OVERLAP = 100
        self.mock_config.CHROMA_PATH = "/tmp/test_chroma"
        self.mock_config.EMBEDDING_MODEL = "test-model"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test_key"
        self.mock_config.ANTHROPIC_MODEL = "test_model"
        self.mock_config.MAX_HISTORY = 2
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_successful_content_query(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc):
        """Test successful content-related query processing"""
        # Setup mocks
        mock_vector_store_instance = MockVectorStore(create_sample_search_results())
        mock_vector_store.return_value = mock_vector_store_instance
        
        mock_ai_gen_instance = Mock()
        mock_ai_gen_instance.generate_response.return_value = "Python variables are containers for storing data values."
        mock_ai_gen.return_value = mock_ai_gen_instance
        
        mock_session_mgr_instance = MockSessionManager()
        mock_session_mgr.return_value = mock_session_mgr_instance
        
        mock_doc_proc.return_value = Mock()
        
        # Create RAG system
        rag = RAGSystem(self.mock_config)
        
        # Override tool manager with mock
        mock_tool_manager = MockToolManager("Search results from CourseSearchTool")
        rag.tool_manager = mock_tool_manager
        
        # Execute query
        response, sources = rag.query("What are Python variables?")
        
        # Verify AI generator was called with correct parameters
        mock_ai_gen_instance.generate_response.assert_called_once()
        call_args = mock_ai_gen_instance.generate_response.call_args
        
        # Check that prompt was created correctly
        self.assertIn("What are Python variables?", call_args[1]['query'])
        
        # Check that tools were provided
        self.assertIsNotNone(call_args[1]['tools'])
        self.assertIsNotNone(call_args[1]['tool_manager'])
        
        # Verify response
        self.assertEqual(response, "Python variables are containers for storing data values.")
        self.assertIsInstance(sources, list)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_session_context(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc):
        """Test query processing with session context"""
        # Setup mocks
        mock_vector_store.return_value = MockVectorStore()
        
        mock_ai_gen_instance = Mock()
        mock_ai_gen_instance.generate_response.return_value = "Based on our previous discussion..."
        mock_ai_gen.return_value = mock_ai_gen_instance
        
        mock_session_mgr_instance = MockSessionManager()
        mock_session_mgr.return_value = mock_session_mgr_instance
        
        mock_doc_proc.return_value = Mock()
        
        # Create RAG system
        rag = RAGSystem(self.mock_config)
        rag.tool_manager = MockToolManager()
        
        # Execute query with session
        session_id = "test_session_123"
        response, sources = rag.query("Continue the explanation", session_id=session_id)
        
        # Verify session history was retrieved and passed
        call_args = mock_ai_gen_instance.generate_response.call_args
        self.assertIsNotNone(call_args[1]['conversation_history'])
        
        # Verify session was updated
        self.assertIn(session_id, mock_session_mgr_instance.sessions)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_failure_handling(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc):
        """Test handling of query failures"""
        # Setup mocks
        mock_vector_store.return_value = MockVectorStore()
        
        # Make AI generator raise an exception
        mock_ai_gen_instance = Mock()
        mock_ai_gen_instance.generate_response.side_effect = Exception("API Error")
        mock_ai_gen.return_value = mock_ai_gen_instance
        
        mock_session_mgr.return_value = MockSessionManager()
        mock_doc_proc.return_value = Mock()
        
        # Create RAG system
        rag = RAGSystem(self.mock_config)
        
        # Execute query and expect exception to be raised
        with self.assertRaises(Exception) as context:
            rag.query("Test query")
        
        self.assertIn("API Error", str(context.exception))
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_tool_manager_integration(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc):
        """Test that RAG system properly integrates with tool manager"""
        # Setup mocks
        mock_vector_store_instance = MockVectorStore(create_sample_search_results())
        mock_vector_store.return_value = mock_vector_store_instance
        
        mock_ai_gen_instance = Mock()
        mock_ai_gen_instance.generate_response.return_value = "Tool-based response"
        mock_ai_gen.return_value = mock_ai_gen_instance
        
        mock_session_mgr.return_value = MockSessionManager()
        mock_doc_proc.return_value = Mock()
        
        # Create RAG system
        rag = RAGSystem(self.mock_config)
        
        # Execute query
        response, sources = rag.query("Search for Python content")
        
        # Verify tools were registered and available
        self.assertIsNotNone(rag.tool_manager)
        self.assertIsNotNone(rag.search_tool)
        self.assertIsNotNone(rag.outline_tool)
        
        # Verify AI generator was called with tool definitions
        call_args = mock_ai_gen_instance.generate_response.call_args
        tools = call_args[1]['tools']
        tool_manager = call_args[1]['tool_manager']
        
        self.assertIsInstance(tools, list)
        self.assertIsNotNone(tool_manager)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_sources_handling(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc):
        """Test that sources are properly retrieved and reset"""
        # Setup mocks
        mock_vector_store.return_value = MockVectorStore()
        mock_ai_gen.return_value = Mock(generate_response=Mock(return_value="Response"))
        mock_session_mgr.return_value = MockSessionManager()
        mock_doc_proc.return_value = Mock()
        
        # Create RAG system with mock tool manager
        rag = RAGSystem(self.mock_config)
        
        # Create mock tool manager that tracks sources
        mock_sources = [{"text": "Test Course - Lesson 1", "link": "https://test.com"}]
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = []
        mock_tool_manager.get_last_sources.return_value = mock_sources
        mock_tool_manager.reset_sources = Mock()
        
        rag.tool_manager = mock_tool_manager
        
        # Execute query
        response, sources = rag.query("Test query")
        
        # Verify sources were retrieved and reset
        mock_tool_manager.get_last_sources.assert_called_once()
        mock_tool_manager.reset_sources.assert_called_once()
        self.assertEqual(sources, mock_sources)


class TestRAGSystemContentTypes(unittest.TestCase):
    """Test RAG system with different types of content queries"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = Mock()
        self.mock_config.CHUNK_SIZE = 800
        self.mock_config.CHUNK_OVERLAP = 100
        self.mock_config.CHROMA_PATH = "/tmp/test_chroma"
        self.mock_config.EMBEDDING_MODEL = "test-model"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test_key"
        self.mock_config.ANTHROPIC_MODEL = "test_model"
        self.mock_config.MAX_HISTORY = 2
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_specific_course_query(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc):
        """Test query for specific course content"""
        # Setup mocks
        mock_vector_store.return_value = MockVectorStore(create_sample_search_results())
        mock_ai_gen.return_value = Mock(generate_response=Mock(return_value="Course-specific response"))
        mock_session_mgr.return_value = MockSessionManager()
        mock_doc_proc.return_value = Mock()
        
        rag = RAGSystem(self.mock_config)
        
        # Test course-specific query
        response, sources = rag.query("Tell me about variables in the Python course")
        
        # Verify the prompt mentions course materials
        ai_call = rag.ai_generator.generate_response.call_args
        prompt = ai_call[1]['query']
        self.assertIn("course materials", prompt)
        self.assertIn("variables in the Python course", prompt)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_general_knowledge_query(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc):
        """Test that system handles general knowledge queries appropriately"""
        # Setup mocks
        mock_vector_store.return_value = MockVectorStore()
        mock_ai_gen.return_value = Mock(generate_response=Mock(return_value="General knowledge response"))
        mock_session_mgr.return_value = MockSessionManager()
        mock_doc_proc.return_value = Mock()
        
        rag = RAGSystem(self.mock_config)
        
        # Test general query that shouldn't need course search
        response, sources = rag.query("What is the weather like today?")
        
        # System should still pass tools to AI, but AI should decide not to use them
        ai_call = rag.ai_generator.generate_response.call_args
        self.assertIsNotNone(ai_call[1]['tools'])
        self.assertIsNotNone(ai_call[1]['tool_manager'])
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_failed_scenario(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc):
        """Test scenario that could lead to 'query failed' response"""
        # Setup mocks to simulate failure conditions
        error_vector_store = MockVectorStore(should_raise=True)
        mock_vector_store.return_value = error_vector_store
        
        # AI generator returns error response
        mock_ai_gen.return_value = Mock(generate_response=Mock(return_value="query failed"))
        mock_session_mgr.return_value = MockSessionManager()
        mock_doc_proc.return_value = Mock()
        
        rag = RAGSystem(self.mock_config)
        
        # This should simulate the conditions that cause "query failed"
        response, sources = rag.query("Find course content about Python")
        
        # Should get the error response
        self.assertEqual(response, "query failed")


class TestRAGSystemDocumentProcessing(unittest.TestCase):
    """Test RAG system document processing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = Mock()
        self.mock_config.CHUNK_SIZE = 800
        self.mock_config.CHUNK_OVERLAP = 100
        self.mock_config.CHROMA_PATH = "/tmp/test_chroma"
        self.mock_config.EMBEDDING_MODEL = "test-model"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test_key"
        self.mock_config.ANTHROPIC_MODEL = "test_model"
        self.mock_config.MAX_HISTORY = 2
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_document_success(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc):
        """Test successful course document addition"""
        # Setup mocks
        mock_course = create_sample_course()
        mock_chunks = create_sample_course_chunks()
        
        mock_doc_proc_instance = Mock()
        mock_doc_proc_instance.process_course_document.return_value = (mock_course, mock_chunks)
        mock_doc_proc.return_value = mock_doc_proc_instance
        
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        mock_ai_gen.return_value = Mock()
        mock_session_mgr.return_value = Mock()
        
        rag = RAGSystem(self.mock_config)
        
        # Add course document
        course, chunk_count = rag.add_course_document("/path/to/course.txt")
        
        # Verify processing occurred
        mock_doc_proc_instance.process_course_document.assert_called_once_with("/path/to/course.txt")
        mock_vector_store_instance.add_course_metadata.assert_called_once_with(mock_course)
        mock_vector_store_instance.add_course_content.assert_called_once_with(mock_chunks)
        
        # Verify return values
        self.assertEqual(course, mock_course)
        self.assertEqual(chunk_count, len(mock_chunks))
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_document_error(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc):
        """Test course document addition with error"""
        # Setup mocks
        mock_doc_proc_instance = Mock()
        mock_doc_proc_instance.process_course_document.side_effect = Exception("Processing error")
        mock_doc_proc.return_value = mock_doc_proc_instance
        
        mock_vector_store.return_value = Mock()
        mock_ai_gen.return_value = Mock()
        mock_session_mgr.return_value = Mock()
        
        rag = RAGSystem(self.mock_config)
        
        # Add course document with error
        course, chunk_count = rag.add_course_document("/path/to/bad_course.txt")
        
        # Should handle error gracefully
        self.assertIsNone(course)
        self.assertEqual(chunk_count, 0)


if __name__ == '__main__':
    unittest.main()