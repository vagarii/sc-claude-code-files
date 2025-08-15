"""Tests for CourseSearchTool.execute() method"""

import unittest
import sys
import os

# Add the backend directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from test_mocks import MockVectorStore, create_sample_search_results


class TestCourseSearchTool(unittest.TestCase):
    """Test cases for CourseSearchTool.execute() method"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = MockVectorStore()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_execute_successful_search(self):
        """Test successful search with results"""
        # Setup mock to return sample results
        sample_results = create_sample_search_results()
        self.mock_vector_store.search_results = sample_results
        
        # Execute search
        result = self.search_tool.execute("Python basics")
        
        # Verify the search was called with correct parameters
        self.assertEqual(len(self.mock_vector_store.search_calls), 1)
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call['query'], "Python basics")
        self.assertIsNone(call['course_name'])
        self.assertIsNone(call['lesson_number'])
        
        # Verify the result is properly formatted
        self.assertIn("Introduction to Python", result)
        self.assertIn("Lesson 1", result)
        self.assertIn("Python basics", result)
        self.assertNotEqual(result, "")
    
    def test_execute_with_course_filter(self):
        """Test search with course name filter"""
        sample_results = create_sample_search_results()
        self.mock_vector_store.search_results = sample_results
        
        result = self.search_tool.execute("variables", course_name="Python Course")
        
        # Verify course filter was passed
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call['course_name'], "Python Course")
        self.assertIn("Introduction to Python", result)
    
    def test_execute_with_lesson_filter(self):
        """Test search with lesson number filter"""
        sample_results = create_sample_search_results()
        self.mock_vector_store.search_results = sample_results
        
        result = self.search_tool.execute("basics", lesson_number=1)
        
        # Verify lesson filter was passed
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call['lesson_number'], 1)
        self.assertIn("Lesson 1", result)
    
    def test_execute_with_both_filters(self):
        """Test search with both course and lesson filters"""
        sample_results = create_sample_search_results()
        self.mock_vector_store.search_results = sample_results
        
        result = self.search_tool.execute("data types", course_name="Python", lesson_number=2)
        
        # Verify both filters were passed
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call['course_name'], "Python")
        self.assertEqual(call['lesson_number'], 2)
        self.assertIn("data types", result)
    
    def test_execute_empty_results(self):
        """Test search that returns no results"""
        empty_results = create_sample_search_results(empty=True)
        self.mock_vector_store.search_results = empty_results
        
        result = self.search_tool.execute("nonexistent topic")
        
        # Should return "no content found" message
        self.assertIn("No relevant content found", result)
    
    def test_execute_empty_results_with_filters(self):
        """Test search with filters that returns no results"""
        empty_results = create_sample_search_results(empty=True)
        self.mock_vector_store.search_results = empty_results
        
        result = self.search_tool.execute("topic", course_name="Missing Course", lesson_number=999)
        
        # Should include filter information in no results message
        self.assertIn("No relevant content found", result)
        self.assertIn("Missing Course", result)
        self.assertIn("lesson 999", result)
    
    def test_execute_search_error(self):
        """Test search that returns an error"""
        error_results = create_sample_search_results(with_error=True)
        self.mock_vector_store.search_results = error_results
        
        result = self.search_tool.execute("test query")
        
        # Should return the error message
        self.assertEqual(result, "Search error occurred")
    
    def test_execute_vector_store_exception(self):
        """Test when vector store raises an exception"""
        self.mock_vector_store.should_raise = True
        
        # This should handle the exception gracefully
        result = self.search_tool.execute("test query")
        
        # The search method should handle exceptions and return SearchResults.empty() with error
        # But since our mock raises an exception, we need to see how the real code handles it
        # Based on vector_store.py:100, exceptions are caught and return SearchResults.empty()
        self.assertTrue(isinstance(result, str))
    
    def test_execute_sources_tracking(self):
        """Test that sources are properly tracked"""
        sample_results = create_sample_search_results()
        self.mock_vector_store.search_results = sample_results
        
        # Execute search
        self.search_tool.execute("Python basics")
        
        # Check that sources were stored
        self.assertIsInstance(self.search_tool.last_sources, list)
        self.assertEqual(len(self.search_tool.last_sources), 2)  # Two results in sample
        
        # Check source structure
        source = self.search_tool.last_sources[0]
        self.assertIn("text", source)
        self.assertIn("link", source)
        self.assertIn("Introduction to Python", source["text"])
    
    def test_format_results_with_lesson_links(self):
        """Test that lesson links are properly included in results"""
        sample_results = create_sample_search_results()
        self.mock_vector_store.search_results = sample_results
        
        result = self.search_tool.execute("test")
        
        # Should have called get_lesson_link for each result
        # Verify result formatting includes lesson information
        self.assertIn("[Introduction to Python - Lesson 1]", result)
        self.assertIn("[Introduction to Python - Lesson 2]", result)
    
    def test_tool_definition(self):
        """Test that tool definition is correctly structured"""
        definition = self.search_tool.get_tool_definition()
        
        # Verify required fields
        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)
        
        # Verify schema structure
        schema = definition["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertEqual(schema["required"], ["query"])
        
        # Verify properties
        props = schema["properties"]
        self.assertIn("query", props)
        self.assertIn("course_name", props)
        self.assertIn("lesson_number", props)


class TestCourseSearchToolIntegration(unittest.TestCase):
    """Integration tests that test CourseSearchTool with more realistic conditions"""
    
    def test_search_parameter_variations(self):
        """Test various parameter combinations"""
        mock_store = MockVectorStore(create_sample_search_results())
        tool = CourseSearchTool(mock_store)
        
        # Test cases with different parameter combinations
        test_cases = [
            {"query": "basics"},
            {"query": "variables", "course_name": "Python"},
            {"query": "functions", "lesson_number": 3},
            {"query": "classes", "course_name": "Python", "lesson_number": 5}
        ]
        
        for i, params in enumerate(test_cases):
            with self.subTest(case=i):
                result = tool.execute(**params)
                
                # Should not be empty and should contain course info
                self.assertNotEqual(result, "")
                self.assertIn("Introduction to Python", result)
                
                # Verify parameters were passed correctly to vector store
                call = mock_store.search_calls[i]
                self.assertEqual(call['query'], params['query'])
                self.assertEqual(call.get('course_name'), params.get('course_name'))
                self.assertEqual(call.get('lesson_number'), params.get('lesson_number'))


if __name__ == '__main__':
    unittest.main()