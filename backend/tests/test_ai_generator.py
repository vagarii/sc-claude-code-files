"""Tests for AIGenerator to verify CourseSearchTool integration"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add the backend directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from test_mocks import MockAnthropicClient, MockToolManager


class TestAIGenerator(unittest.TestCase):
    """Test cases for AIGenerator CourseSearchTool integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        # We'll mock the anthropic client to avoid real API calls
        self.api_key = "test_api_key"
        self.model = "claude-sonnet-4-20250514"
        
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test basic response generation without tools"""
        # Setup mock client
        mock_client = MockAnthropicClient("This is a basic response")
        mock_anthropic.return_value.messages = mock_client
        
        ai_gen = AIGenerator(self.api_key, self.model)
        ai_gen.client.messages = mock_client
        
        # Generate response without tools
        response = ai_gen.generate_response("What is Python?")
        
        # Verify response
        self.assertEqual(response, "This is a basic response")
        self.assertEqual(len(mock_client.messages_calls), 1)
        
        # Verify API call parameters
        call = mock_client.messages_calls[0]
        self.assertEqual(call['model'], self.model)
        self.assertEqual(call['temperature'], 0)
        self.assertEqual(call['max_tokens'], 800)
        self.assertEqual(len(call['messages']), 1)
        self.assertEqual(call['messages'][0]['content'], "What is Python?")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation with conversation history"""
        mock_client = MockAnthropicClient("Response with history")
        mock_anthropic.return_value.messages = mock_client
        
        ai_gen = AIGenerator(self.api_key, self.model)
        ai_gen.client.messages = mock_client
        
        # Generate response with history
        history = "User: Hello\nAssistant: Hi there!"
        response = ai_gen.generate_response("What is Python?", conversation_history=history)
        
        # Verify history was included in system prompt
        call = mock_client.messages_calls[0]
        self.assertIn(history, call['system'])
        self.assertIn("Previous conversation:", call['system'])
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic):
        """Test response generation with tools available but not used"""
        mock_client = MockAnthropicClient("Direct response without tools")
        mock_anthropic.return_value.messages = mock_client
        
        ai_gen = AIGenerator(self.api_key, self.model)
        ai_gen.client.messages = mock_client
        
        # Create mock tool manager
        mock_tool_manager = MockToolManager()
        tools = mock_tool_manager.get_tool_definitions()
        
        # Generate response
        response = ai_gen.generate_response(
            "What is the capital of France?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify tools were passed to API but not used
        call = mock_client.messages_calls[0]
        self.assertIn('tools', call)
        self.assertEqual(call['tool_choice'], {"type": "auto"})
        
        # Since mock doesn't use tools, should get direct response
        self.assertEqual(response, "Direct response without tools")
        self.assertEqual(len(mock_tool_manager.execute_calls), 0)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_execution(self, mock_anthropic):
        """Test response generation that uses CourseSearchTool"""
        # Setup mock client to simulate tool use
        mock_client = MockAnthropicClient("Final response after tool use", should_use_tool=True)
        mock_anthropic.return_value.messages = mock_client
        
        ai_gen = AIGenerator(self.api_key, self.model)
        ai_gen.client.messages = mock_client
        
        # Create mock tool manager
        mock_tool_manager = MockToolManager("Search results: Python is a programming language")
        tools = mock_tool_manager.get_tool_definitions()
        
        # Generate response that should trigger tool use
        response = ai_gen.generate_response(
            "Tell me about Python basics from the course",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed
        self.assertEqual(len(mock_tool_manager.execute_calls), 1)
        tool_call = mock_tool_manager.execute_calls[0]
        self.assertEqual(tool_call['tool_name'], 'search_course_content')
        self.assertEqual(tool_call['kwargs'], {"query": "test query"})
        
        # Verify multiple API calls were made (initial + follow-up)
        self.assertEqual(len(mock_client.messages_calls), 2)
        
        # Verify final response
        self.assertEqual(response, "Final response after tool use")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_error(self, mock_anthropic):
        """Test handling of tool execution errors"""
        mock_client = MockAnthropicClient("Error response", should_use_tool=True)
        mock_anthropic.return_value.messages = mock_client
        
        ai_gen = AIGenerator(self.api_key, self.model)
        ai_gen.client.messages = mock_client
        
        # Create mock tool manager that raises errors
        mock_tool_manager = MockToolManager(should_raise=True)
        tools = mock_tool_manager.get_tool_definitions()
        
        # This should handle the tool execution error gracefully
        try:
            response = ai_gen.generate_response(
                "Search for course content",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            # If it doesn't raise, verify the error was handled
            self.assertTrue(True)  # Test passes if no exception
        except Exception as e:
            # The _handle_tool_execution method should catch tool errors
            # and still proceed with API call
            self.fail(f"Tool execution error was not handled: {e}")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic):
        """Test handling of Anthropic API errors"""
        mock_client = MockAnthropicClient(should_raise=True)
        mock_anthropic.return_value.messages = mock_client
        
        ai_gen = AIGenerator(self.api_key, self.model)
        ai_gen.client.messages = mock_client
        
        # This should raise the API error (not caught in AIGenerator)
        with self.assertRaises(Exception) as context:
            ai_gen.generate_response("Test query")
        
        self.assertIn("Mock API error", str(context.exception))
    
    def test_system_prompt_content(self):
        """Test that system prompt includes tool usage guidelines"""
        ai_gen = AIGenerator(self.api_key, self.model)
        
        # Check that system prompt contains important guidance
        system_prompt = ai_gen.SYSTEM_PROMPT
        
        # Should mention course search tools
        self.assertIn("Course Content Search", system_prompt)
        self.assertIn("Course Outline Tool", system_prompt)
        
        # Should contain tool usage guidelines
        self.assertIn("Tool Usage Guidelines", system_prompt)
        self.assertIn("search_course_content", system_prompt)
        
        # Should contain response protocol
        self.assertIn("Response Protocol", system_prompt)
        self.assertIn("Course-specific questions", system_prompt)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_response_integration(self, mock_anthropic):
        """Test that tool responses are properly integrated into conversation"""
        mock_client = MockAnthropicClient("Response based on search results", should_use_tool=True)
        mock_anthropic.return_value.messages = mock_client
        
        ai_gen = AIGenerator(self.api_key, self.model)
        ai_gen.client.messages = mock_client
        
        # Mock tool manager with specific search results
        search_results = "Course Python Basics Lesson 1: Variables are containers for data"
        mock_tool_manager = MockToolManager(search_results)
        tools = mock_tool_manager.get_tool_definitions()
        
        response = ai_gen.generate_response(
            "What are variables in Python?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify that the follow-up call includes tool results
        self.assertEqual(len(mock_client.messages_calls), 2)
        
        # Check the follow-up call structure
        follow_up_call = mock_client.messages_calls[1]
        messages = follow_up_call['messages']
        
        # Should have: original user message, assistant tool use, user tool results
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[1]['role'], 'assistant')
        self.assertEqual(messages[2]['role'], 'user')
        
        # Tool results should be in the final user message
        tool_result_message = messages[2]
        self.assertIn('content', tool_result_message)
        self.assertIsInstance(tool_result_message['content'], list)


class TestAIGeneratorCourseSearchIntegration(unittest.TestCase):
    """Integration tests for AIGenerator with CourseSearchTool"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_course_search_tool_parameters(self, mock_anthropic):
        """Test that CourseSearchTool receives correct parameters"""
        mock_client = MockAnthropicClient("Search results", should_use_tool=True)
        mock_anthropic.return_value.messages = mock_client
        
        ai_gen = AIGenerator("test_key", "test_model")
        ai_gen.client.messages = mock_client
        
        # Create a more specific mock for CourseSearchTool
        mock_tool_manager = MockToolManager()
        
        # Override the tool use to simulate specific search parameters
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {
            "query": "Python variables",
            "course_name": "Introduction to Python",
            "lesson_number": 2
        }
        mock_tool_block.id = "tool_123"
        
        # First call returns tool use, second call returns final text
        tool_response = Mock(stop_reason="tool_use", content=[mock_tool_block])
        text_response = Mock(stop_reason="end_turn", content=[Mock(text="Search results")])
        mock_client.create = Mock(side_effect=[tool_response, text_response])
        
        response = ai_gen.generate_response(
            "Tell me about variables in Python lesson 2",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify CourseSearchTool was called with correct parameters
        self.assertEqual(len(mock_tool_manager.execute_calls), 1)
        call = mock_tool_manager.execute_calls[0]
        
        self.assertEqual(call['tool_name'], 'search_course_content')
        self.assertEqual(call['kwargs']['query'], 'Python variables')
        self.assertEqual(call['kwargs']['course_name'], 'Introduction to Python')
        self.assertEqual(call['kwargs']['lesson_number'], 2)


class TestSequentialToolCalling(unittest.TestCase):
    """Test cases for sequential tool calling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.model = "claude-sonnet-4-20250514"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_two_round_sequential_tool_calling(self, mock_anthropic):
        """Test successful 2-round sequential tool calling"""
        # Setup mock client to simulate round 1 tool use, then round 2 final response
        mock_client = Mock()
        mock_anthropic.return_value.messages = mock_client
        
        # Round 1: Tool use response
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "get_course_outline"
        mock_tool_block.input = {"course_title": "Python Course"}
        mock_tool_block.id = "tool_123"
        round1_response.content = [mock_tool_block]
        
        # Round 2: Final response after tool results
        round2_response = Mock()
        round2_response.stop_reason = "end_turn"
        mock_text_block = Mock()
        mock_text_block.text = "Based on the course outline, lesson 4 covers advanced functions."
        round2_response.content = [mock_text_block]
        
        # Setup mock client to return different responses per call
        mock_client.create = Mock(side_effect=[round1_response, round2_response])
        
        ai_gen = AIGenerator(self.api_key, self.model)
        ai_gen.client.messages = mock_client
        
        # Create mock tool manager
        mock_tool_manager = MockToolManager("Course outline: Lesson 1: Basics, Lesson 2: Variables...")
        tools = mock_tool_manager.get_tool_definitions()
        
        # Execute query that should trigger 2 rounds
        response = ai_gen.generate_response(
            "What topic does lesson 4 cover in the Python course?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify 2 API calls were made
        self.assertEqual(len(mock_client.create.call_args_list), 2)
        
        # Verify tool was executed
        self.assertEqual(len(mock_tool_manager.execute_calls), 1)
        
        # Verify final response
        self.assertEqual(response, "Based on the course outline, lesson 4 covers advanced functions.")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_single_round_completion(self, mock_anthropic):
        """Test completion after single round (no tool use)"""
        mock_client = MockAnthropicClient("This is a direct answer without tools")
        mock_anthropic.return_value.messages = mock_client
        
        ai_gen = AIGenerator(self.api_key, self.model)
        ai_gen.client.messages = mock_client
        
        mock_tool_manager = MockToolManager()
        tools = mock_tool_manager.get_tool_definitions()
        
        response = ai_gen.generate_response(
            "What is the capital of France?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Should only make 1 API call
        self.assertEqual(len(mock_client.messages_calls), 1)
        
        # No tools should be executed
        self.assertEqual(len(mock_tool_manager.execute_calls), 0)
        
        # Should get direct response
        self.assertEqual(response, "This is a direct answer without tools")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_max_rounds_termination(self, mock_anthropic):
        """Test that system terminates after 2 rounds even if Claude wants more tools"""
        mock_client = Mock()
        mock_anthropic.return_value.messages = mock_client
        
        # Both rounds return tool use responses
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_block.id = "tool_123"
        tool_response.content = [mock_tool_block]
        
        # Final response after round 2 tool execution
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        mock_text_block = Mock()
        mock_text_block.text = "Final response after 2 rounds"
        final_response.content = [mock_text_block]
        
        # Setup responses: round1 tool, round1 follow-up tool, round2 tool, then final
        mock_client.create = Mock(side_effect=[tool_response, tool_response, tool_response, final_response])
        
        ai_gen = AIGenerator(self.api_key, self.model)
        ai_gen.client.messages = mock_client
        
        mock_tool_manager = MockToolManager("Tool result")
        tools = mock_tool_manager.get_tool_definitions()
        
        response = ai_gen.generate_response(
            "Complex query needing multiple searches",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Should make exactly 4 calls: round1 initial, round1 follow-up, round2 initial, round2 final
        self.assertEqual(len(mock_client.create.call_args_list), 4)
        
        # Should execute 2 tools (one per round)
        self.assertEqual(len(mock_tool_manager.execute_calls), 2)
        
        # Should get final response
        self.assertEqual(response, "Final response after 2 rounds")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic):
        """Test graceful handling of tool execution errors"""
        # Round 1: Tool use
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_block.id = "tool_123"
        round1_response.content = [mock_tool_block]
        
        # Final response after error
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        mock_text_block = Mock()
        mock_text_block.text = "I encountered an issue with the search, but here's what I can tell you..."
        final_response.content = [mock_text_block]
        
        mock_client = Mock()
        mock_client.create = Mock(side_effect=[round1_response, final_response])
        
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value.messages = mock_client
            
            ai_gen = AIGenerator(self.api_key, self.model)
            ai_gen.client.messages = mock_client
            
            # Tool manager that raises errors
            mock_tool_manager = MockToolManager(should_raise=True)
            tools = mock_tool_manager.get_tool_definitions()
            
            response = ai_gen.generate_response(
                "Search for course content",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            # Should still get a response despite tool error
            self.assertIsNotNone(response)
            self.assertIn("encountered an issue", response)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_round_specific_system_prompts(self, mock_anthropic):
        """Test that system prompts are modified appropriately for each round"""
        mock_client = Mock()
        mock_anthropic.return_value.messages = mock_client
        
        # Round 1: Tool use
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_block.id = "tool_123"
        round1_response.content = [mock_tool_block]
        
        # Round 1 follow-up: Tool use again (to trigger round 2)
        round1_followup_response = Mock()
        round1_followup_response.stop_reason = "tool_use"
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "search_course_content"
        mock_tool_block2.input = {"query": "test2"}
        mock_tool_block2.id = "tool_456"
        round1_followup_response.content = [mock_tool_block2]
        
        # Round 2: Final response
        round2_response = Mock()
        round2_response.stop_reason = "end_turn"
        mock_text_block = Mock()
        mock_text_block.text = "Final answer"
        round2_response.content = [mock_text_block]
        
        mock_client.create = Mock(side_effect=[round1_response, round1_followup_response, round2_response])
        
        ai_gen = AIGenerator(self.api_key, self.model)
        ai_gen.client.messages = mock_client
        
        mock_tool_manager = MockToolManager("Tool result")
        tools = mock_tool_manager.get_tool_definitions()
        
        ai_gen.generate_response(
            "Test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Check that system prompts were different for each round
        calls = mock_client.create.call_args_list
        self.assertEqual(len(calls), 3)
        
        round1_system = calls[0][1]['system']
        round1_followup_system = calls[1][1]['system']
        round2_system = calls[2][1]['system']
        
        # Round 1 should mention multi-step reasoning
        self.assertIn("multi-step reasoning", round1_system)
        self.assertIn("multi-step reasoning", round1_followup_system)
        
        # Round 2 should mention final round
        self.assertIn("final round", round2_system)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_conversation_context_preservation(self, mock_anthropic):
        """Test that conversation context is preserved between rounds"""
        mock_client = Mock()
        mock_anthropic.return_value.messages = mock_client
        
        # Round 1: Tool use
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "Python basics"}
        mock_tool_block.id = "tool_123"
        round1_response.content = [mock_tool_block]
        
        # Round 2: Final response
        round2_response = Mock()
        round2_response.stop_reason = "end_turn"
        mock_text_block = Mock()
        mock_text_block.text = "Based on the search results, here's what I found..."
        round2_response.content = [mock_text_block]
        
        mock_client.create = Mock(side_effect=[round1_response, round2_response])
        
        ai_gen = AIGenerator(self.api_key, self.model)
        ai_gen.client.messages = mock_client
        
        mock_tool_manager = MockToolManager("Python is a programming language...")
        tools = mock_tool_manager.get_tool_definitions()
        
        response = ai_gen.generate_response(
            "Tell me about Python basics",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify round 2 includes full conversation history
        round2_call = mock_client.create.call_args_list[1]
        round2_messages = round2_call[1]['messages']
        
        # Should have: user query, assistant tool use, user tool results
        self.assertEqual(len(round2_messages), 3)
        self.assertEqual(round2_messages[0]['role'], 'user')  # Original query
        self.assertEqual(round2_messages[1]['role'], 'assistant')  # Tool use
        self.assertEqual(round2_messages[2]['role'], 'user')  # Tool results


if __name__ == '__main__':
    unittest.main()