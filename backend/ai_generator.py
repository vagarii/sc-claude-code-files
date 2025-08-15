import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import copy


@dataclass
class ConversationState:
    """Tracks state across multiple rounds of tool calling"""
    round_number: int = 1
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    user_query: str = ""
    system_prompt: str = ""
    is_complete: bool = False
    last_response: Optional[str] = None
    error_state: Optional[Dict[str, Any]] = None
    
    def create_next_round(self) -> 'ConversationState':
        """Create state for next round, preserving context"""
        return ConversationState(
            round_number=self.round_number + 1,
            message_history=copy.deepcopy(self.message_history),
            tool_results=copy.deepcopy(self.tool_results),
            user_query=self.user_query,
            system_prompt=self.system_prompt,
            is_complete=False,
            last_response=self.last_response,
            error_state=self.error_state
        )
    
    def has_reached_max_rounds(self, max_rounds: int = 2) -> bool:
        """Check if maximum rounds reached"""
        return self.round_number > max_rounds



class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **Course Content Search**: For questions about specific course content or detailed educational materials
2. **Course Outline Tool**: For questions about course structure, lesson lists, course links, or course overviews

Tool Usage Guidelines:
- **Course outline queries**: Use the course outline tool for questions about:
  - Course structure or organization
  - List of lessons in a course
  - Course links or overview information
  - "What lessons are in [course]?" or "Show me the outline of [course]"
- **Course content queries**: Use the content search tool for questions about:
  - Specific topics within lessons
  - Detailed explanations of concepts
  - Code examples or technical details
- **Tool names**: Use `search_course_content` for content search and `get_course_outline` for course outlines
- **One tool use per query maximum**
- When using course outline tool, return the complete course information including:
  - Course title
  - Course link (if available)
  - Total number of lessons
  - Complete lesson list with lesson numbers and titles
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def _build_round_specific_system_prompt(self, base_prompt: str, round_number: int) -> str:
        """Build system prompt with round-specific guidance"""
        if round_number == 1:
            round_guidance = """
You are in a multi-step reasoning session. After using tools, you may get another opportunity to use additional tools based on the results. Focus on gathering the information you need in this step. If you need to use tools sequentially (like getting course outline first, then searching for specific content), start with the first tool call."""
        else:  # round 2
            round_guidance = """
This is your final round. You have access to previous tool results in the conversation history. If you need additional information, you can make one more tool call, but after this response, provide a complete, comprehensive answer to the user's original question. No additional tool calls will be possible after this response."""
        
        return f"{base_prompt}\n\n{round_guidance}"
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 rounds of sequential tool calling.
        
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        
        # Initialize conversation state
        state = ConversationState(
            user_query=query,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        # Build initial system content
        base_system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )
        
        # Execute rounds with max 2 rounds
        max_rounds = 2
        
        while not state.is_complete and not state.has_reached_max_rounds(max_rounds):
            try:
                # Build round-specific system prompt
                round_system_prompt = self._build_round_specific_system_prompt(
                    base_system_content, state.round_number
                )
                
                # Execute single round
                response_text, state = self._execute_round(
                    state, round_system_prompt, tools, tool_manager
                )
                
                # Check termination conditions
                if state.is_complete or state.has_reached_max_rounds(max_rounds):
                    return response_text
                    
                # Prepare for next round
                state = state.create_next_round()
                
            except Exception as e:
                # For first round API errors, re-raise to maintain backward compatibility
                if state.round_number == 1 and state.last_response is None:
                    raise e
                
                # Handle errors gracefully for subsequent rounds
                error_msg = f"Error in round {state.round_number}: {str(e)}"
                if state.last_response:
                    return f"{state.last_response}\n\n[Note: {error_msg}]"
                return f"I encountered an error while processing your request: {error_msg}"
        
        # Return final response
        return state.last_response or "Unable to generate response."
    
    def _execute_round(self, state: ConversationState, system_prompt: str, 
                      tools: Optional[List], tool_manager) -> tuple[str, ConversationState]:
        """
        Execute a single round of conversation with Claude.
        
        Args:
            state: Current conversation state
            system_prompt: System prompt for this round
            tools: Available tools
            tool_manager: Tool execution manager
            
        Returns:
            Tuple of (response_text, updated_state)
        """
        # Build messages for this round
        if state.round_number == 1:
            # First round: start with user query
            messages = [{"role": "user", "content": state.user_query}]
        else:
            # Subsequent rounds: use accumulated message history
            messages = state.message_history
        
        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_prompt
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution_sequential(response, api_params, tool_manager, state)
        
        # No tool use - conversation complete
        response_text = response.content[0].text
        state.last_response = response_text
        state.is_complete = True
        
        return response_text, state
    
    def _handle_tool_execution_sequential(self, initial_response, base_params: Dict[str, Any], 
                                        tool_manager, state: ConversationState) -> tuple[str, ConversationState]:
        """
        Handle tool execution in sequential calling context.
        
        Args:
            initial_response: Claude's response containing tool calls
            base_params: Base API parameters
            tool_manager: Tool execution manager
            state: Current conversation state
            
        Returns:
            Tuple of (response_text, updated_state)
        """
        # Start with existing messages or create from base params
        messages = base_params["messages"].copy()
        
        # Add Claude's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        tool_execution_errors = []
        
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                    
                except Exception as e:
                    # Handle tool execution errors gracefully
                    error_result = f"Tool execution failed: {str(e)}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": error_result,
                        "is_error": True
                    })
                    tool_execution_errors.append(str(e))
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Update state with new message history
        state.message_history = messages
        state.tool_results.extend(tool_results)
        
        # If this is round 2 or we have tool errors, get final response without tools
        if state.round_number >= 2 or tool_execution_errors:
            # Final API call without tools for this round
            final_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"]
            }
            
            final_response = self.client.messages.create(**final_params)
            response_text = final_response.content[0].text
            state.last_response = response_text
            state.is_complete = True
            
            return response_text, state
        
        # Round 1 with successful tool execution - get Claude's response to tool results
        follow_up_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
            "tools": base_params.get("tools", []),  # Keep tools available
            "tool_choice": {"type": "auto"}
        }
        
        follow_up_response = self.client.messages.create(**follow_up_params)
        
        # Check if Claude wants to make another tool call
        if follow_up_response.stop_reason == "tool_use":
            # Claude wants to make another tool call - add this response to message history
            messages.append({"role": "assistant", "content": follow_up_response.content})
            state.message_history = messages
            state.last_response = "Preparing for additional tool call..."
            return state.last_response, state
        else:
            # Claude is done - return final response
            response_text = follow_up_response.content[0].text
            state.last_response = response_text
            state.is_complete = True
            return response_text, state
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Legacy method for backward compatibility.
        Handle execution of tool calls and get follow-up response (single round only).
        
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
