# Real-Time Translation Agents

This project implements two different approaches to real-time translation using LLM-powered agents: a deterministic graph-based agent (RT_Graph) and a non-deterministic loop-based agent (RT_Graph_v2).

## Agents Overview

### Deterministic Agent (RT_Graph)
The first implementation uses a deterministic state graph approach with predefined edges and nodes:

- **State Management**: Uses `RT_State` with fixed fields for selected elements, context, and message history
- **Flow Control**: Implements a strict graph structure with predefined paths:
  - START → set_selected_element → decide_if_selected_is_message_box
  - Conditional branching based on message box detection
  - Translation or waiting states with fixed timeouts

### Non-Deterministic Agent (RT_Graph_v2)
The second implementation uses a more flexible, loop-based approach:

- **State Management**: Uses `RT_State_v2` with dynamic context tracking
- **Flow Control**: Implements a main loop that:
  - Continuously monitors active elements
  - Makes decisions based on current context
  - Can switch between states more fluidly:
    - translate
    - set_context_html
    - set_context_img
    - wait

## Key Learnings

1. **Context Gathering**
   - HTML context can be too large (>1000 chars) for non-relevant elements
   - Screenshot analysis helps verify text input fields
   - Multiple context sources improve decision accuracy

2. **State Management**
   - Keep message history limited (last 6 messages) for efficiency
   - Track last translation to prevent redundant operations
   - Store intermediate results for error recovery

3. **Error Handling**
   - Multiple retry attempts for JSON parsing
   - Graceful fallbacks for element selection
   - Timeout management for element visibility

4. **Performance Optimizations**
   - Async operations for I/O-bound tasks
   - Smart waiting periods (3-10 seconds) based on context
   - Message chain optimization to prevent memory bloat

## Determinitic Graph

![LangGraph](https://github.com/NawidT/rapid_translate/blob/main/graph.png)

## Future Improvements

1. Implement better element detection heuristics
2. Add support for more input types beyond text
3. Optimize translation caching
4. Add support for multiple languages simultaneously
5. Implement better error recovery mechanisms

## Dependencies

- langgraph
- langchain
- playwright
- OpenAI API (GPT-4)

[Add screenshots and specific testing results from testing.ipynb]
