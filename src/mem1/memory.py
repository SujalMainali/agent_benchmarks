"""
MEM1 Think Memory - Manages <think> state consolidation.

The core insight of MEM1 is memory consolidation rather than accumulation.
The model rewrites a compact internal state each turn containing only
information needed for future reasoning.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedResponse:
    """Parsed MEM1 model response."""
    think: Optional[str] = None
    search: Optional[str] = None
    information: Optional[str] = None
    answer: Optional[str] = None
    raw: str = ""


class Mem1ThinkMemory:
    """
    Manages MEM1's <think> internal state.
    
    The <think> block contains:
    - Important facts discovered so far
    - Intermediate reasoning
    - Current progress toward objectives
    - Plans for future actions
    - Extracted information from searches
    
    The state is REWRITTEN (not appended) each turn.
    """
    
    def __init__(self, max_think_chars: int = 1000):
        self.max_think_chars = max_think_chars
        self._current_state: str = ""
        self._history: list[str] = []
    
    @property
    def current_state(self) -> str:
        """Get current consolidated <think> state."""
        return self._current_state
    
    @property
    def history(self) -> list[str]:
        """Get history of all <think> states (for debugging/logging)."""
        return self._history.copy()
    
    def update_state(self, new_think: str) -> None:
        """
        Update internal state with new consolidated <think> content.
        
        This replaces the previous state entirely - MEM1's key insight
        is consolidation, not accumulation.
        """
        if self._current_state:
            self._history.append(self._current_state)
        
        # Truncate if necessary (model should handle this, but safety check)
        if len(new_think) > self.max_think_chars:
            new_think = new_think[:self.max_think_chars] + "..."
        
        self._current_state = new_think
    
    def reset(self) -> None:
        """Clear all state for a new episode."""
        self._current_state = ""
        self._history = []
    
    def get_state_for_prompt(self) -> str:
        """Get <think> block formatted for prompt injection."""
        if not self._current_state:
            return ""
        return f"<think>\n{self._current_state}\n</think>"
    
    @staticmethod
    def extract_tag(text: str, tag: str) -> Optional[str]:
        """Extract content from a specific XML-style tag."""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    @staticmethod
    def parse_response(response: str) -> ParsedResponse:
        """
        Parse MEM1 model response to extract all tags.
        
        Expected formats:
        - <think>state</think><search>query</search>
        - <think>state</think><answer>final answer</answer>
        """
        return ParsedResponse(
            think=Mem1ThinkMemory.extract_tag(response, "think"),
            search=Mem1ThinkMemory.extract_tag(response, "search"),
            information=Mem1ThinkMemory.extract_tag(response, "information"),
            answer=Mem1ThinkMemory.extract_tag(response, "answer"),
            raw=response,
        )
    
    def process_response(self, response: str) -> ParsedResponse:
        """
        Parse response and update internal state.
        
        Returns parsed response with all extracted components.
        """
        parsed = self.parse_response(response)
        
        if parsed.think:
            self.update_state(parsed.think)
        
        return parsed