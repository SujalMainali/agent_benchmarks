"""
MEM1 Agent - Implements the MEM1 reasoning loop.

The agent:
1. Takes a question
2. Generates <think> state and optionally <search> queries
3. Retrieves information if <search> is present
4. Continues until <answer> is produced
"""

from dataclasses import dataclass, field
from typing import Optional
from huggingface_hub import InferenceClient

from src.mem1.config import Mem1Settings, load_mem1_settings
from src.mem1.memory import Mem1ThinkMemory, ParsedResponse
from src.mem1.retriever import Mem1Retriever


# System prompt to instruct Qwen to use MEM1-style tags
MEM1_SYSTEM_PROMPT = """You are a reasoning assistant that maintains a persistent memory using <think> tags.

For each interaction:
1. First, output <think>...</think> containing:
   - Important facts discovered so far
   - Your current reasoning
   - Progress toward the objective
   - Plans for next steps

2. If you need more information, output <search>your query</search>

3. When you have enough information to answer, output <answer>your final answer</answer>

Rules:
- ALWAYS start with <think> tags
- The <think> content should be a CUMULATIVE summary (not just this turn)
- End with either <search> OR <answer>, never both
- Keep <think> content concise but complete"""


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_num: int
    prompt: str
    response: str
    parsed: ParsedResponse
    information: Optional[str] = None


@dataclass
class AgentResult:
    """Result from agent reasoning."""
    answer: str
    final_think: str
    steps: list[ReasoningStep] = field(default_factory=list)
    total_tokens: int = 0


class Mem1Agent:
    """
    MEM1 Agent with <think> memory consolidation.
    
    Implements the MEM1 reasoning loop:
    1. Build prompt with question and previous <think> state
    2. Generate response
    3. Parse <think>, <search>, <answer> tags
    4. If <search>: retrieve information, inject <information>, continue
    5. If <answer>: return final answer
    6. Update <think> state each iteration
    """
    
    def __init__(
        self,
        settings: Optional[Mem1Settings] = None,
        retriever: Optional[Mem1Retriever] = None,
    ):
        self.settings = settings or load_mem1_settings()
        
        # Initialize HuggingFace client
        self.client = InferenceClient(
            model=self.settings.model_id,
            token=self.settings.hf_token,
        )
        
        # Initialize memory
        self.memory = Mem1ThinkMemory(
            max_think_chars=self.settings.max_think_chars
        )
        
        # Initialize retriever
        self.retriever = retriever or Mem1Retriever(
            retriever_type=self.settings.retriever_type,
            serper_api_key=self.settings.serper_api_key,
            top_k=self.settings.top_k_results,
            max_chars=self.settings.max_information_chars,
        )
        
        # Reasoning history for current question
        self._steps: list[ReasoningStep] = []
    
    def reset(self, corpus: list[str] = None) -> None:
        """Reset agent state for a new episode."""
        self.memory.reset()
        self._steps = []
        if corpus:
            self.retriever.set_corpus(corpus)
    
    def _build_initial_prompt(self, question: str) -> str:
        """Build the initial prompt for a question."""
        parts = [f"Question: {question}"]
        
        # Add previous think state if exists
        think_block = self.memory.get_state_for_prompt()
        if think_block:
            parts.append(f"Previous memory state:\n{think_block}")
        
        parts.append("Please reason through this and provide your response with <think>, and then <search> or <answer> tags.")
        
        return "\n\n".join(parts)
    
    def _build_continuation_prompt(
        self,
        previous_prompt: str,
        previous_response: str,
        information: str,
    ) -> str:
        """Build prompt for continuation after retrieval."""
        return f"{previous_prompt}\n\nYour previous response:\n{previous_response}\n\nSearch results:\n{information}\n\nContinue reasoning with updated <think>, then <search> or <answer>."
    
    def _generate(self, prompt: str) -> str:
        """Generate response from the model using chat completion."""
        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": MEM1_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.settings.max_new_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[DEBUG] Generation failed: {type(e).__name__}: {e}")
            return f"<think>Error: {e}</think><answer>Error: Unable to generate response</answer>"
    
    def _enforce_context_limit(self, prompt: str) -> str:
        """Ensure prompt doesn't exceed context limit."""
        if len(prompt) <= self.settings.max_context_chars:
            return prompt
        
        # Truncate from the beginning, keeping the question and recent context
        excess = len(prompt) - self.settings.max_context_chars
        return "..." + prompt[excess + 3:]
    
    def chat(self, question: str) -> AgentResult:
        """
        Process a question through the MEM1 reasoning loop.
        
        Returns:
            AgentResult with answer, final think state, and reasoning steps
        """
        self._steps = []
        prompt = self._build_initial_prompt(question)
        
        for step_num in range(self.settings.max_reasoning_steps):
            # Enforce context limit
            prompt = self._enforce_context_limit(prompt)
            
            # Generate response
            response = self._generate(prompt)
            
            # Parse response
            parsed = self.memory.process_response(response)
            
            # Create step record
            step = ReasoningStep(
                step_num=step_num + 1,
                prompt=prompt,
                response=response,
                parsed=parsed,
            )
            
            # Check if we have an answer
            if parsed.answer:
                self._steps.append(step)
                return AgentResult(
                    answer=parsed.answer,
                    final_think=self.memory.current_state,
                    steps=self._steps,
                )
            
            # Check if we need to search
            if parsed.search:
                information = self.retriever.search(parsed.search)
                step.information = information
                self._steps.append(step)
                
                # Build continuation prompt
                prompt = self._build_continuation_prompt(
                    prompt, response, information
                )
            else:
                # No search and no answer - force termination
                self._steps.append(step)
                return AgentResult(
                    answer=parsed.think or response or "No answer generated",
                    final_think=self.memory.current_state,
                    steps=self._steps,
                )
        
        # Max steps reached
        return AgentResult(
            answer=f"Max reasoning steps ({self.settings.max_reasoning_steps}) reached.",
            final_think=self.memory.current_state,
            steps=self._steps,
        )
    
    def get_current_state(self) -> str:
        """Get current <think> state for inspection."""
        return self.memory.current_state
    
    def get_reasoning_steps(self) -> list[ReasoningStep]:
        """Get reasoning steps from last chat."""
        return self._steps.copy()