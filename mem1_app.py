#!/usr/bin/env python3
"""
MEM1 Interactive Chat Application.

A simple REPL for interacting with the MEM1 agent.

Usage:
    python mem1_app.py

Commands:
    /state  - Show current <think> state
    /steps  - Show reasoning steps from last response
    /reset  - Reset agent state
    /help   - Show help
    /quit   - Exit
"""

from src.mem1.agent import Mem1Agent
from src.mem1.config import load_mem1_settings


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("  MEM1 Agent - Memory Consolidation Chat")
    print("  Model: Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release")
    print("=" * 60)
    print("Commands: /state /steps /reset /help /quit")
    print()


def print_help():
    """Print help message."""
    print("""
Available commands:
    /state  - Show current <think> memory state
    /steps  - Show reasoning steps from last response
    /reset  - Reset agent memory
    /help   - Show this help message
    /quit   - Exit the application
    
Just type your question to chat with the agent.
""")


def main():
    """Main REPL loop."""
    print_banner()
    
    # Load settings and create agent
    print("Loading MEM1 agent...")
    try:
        settings = load_mem1_settings()
        agent = Mem1Agent(settings=settings)
        print(f"✓ Agent loaded (model: {settings.model_id})")
        print(f"✓ Max context: {settings.max_context_chars} chars")
        print(f"✓ Retriever: {settings.retriever_type}")
    except Exception as e:
        print(f"✗ Failed to load agent: {e}")
        return
    
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break
        
        elif user_input.lower() == "/help":
            print_help()
            continue
        
        elif user_input.lower() == "/state":
            state = agent.get_current_state()
            if state:
                print(f"\n<think>\n{state}\n</think>\n")
            else:
                print("\n[No state yet]\n")
            continue
        
        elif user_input.lower() == "/steps":
            steps = agent.get_reasoning_steps()
            if steps:
                print(f"\n[{len(steps)} reasoning steps]")
                for step in steps:
                    print(f"\n--- Step {step.step_num} ---")
                    if step.parsed.think:
                        print(f"<think>{step.parsed.think[:200]}...</think>")
                    if step.parsed.search:
                        print(f"<search>{step.parsed.search}</search>")
                    if step.parsed.answer:
                        print(f"<answer>{step.parsed.answer}</answer>")
                print()
            else:
                print("\n[No steps yet]\n")
            continue
        
        elif user_input.lower() == "/reset":
            agent.reset()
            print("[Agent state reset]\n")
            continue
        
        # Process as question
        print("\nThinking...")
        try:
            result = agent.chat(user_input)
            print(f"\nMEM1: {result.answer}")
            print(f"      [{len(result.steps)} steps, think: {len(result.final_think)} chars]\n")
        except Exception as e:
            print(f"\n[Error: {e}]\n")


if __name__ == "__main__":
    main()