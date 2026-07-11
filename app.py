from src.config import load_settings
from src.llm import build_provider
from src.agent import ResearchHelperAgent
from src.memory import TemporaryMemory
from src.tools import TOOLS


def build_llm(settings):
    """
    Create the LLM provider from config.

    Provider selection (Hugging Face vs OpenAI) lives entirely in the factory;
    this is the only wiring app.py needs and it contains no provider-specific
    code.
    """

    return build_provider(settings)


def build_memory(settings):
    """
    Creates the temporary memory object.

    Keeping this separate makes it very easy later to
    replace TemporaryMemory with a MEM1 implementation,
    a LangGraph state, or AdaMem.
    """

    return TemporaryMemory(
        summary_every_n_turns=settings.summary_every_n_turns,
        recent_window_turns=settings.recent_window_turns,
    )


def build_agent(llm, memory, settings):
    """
    Creates the complete Research Helper Agent.
    """

    return ResearchHelperAgent(
        llm=llm,
        tools=TOOLS,
        memory=memory,
        max_tool_steps=settings.max_tool_steps,
    )


def main():

    # ----------------------------
    # Load configuration
    # ----------------------------
    settings = load_settings()

    # ----------------------------
    # Build application components
    # ----------------------------
    llm = build_llm(settings)

    memory = build_memory(settings)

    agent = build_agent(
        llm=llm,
        memory=memory,
        settings=settings,
    )

    print("=" * 60)
    print("Research Helper Agent")
    print(f"Provider: {settings.llm_provider} | Model: {settings.model_id}")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    while True:

        user_input = input("You : ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("\nGoodbye.")
            break

        answer = agent.run_turn(user_input)

        # Useful for learning.
        # Later we can replace this with structured logging.
        memory.print_state()

        print(f"\nAssistant : {answer}\n")


if __name__ == "__main__":
    main()
