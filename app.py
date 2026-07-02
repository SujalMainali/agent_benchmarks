from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)

from src.agent import ResearchHelperAgent
from src.config import load_settings
from src.memory import TemporaryMemory
from src.tools import TOOLS


def build_chat_model(settings):
    """
    Creates the underlying LLM.

    This is the only place in the project that knows
    which model/provider we are using.
    """

    llm = HuggingFaceEndpoint(
        repo_id=settings.model_id,
        task="text-generation",
        huggingfacehub_api_token=settings.hf_token,
        provider=settings.provider,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        do_sample=settings.do_sample,
    )

    return ChatHuggingFace(llm=llm)


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


def build_agent(chat_model, memory, settings):
    """
    Creates the complete Research Helper Agent.
    """

    return ResearchHelperAgent(
        chat_model=chat_model,
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
    chat_model = build_chat_model(settings)

    memory = build_memory(settings)

    agent = build_agent(
        chat_model=chat_model,
        memory=memory,
        settings=settings,
    )

    print("=" * 60)
    print("Research Helper Agent")
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