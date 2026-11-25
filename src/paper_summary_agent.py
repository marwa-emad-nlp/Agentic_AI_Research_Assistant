from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel


class PaperSummaryAgent:
    """Summarize research papers from their supplied abstracts."""

    def __init__(self, model: BaseChatModel) -> None:
        self.graph = create_agent(
            model=model,
            tools=[],
            name="paper_summary_agent",
            system_prompt=(
                "You summarize research papers from their supplied abstracts. "
                "For each paper, keep the supplied title and source URL if present. "
                "Write a short abstract-based summary of the aim, approach, and "
                "findings when stated. Use only the supplied abstracts. "
                "Use search_papers tool results in the shared history as evidence "
                "when available. If no abstract is available, say so and stop."
            ),
        )

    def invoke(self, request: str) -> dict:
        """Run the summary agent on its own and return its message history."""
        return self.graph.invoke({"messages": [{"role": "user", "content": request}]})
