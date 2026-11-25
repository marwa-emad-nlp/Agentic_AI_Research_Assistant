from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from .paper_tools import PaperTools


class PaperSearchAgent:
    """Find arXiv papers through the paper search tool."""

    def __init__(self, model: BaseChatModel, paper_tools: PaperTools) -> None:
        self.graph = create_agent(
            model=model,
            tools=[paper_tools.search_papers],
            name="paper_search_agent",
            system_prompt=(
                "You are a research paper search specialist. "
                "Your first action on every delegated search request must be "
                "a search_papers tool call. Use an English arXiv query and "
                "respect the requested result count. "
                "Report only papers returned by the tool, with their titles "
                "and source URLs. Say if the search returns no papers."
            ),
        )

    def invoke(self, request: str) -> dict:
        """Run the search agent on its own and return its message history."""
        return self.graph.invoke({"messages": [{"role": "user", "content": request}]})
