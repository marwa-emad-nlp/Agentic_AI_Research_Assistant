from langchain_core.language_models import BaseChatModel
from langgraph_supervisor import create_supervisor

from .paper_search_agent import PaperSearchAgent
from .paper_summary_agent import PaperSummaryAgent


class PaperResearchSupervisor:
    """Coordinate paper search and abstract-based summarization."""

    def __init__(
        self,
        model: BaseChatModel,
        search_agent: PaperSearchAgent,
        summary_agent: PaperSummaryAgent,
    ) -> None:
        self.graph = create_supervisor(
            [search_agent.graph, summary_agent.graph], # here we pass the graphs of the search and summary agents to the supervisor
            model=model,
            prompt=(
                "You coordinate two research paper specialists. "
                "For requests to find papers, first call transfer_to_paper_search_agent. "
                "If the user also wants summaries and search_papers returned papers, "
                "next call transfer_to_paper_summary_agent so it can use the abstracts "
                "in the shared history. If the user supplied an abstract and asks only "
                "for a summary, call transfer_to_paper_summary_agent directly. "
                "Delegate to one worker at a time and stop after the requested work is done. "
                "Report when no papers were found. Preserve source links and label "
                "summaries as abstract-based. Use the user's language."
            ),
            output_mode="full_history",
            include_agent_name="inline",
        ).compile()

    def invoke(self, request: str) -> dict:
        """Run the supervisor and return the conversation and tool messages."""
        return self.graph.invoke({"messages": [{"role": "user", "content": request}]})
