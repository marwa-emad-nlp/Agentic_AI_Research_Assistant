import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
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
        self._summary_agent = summary_agent
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
            post_model_hook=self._finalize_response,
            output_mode="full_history",
            include_agent_name="inline",
        ).compile()

    def invoke(self, request: str) -> dict:
        """Run the supervisor and return its trace and final answer."""
        result = self.graph.invoke({"messages": [{"role": "user", "content": request}]})
        return {**result, "answer": result["messages"][-1].content}

    def _finalize_response(self, state: dict) -> dict:
        """Ground the final supervisor message before the graph returns it."""
        messages = state["messages"]
        if messages[-1].tool_calls:
            return {}

        papers = {}
        last_success = None
        search_attempted = False

        for index, message in enumerate(messages):
            if message.type != "tool":
                continue
            if message.name == "transfer_to_paper_search_agent":
                search_attempted = True
            if message.name == "search_papers":
                search_attempted = True
                if message.status == "success":
                    last_success = index
                    records = message.content
                    if isinstance(records, str):
                        records = json.loads(records)
                    for paper in records:
                        papers[paper["url"]] = paper

        answer = messages[-1].content
        if papers:
            summary = next(
                (
                    message.content
                    for message in reversed(messages[last_success + 1:])
                    if message.type == "ai"
                    and message.name == "paper_summary_agent"
                    and not message.tool_calls
                    and message.content
                ),
                None,
            )
            if summary is None:
                evidence = json.dumps(list(papers.values()), ensure_ascii=False)
                request = next(message.content for message in messages if message.type == "human")
                summary_result = self._summary_agent.invoke(
                    f"Summarize these abstracts for this request: {request}\n\n{evidence}"
                )
                summary = summary_result["messages"][-1].content
            answer = f"Abstract-based summary\n\n{summary}\n\nSources\n" + "\n".join(
                f"- [{paper['title']}]({paper['url']})" for paper in papers.values()
            )
        elif last_success is not None:
            answer = "No papers were found."
        elif search_attempted or "no papers were found" in str(answer).lower():
            answer = "The search did not return a verified result."

        final_message = AIMessage(content=answer, name="supervisor", id=messages[-1].id)
        return {"messages": [final_message]}
