import arxiv


class PaperTools:
    """Tools for finding research papers on arXiv."""

    def __init__(self) -> None:
        self._client = arxiv.Client()

    def search_papers(self, query: str, max_results: int = 3) -> list[dict[str, str]]:
        """Return matching papers with their titles, links, and abstracts."""
        search = arxiv.Search(query=query, max_results=max_results)
        papers = []
        for paper in self._client.results(search):
            papers.append({
                "title": paper.title,
                "url": paper.entry_id,
                "abstract": paper.summary,
            })
        return papers