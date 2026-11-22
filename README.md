# Agentic AI Research Assistant 🔎

ResearchScout is a tool-using AI assistant designed to help researchers find, screen, and summarize scientific papers from arXiv. It links each paper to its source and can export results as CSV or JSON on request.

This is my first agentic AI project. I chose literature discovery to practice translating a research question into a search query, selecting tools, refining weak searches, and grounding an answer in retrieved evidence. The repository currently documents the proof-of-concept design.

---

## 1. Use Case

A researcher asks, “Find five recent papers on agentic AI in healthcare and summarize the most relevant findings.” The assistant turns the request into a search query and screens the returned titles and abstracts for relevance. If the first search yields too few relevant papers, the assistant can refine its terms and search again.

Users can specify a topic, keywords, a date range, and a target paper count. The planned response includes a ranked list with each paper’s title, authors, year, source link, relevance to the question, and an abstract-based summary. It can also report themes across papers and flag gaps in the evidence.

---

## 2. LLM & Provider

| Component | Planned choice | Role |
| --- | --- | --- |
| Provider | OpenAI | Provides API access to the language model. |
| Language model | GPT-4o mini (`gpt-4o-mini`) | Interprets the request, selects tools, and summarizes retrieved evidence. |
| Agent framework | LlamaIndex | Orchestrates tool execution in Python. |
| Paper source | arXiv API | Supplies paper identifiers, metadata, abstracts, and source links. |

---

## 3. Agent Tools

### `search_papers(query, max_results)`

Searches arXiv using a keyword query and a result limit. Returns candidate paper identifiers and metadata for initial screening.

### `get_paper_details(paper_id)`

Retrieves the title, authors, date, abstract, and original source URL for a selected paper.

### `save_results(records, format)`

Writes the retrieved records and their summaries to a local CSV or JSON file when the user requests an export.

---

## 4. Agent Workflow

1. Translate the research question and constraints into an arXiv query.
2. Search for candidates and retrieve details for promising papers.
3. Screen papers for relevance. Refine the query when coverage is weak, with a default limit of three search attempts in total.
4. Rank the relevant papers and summarize what the retrieved abstracts support. If fewer papers qualify than requested, report the shortfall.
5. Export the results if the user requested a file.

The application is responsible for validating tool arguments, enforcing limits, and handling API errors, duplicate records, and missing fields.

---

## 5. System Prompt

```text
You are ResearchScout, an assistant for scientific paper discovery.

1. Identify the research topic, date range, and target paper count. Ask a
   clarifying question only if you cannot make useful progress. State any
   assumptions that affect the search.
2. Use search_papers to discover candidates. Refine weak queries when useful,
   with at most three search attempts unless the user asks you to continue.
3. Use get_paper_details before making substantive claims about a paper.
   Rank papers by relevance and link to their original source records.
4. Base summaries on retrieved evidence. Label abstract-based summaries
   and identify preprints when that status is known. Do not invent citations,
   imply full-text review, or turn research findings into clinical advice.
5. Briefly report the search scope, query refinements, relevant papers, and
   limitations. Label cross-paper observations as your synthesis.
6. Call save_results only when the user requests an export. Report whether
   the export succeeded.
```

---

## 6. Proof-of-Concept Evaluation

- **Citation validity** checks that each source link resolves to the reported paper.
- **Top-k relevance** measures how many of the highest-ranked papers address the research question.
- **Summary faithfulness** checks that the retrieved abstract supports each summary.
- **Tool efficiency** tracks tool calls and search retries per request.
- **Completeness** checks that the response includes paper metadata, source links, summaries, and limitations.

A possible extension is to compare a single-search workflow with an agent that refines queries and retrieves paper details as needed. The comparison would measure relevance, summary faithfulness, cost, latency, and tool use on the same research questions.
