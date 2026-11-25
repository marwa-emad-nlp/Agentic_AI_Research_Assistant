# Agentic AI Research Assistant 🔎

ResearchScout is a Python proof of concept for finding papers on arXiv and summarizing their abstracts. This is my first agentic AI project. I chose literature discovery to practice tool calling, task delegation, and grounding answers in retrieved evidence.

## 1. Use Case

A researcher can ask, “Find one arXiv paper about large language models and summarize its abstract.” The supervisor delegates discovery to a search agent, then sends the retrieved material to a summary agent. The intended response includes the paper title, its arXiv link, and a short abstract-based summary.

## 2. LLM & Framework

| Component | Role |
| --- | --- |
| Python 3.12 | Runs the project and its notebooks. |
| Ollama with Llama 3.2 (`llama3.2:latest`) | Runs the local model used by the supervisor and both agents. |
| LangChain | Builds the specialized agents with `create_agent`. |
| LangGraph Supervisor | Connects the agents with `create_supervisor` and manages handoffs. |
| arXiv API | Supplies paper titles, links, and abstracts through the `arxiv` Python package. |

## 3. Agent Workflow

1. **Supervisor** receives the request and delegates paper discovery to the search agent.
2. **Search agent** builds an arXiv query and calls `PaperTools.search_papers(query, max_results)`. The tool returns each paper's title, URL, and abstract.
3. **Supervisor** receives the search results and delegates to the summary agent when a summary is requested and abstracts are available.
4. **Summary agent** uses the available abstracts, then returns its summary to the supervisor for the final answer.

The supervisor compiles the workflow with `.compile()`. The shared message history keeps tool results available to both agents through `output_mode="full_history"`. The `include_agent_name="inline"` setting identifies the agent that wrote a message.

## 4. Project Structure

| Python component | Jupyter test |
| --- | --- |
| [PaperTools](src/paper_tools.py) | [01 · Search arXiv directly](notebooks/01_search_papers.ipynb) |
| [PaperSearchAgent](src/paper_search_agent.py) | [02 · Run the search agent](notebooks/02_paper_search_agent.ipynb) |
| [PaperSummaryAgent](src/paper_summary_agent.py) | [03 · Summarize a sample abstract](notebooks/03_paper_summary_agent.ipynb) |
| [PaperResearchSupervisor](src/paper_research_supervisor.py) | [04 · Run both agents through the supervisor](notebooks/04_paper_research_supervisor.ipynb) |

Each notebook imports the classes from `src`, creates objects, and exercises one part of the workflow.

## 5. Run the Notebooks

From the repository folder in PowerShell, create a virtual environment and install the dependencies. Start the Ollama app before pulling the model.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install arxiv "langchain>=1,<2" langchain-ollama langgraph-supervisor notebook
ollama pull llama3.2:latest
.\.venv\Scripts\python.exe -m notebook
```

Open a notebook using the project's `.venv` kernel and run the cells from top to bottom. The supervisor notebook prints the handoffs and checks for search results before displaying the final answer.

## 6. Proof-of-Concept Evaluation

The search and summary agents have been exercised individually. A controlled test verified both supervisor handoffs and the search tool call using a simulated paper. In live supervisor tests, Llama 3.2 sometimes skipped the search tool, so end-to-end routing with Ollama and arXiv remains unverified.

For future evaluation, I plan to track whether the supervisor picks the correct agent, whether the top results match the research question, and whether each summary is supported by its abstract. CSV and JSON export are planned extensions.
