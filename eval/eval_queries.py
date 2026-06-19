"""
Evaluation dataset for the AI Research Assistant.

10 queries covering all system capabilities:
- RAG retrieval quality
- Tool usage correctness
- Routing accuracy
- Edge cases

Run with: python -m eval.eval_queries
"""
from src.logging_config import setup_logging, get_logger
from src.agents.coordinator import CoordinatorTeam
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

load_dotenv()
setup_logging()
logger = get_logger("eval")
console = Console()


EVAL_QUERIES = [
    # --- RAG Queries (should route to Retriever Agent) ---
    {
        "id": "rag_1",
        "query": "What is cross-validation and why is it important in machine learning?",
        "expected_agent": "Retriever Agent",
        "expected_tool": "knowledge_search",
        "category": "rag",
        "success_criteria": "Answer mentions k-fold, train/test split, and cites textbook content",
    },
    {
        "id": "rag_2",
        "query": "Explain the difference between supervised and unsupervised learning",
        "expected_agent": "Retriever Agent",
        "expected_tool": "knowledge_search",
        "category": "rag",
        "success_criteria": "Distinguishes labeled vs unlabeled data, gives examples",
    },
    {
        "id": "rag_3",
        "query": "How do random forests prevent overfitting compared to single decision trees?",
        "expected_agent": "Retriever Agent",
        "expected_tool": "knowledge_search",
        "category": "rag",
        "success_criteria": "Mentions ensemble, bagging, feature randomization",
    },
    {
        "id": "rag_4",
        "query": "What preprocessing steps does scikit-learn provide?",
        "expected_agent": "Retriever Agent",
        "expected_tool": "knowledge_search",
        "category": "rag",
        "success_criteria": "Mentions StandardScaler, MinMaxScaler, or similar sklearn tools",
    },

    # --- Tool Queries (should route to General Agent) ---
    {
        "id": "tool_1",
        "query": "What is 127 * 53 + 89?",
        "expected_agent": "General Agent",
        "expected_tool": "calculator",
        "category": "calculator",
        "success_criteria": "Returns 6820 (127*53=6731, +89=6820)",
    },
    {
        "id": "tool_2",
        "query": "Convert 98.6 degrees Fahrenheit to Celsius",
        "expected_agent": "General Agent",
        "expected_tool": "calculator",
        "category": "calculator",
        "success_criteria": "Returns 37.0 (body temperature)",
    },
    {
        "id": "tool_3",
        "query": "Search the web for the latest developments in large language models",
        "expected_agent": "General Agent",
        "expected_tool": "web_search",
        "category": "web_search",
        "success_criteria": "Returns search results (real or mock)",
    },

    # --- Direct Answer Queries (should route to General Agent, no tool) ---
    {
        "id": "direct_1",
        "query": "What is the capital of Japan?",
        "expected_agent": "General Agent",
        "expected_tool": None,
        "category": "general_knowledge",
        "success_criteria": "Returns 'Tokyo' without using any tool",
    },

    # --- Edge Cases ---
    {
        "id": "edge_1",
        "query": "Calculate the square root of the number of features in a typical iris dataset",
        "expected_agent": "General Agent",
        "expected_tool": "calculator",
        "category": "hybrid",
        "success_criteria": "Identifies 4 features, calculates sqrt(4)=2",
    },
    {
        "id": "edge_2",
        "query": "What is the meaning of life?",
        "expected_agent": "General Agent",
        "expected_tool": None,
        "category": "general_knowledge",
        "success_criteria": "Answers philosophically, does not search the knowledge base",
    },
]


class Evaluator:
    """Runs the evaluation dataset against the multi-agent system.

    Responsibilities:
        - Iterates over EVAL_QUERIES and runs each through the coordinator
        - Captures responses and tracks success/failure
        - Prints a summary report

    Usage:
        evaluator = Evaluator()
        evaluator.run()
    """

    def __init__(self):
        
        self._coordinator = CoordinatorTeam()
        self._results = []

    def _extract_retrieved_chunks(self, response) -> str:
        """Helper to extract retrieved chunks from team run output."""
        if not response:
            return ""
        
        # 1. Search top-level tools
        if getattr(response, "tools", None):
            for tool in response.tools:
                if tool.tool_name == "knowledge_search" and tool.result:
                    return tool.result
        
        # 2. Search member agent responses
        if getattr(response, "member_responses", None):
            for member in response.member_responses:
                # Check member tools
                if getattr(member, "tools", None):
                    for tool in member.tools:
                        if tool.tool_name == "knowledge_search" and tool.result:
                            return tool.result
                # Recurse into member
                nested = self._extract_retrieved_chunks(member)
                if nested:
                    return nested

        # 3. Fallback to message history
        if getattr(response, "messages", None):
            for msg in response.messages:
                if msg.role == "tool" and msg.name == "knowledge_search":
                    return str(msg.content)

        return ""

    def run(self) -> None:
        """Execute all evaluation queries and print a summary report."""
        logger.info("Starting evaluation with %d queries", len(EVAL_QUERIES))

        for query_spec in EVAL_QUERIES:
            self._run_single(query_spec)

        self._print_summary()

    def _run_single(self, query_spec: dict) -> None:
        """Run a single evaluation query."""
        query_id = query_spec["id"]
        query = query_spec["query"]

        console.print(f"\n[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold magenta]")
        console.print(Panel(
            f"[bold yellow][{query_id}][/bold yellow] [white]{query}[/white]\n\n"
            f"[bold dim]Expected Path:[/bold dim] [cyan]{query_spec['expected_agent']}[/cyan] -> [green]{query_spec['expected_tool'] or 'Direct'}[/green]",
            title=f"[bold blue]Evaluating Query[/bold blue]",
            title_align="left",
            border_style="blue"
        ))

        logger.info(
            "Running eval query [%s]: '%s' (expected: %s -> %s)",
            query_id, query, query_spec["expected_agent"], query_spec["expected_tool"],
        )

        try:
            response = self._coordinator.team.run(query)
            content = response.content if response else "No response"
            
            # Extract retrieved chunks for RAG transparency
            retrieved_chunks = self._extract_retrieved_chunks(response)

            if retrieved_chunks:
                console.print(Panel(
                    retrieved_chunks.strip(),
                    title="[bold green]Retrieved Chunks (RAG Transparency)[/bold green]",
                    title_align="left",
                    border_style="green",
                    padding=(1, 2)
                ))
            
            console.print(Panel(
                Markdown(str(content).strip()),
                title="[bold gold1]Final Answer[/bold gold1]",
                title_align="left",
                border_style="gold1",
                padding=(1, 2)
            ))

            self._results.append({
                "id": query_id,
                "category": query_spec["category"],
                "status": "completed",
                "response_preview": str(content)[:200],
            })
            logger.info("Eval query [%s] completed successfully", query_id)

        except Exception as e:
            self._results.append({
                "id": query_id,
                "category": query_spec["category"],
                "status": "error",
                "error": str(e),
            })
            logger.error("Eval query [%s] failed: %s", query_id, e)
            console.print(Panel(f"[bold red]ERROR:[/bold red] {e}", border_style="red"))

    def _print_summary(self) -> None:
        """Print the evaluation summary report."""
        completed = sum(1 for r in self._results if r["status"] == "completed")
        errors = len(self._results) - completed

        console.print(f"\n[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold magenta]\n")
        
        # Create rich Table
        table = Table(title="[bold cyan]Category Breakdown Summary[/bold cyan]", border_style="cyan")
        table.add_column("Category", style="yellow")
        table.add_column("Passed/Total", style="green", justify="center")
        table.add_column("Status", justify="center")

        categories = set(r["category"] for r in self._results)
        for cat in sorted(categories):
            cat_results = [r for r in self._results if r["category"] == cat]
            cat_completed = sum(1 for r in cat_results if r["status"] == "completed")
            status = "[bold green]PASS[/bold green]" if cat_completed == len(cat_results) else "[bold red]FAIL[/bold red]"
            table.add_row(cat, f"{cat_completed}/{len(cat_results)}", status)

        console.print(table)
        console.print()

        summary_panel_text = (
            f"[bold]Total Queries Evaluated:[/bold] {len(EVAL_QUERIES)}\n"
            f"[bold green]Completed successfully:[/bold green] {completed}\n"
            f"[bold red]Errors/Failed execution:[/bold red] {errors}"
        )
        console.print(Panel(summary_panel_text, title="[bold blue]Overall Evaluation Metrics[/bold blue]", border_style="blue", expand=False))

        if errors > 0:
            console.print("\n[bold red]Failed Queries:[/bold red]")
            for r in self._results:
                if r["status"] == "error":
                    console.print(f"  [bold red]• {r['id']}:[/bold red] {r.get('error', 'Unknown error')}")

        console.print(f"\n[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold magenta]")
        logger.info(
            "Evaluation complete: %d/%d passed, %d errors",
            completed, len(EVAL_QUERIES), errors,
        )


def main():
    """Entry point for running the evaluation."""
    console.print(Panel(
        "[bold cyan]AI Research Assistant -- Evaluation Suite[/bold cyan]\n"
        "[dim]Automated Evaluation of RAG Quality, Math, Web Search, and Routing Accuracy[/dim]",
        border_style="cyan",
        expand=False
    ))

    evaluator = Evaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
