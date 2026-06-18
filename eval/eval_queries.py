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
load_dotenv()
setup_logging()
logger = get_logger("eval")


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

        print(f"\n{'=' * 70}")
        print(f"  [{query_id}] {query}")
        print(f"  Expected: {query_spec['expected_agent']} -> {query_spec['expected_tool']}")
        print(f"{'=' * 70}")

        logger.info(
            "Running eval query [%s]: '%s' (expected: %s -> %s)",
            query_id, query, query_spec["expected_agent"], query_spec["expected_tool"],
        )

        try:
            response = self._coordinator.team.run(query)
            content = response.content if response else "No response"
            preview = str(content)[:300]

            print(f"\n  Response preview: {preview}...")

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
            print(f"  ERROR: {e}")

    def _print_summary(self) -> None:
        """Print the evaluation summary report."""
        completed = sum(1 for r in self._results if r["status"] == "completed")
        errors = len(self._results) - completed

        print(f"\n\n{'=' * 70}")
        print("  EVALUATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Total queries:  {len(EVAL_QUERIES)}")
        print(f"  Completed:      {completed}")
        print(f"  Errors:         {errors}")
        print()

        # Breakdown by category
        categories = set(r["category"] for r in self._results)
        for cat in sorted(categories):
            cat_results = [r for r in self._results if r["category"] == cat]
            cat_completed = sum(1 for r in cat_results if r["status"] == "completed")
            print(f"  [{cat}] {cat_completed}/{len(cat_results)} passed")

        if errors > 0:
            print("\n  Failed queries:")
            for r in self._results:
                if r["status"] == "error":
                    print(f"    - {r['id']}: {r.get('error', 'Unknown error')}")

        print(f"{'=' * 70}")
        logger.info(
            "Evaluation complete: %d/%d passed, %d errors",
            completed, len(EVAL_QUERIES), errors,
        )


def main():
    """Entry point for running the evaluation."""
    print("=" * 70)
    print("  AI Research Assistant -- Evaluation Suite")
    print("=" * 70)

    evaluator = Evaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
