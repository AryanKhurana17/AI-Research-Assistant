"""
One-time script to index the PDF into DuckDB.
Run this before starting the agent:
    python -m src.knowledge.index
"""
from src.knowledge.pdf_knowledge import KnowledgeBase


def main():
    print("=" * 60)
    print("Indexing PDF")
    print("=" * 60)

    kb = KnowledgeBase()
    count = kb.index(recreate=False)
    print(f"\nDone. {count} chunks ready for retrieval.")


if __name__ == "__main__":
    main()
