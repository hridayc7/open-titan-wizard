import json
import argparse
from opentitan_rag import OpenTitanRAG
import os


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenTitan RAG on questions in a JSON file and overwrite answers.")
    parser.add_argument('--input', required=True,
                        help='Path to input JSON file (with question/answer pairs).')
    parser.add_argument('--output', required=True,
                        help='Path to output JSON file to save new answers.')
    parser.add_argument('--docs_dir', default="raptor_output",
                        help='Directory with vector store and document files.')
    parser.add_argument('--translate', action='store_true',
                        help='Use query translation.')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Documents to retrieve per query.')
    parser.add_argument('--top_sources', type=int, default=7,
                        help='Sources to include in the context.')
    parser.add_argument('--session_id', default=None,
                        help='Session ID (optional).')
    parser.add_argument(
        '--model', default="claude-3-opus-20240229", help='Claude model to use.')

    args = parser.parse_args()

    # Load the input questions
    with open(args.input, 'r') as f:
        original_data = json.load(f)

    # Initialize the RAG system
    rag = OpenTitanRAG(
        docs_dir=args.docs_dir,
        model=args.model
    )

    # Generate new answers for each question
    updated_data = []
    for i, item in enumerate(original_data):
        question = item["question"]
        print(f"[{i+1}/{len(original_data)}] Querying: {question}")

        result = rag.process_query(
            query=question,
            session_id=args.session_id,
            use_query_translation=args.translate,
            top_k=args.top_k,
            top_sources=args.top_sources
        )

        updated_data.append({
            "question": question,
            "answer": result["answer"]
        })

    # Save the new results
    with open(args.output, 'w') as f:
        json.dump(updated_data, f, indent=2)

    print(f"\nResponses saved to {args.output}")


if __name__ == "__main__":
    main()
