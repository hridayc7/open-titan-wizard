#!/usr/bin/env python3
import os
import torch
import pickle
import logging
import argparse
from typing import List, Dict, Any, Optional
from collections import Counter, deque
import anthropic
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenTitanRAG:
    """
    OpenTitan RAG system with optional query translation and chat history.
    """

    def __init__(
        self,
        docs_dir: str,
        api_key: str = None,
        model: str = "claude-3-7-sonnet-20250219",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        base_url: str = "https://opentitan.org/book/",
        max_history_length: int = 20
    ):
        """Initialize the OpenTitan RAG system"""
        self.docs_dir = docs_dir
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.embedding_model_name = embedding_model
        self.device = device
        self.base_url = base_url
        self.max_history_length = max_history_length

        # Initialize chat history storage
        # We'll use a dictionary to store chat histories for different sessions
        self.chat_histories = {}

        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set as ANTHROPIC_API_KEY env variable")

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(
            embedding_model, device=device)

        # Initialize langchain embeddings
        self.lc_embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
        )

        # Load vector store
        self.vector_store = self._load_vector_store()

        # Load document tree for metadata access (optional)
        try:
            self.document_tree = self._load_document_tree()
            self.node_summaries = self._load_node_summaries()
        except Exception as e:
            logger.warning(f"Could not load document tree: {e}")
            self.document_tree = None
            self.node_summaries = {}

        logger.info("OpenTitan RAG system initialized successfully")

    def _load_document_tree(self):
        """Load the document tree from pickle file."""
        tree_path = os.path.join(self.docs_dir, 'document_tree.pkl')
        if not os.path.exists(tree_path):
            logger.warning(f"Document tree not found at {tree_path}")
            return None

        logger.info(f"Loading document tree from {tree_path}")
        with open(tree_path, 'rb') as f:
            return pickle.load(f)

    def _load_node_summaries(self):
        """Load the node summaries from pickle file."""
        summaries_path = os.path.join(self.docs_dir, 'node_summaries.pkl')
        if not os.path.exists(summaries_path):
            logger.warning(f"Node summaries not found at {summaries_path}")
            return {}

        logger.info(f"Loading node summaries from {summaries_path}")
        with open(summaries_path, 'rb') as f:
            return pickle.load(f)

    def _load_vector_store(self):
        """Load the FAISS vector store."""
        index_path = os.path.join(self.docs_dir, 'faiss_index')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        logger.info(f"Loading vector store from {index_path}")
        # Set allow_dangerous_deserialization to True since you created this index yourself
        return FAISS.load_local(index_path, self.lc_embeddings, allow_dangerous_deserialization=True)

    def _claude_generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text using Claude API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=temperature,
            system="You are a helpful expert assistant that provides accurate, concise answers based solely on the provided context.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get chat history for a specific session"""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = deque(
                maxlen=self.max_history_length)
        return self.chat_histories[session_id]

    def add_to_chat_history(self, session_id: str, query: str, answer: str):
        """Add a query-answer pair to the chat history"""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = deque(
                maxlen=self.max_history_length)

        self.chat_histories[session_id].append({
            "query": query,
            "answer": answer
        })

    def clear_chat_history(self, session_id: str):
        """Clear chat history for a specific session"""
        if session_id in self.chat_histories:
            self.chat_histories[session_id].clear()

    def format_chat_history(self, session_id: str) -> str:
        """Format chat history as a string for inclusion in prompts"""
        if session_id not in self.chat_histories or not self.chat_histories[session_id]:
            return ""

        history_text = "Previous conversation:\n"
        for i, entry in enumerate(self.chat_histories[session_id]):
            history_text += f"User: {entry['query']}\n"
            history_text += f"Assistant: {entry['answer']}\n\n"

        return history_text

    def _create_source_url(self, source: str) -> str:
        """
        Create a URL for a source based on its path or ID
        Simple rules:
        1. If source is a number or has a number in parentheses -> document-view URL
        2. If source is a .txt file -> convert underscores to slashes and append to base_url
        3. For real files (.c, .h, etc.) -> use GitHub URL
        """
        if source == "Unknown":
            return "#"

        # Handle numeric IDs
        if source.isdigit() or (len(source) > 0 and source[-1] == ')' and source.rsplit('(', 1)[1][:-1].isdigit()):

            return "#"

        # Simple conversion for text files: replace underscores with slashes
        if source.endswith('.txt'):
            # Remove .txt extension
            path = source.replace('.txt', '')

            # Replace underscores with slashes
            url_path = path.replace('_', '/')

            # Append .html extension
            return f"{self.base_url}{url_path}.html"

        # For real files in the GitHub repo
        if source.endswith(('.c', '.h', '.rs', '.py', '.cpp', '.json', '.html')):
            # Clean up path if needed
            if source.startswith('./') or source.startswith('/'):
                source = source.lstrip('./')
            return f"https://github.com/lowRISC/opentitan/tree/master/{source}"

        # Default fallback for other sources
        return "#"

    def _add_missing_source_links(self, answer: str, source_to_url: Dict[int, str], valid_citation_sources: List[int]) -> str:
        """Add clickable URLs to any source references that don't have them, remove invalid citations"""
        import re

        # Find all source references without links: Source X or [Source X]
        source_pattern = r'(?:\[)?Source\s+(\d+)(?:\])?(?!\()'

        def replace_with_link(match):
            source_num = int(match.group(1))
            if source_num in valid_citation_sources and source_num in source_to_url:
                return f"[Source {source_num}]({source_to_url[source_num]})"
            else:
                # Replace citations to non-citable sources
                return f"[Information from context]"

        # Replace all source references with proper links or generalize invalid ones
        linked_answer = re.sub(source_pattern, replace_with_link, answer)

        return linked_answer

    def translate_query(self, query: str) -> List[str]:
        """
        Translate a complex query into simpler component questions
        """
        logger.info(f"Translating query into components: {query}")

        # Use Claude to generate relevant sub-queries
        breakdown_prompt = f"""Break down this complex query into 3-5 simpler component questions that would help answer the main question. 
Return ONLY the list of questions, one per line, without any additional text.

Main question: {query}"""

        response = self._claude_generate(breakdown_prompt, temperature=0.2)

        # Clean up response and extract questions
        sub_queries = [
            line.strip() for line in response.split('\n')
            if line.strip() and "?" in line
        ]

        # Always include the original query
        if query not in sub_queries:
            sub_queries.append(query)

        logger.info(f"Generated {len(sub_queries)} sub-queries")
        return sub_queries

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_clusters: bool = True,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not loaded")

        logger.info(f"Retrieving documents for query: {query}")

        # Perform retrieval
        if include_clusters:
            retrieved_docs = self.vector_store.similarity_search_with_score(
                query, k=top_k)
        else:
            # Only retrieve document chunks (not cluster summaries)
            retrieved_docs = self.vector_store.similarity_search_with_score(
                query,
                k=top_k,
                filter={"type": "document"}
            )

        # Format results
        results = []
        for doc, score in retrieved_docs:
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            })

        # Print results if verbose
        if verbose:
            print(f"\nQuery: {query}")
            print("\nRetrieved Documents:")
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} (Score: {result['score']:.4f}) ---")
                # print(f"Type: {result['metadata']['type']}")
                # source = result['metadata'].get(
                #     'source', result['metadata'].get('node_id', 'N/A'))
                # print(f"Source: {source}")
                # print(f"Content: {result['content'][:100]}...")
                print(f"Result Object: {result}")

        return results

    def aggregate_documents(
        self,
        sub_queries: List[str],
        top_k: int = 5,
        include_clusters: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve and aggregate documents for multiple sub-queries
        """
        # Track document frequencies and content
        all_docs = []
        doc_sources = []
        doc_contents = {}

        # Retrieve documents for each sub-query
        for i, sub_query in enumerate(sub_queries):
            logger.info(f"Retrieving for sub-query {i+1}: {sub_query}")

            docs = self.retrieve(
                sub_query,
                top_k=top_k,
                include_clusters=include_clusters,
                verbose=False
            )

            all_docs.extend(docs)

            # Track document sources and content
            for doc in docs:
                source = doc['metadata'].get(
                    'source', doc['metadata'].get('node_id', 'Unknown'))
                doc_sources.append(source)
                doc_contents[source] = doc['content']

        # Count source frequencies
        source_counter = Counter(doc_sources)

        # Sort documents by frequency
        unique_sources = list(source_counter.keys())
        unique_sources.sort(key=lambda s: source_counter[s], reverse=True)

        logger.info(
            f"Retrieved {len(all_docs)} total documents across {len(unique_sources)} unique sources")

        return {
            "all_docs": all_docs,
            "source_counter": source_counter,
            "doc_contents": doc_contents,
            "unique_sources": unique_sources
        }

    def generate_answer(
        self,
        query: str,
        session_id: str = None,
        documents: List[Dict[str, Any]] = None,
        aggregated_results: Dict[str, Any] = None,
        top_sources: int = 7,
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Generate an answer based on retrieved documents and chat history (if available)
        """
        logger.info(f"Generating answer for query: {query}")

        # Create a mapping of source index to URL
        source_to_url = {}
        source_index_mapping = {}
        valid_citation_sources = []

        # Format context based on retrieval method
        if aggregated_results:
            # Use weighted aggregation results
            source_counter = aggregated_results["source_counter"]
            doc_contents = aggregated_results["doc_contents"]
            unique_sources = aggregated_results["unique_sources"]

            # Build weighted context with URLs only for text files
            context = ""
            for i, source in enumerate(unique_sources[:top_sources]):
                if source in doc_contents:
                    # Check if this is a text file (citable)
                    is_citable = False
                    source_url = "#"  # Default: non-citable

                    # Rule 1: If URL is in metadata, use it and mark as citable
                    for doc in aggregated_results["all_docs"]:
                        if doc['metadata'].get('source') == source and 'url' in doc['metadata']:
                            source_url = doc['metadata']['url']
                            is_citable = True
                            break

                    # Rule 2: If source is a .txt file, generate URL and mark as citable
                    if not is_citable and source.endswith('.txt'):
                        source_url = self._create_source_url(source)
                        is_citable = True

                    # Mark source appropriately
                    if is_citable:
                        source_to_url[i+1] = source_url
                        valid_citation_sources.append(i+1)
                        citation_mark = "[CITABLE]"
                    else:
                        citation_mark = "[REFERENCE ONLY]"

                    source_index_mapping[source] = i+1

                    # Add to context with URL and citation mark
                    context += f"{citation_mark} [Source {i+1}: {source} (appeared {source_counter[source]} times)]\n"
                    context += f"URL: {source_url}\n{doc_contents[source]}\n\n"
        else:
            # Use standard retrieval results
            context = ""
            for i, doc in enumerate(documents[:top_sources]):
                source = doc['metadata'].get(
                    'source', doc['metadata'].get('node_id', 'Unknown'))

                # Check if this is a text file (citable)
                is_citable = False
                source_url = "#"  # Default: non-citable

                # Rule 1: If URL is in metadata, use it and mark as citable
                if 'url' in doc['metadata']:
                    source_url = doc['metadata']['url']
                    is_citable = True
                # Rule 2: If source is a .txt file, generate URL and mark as citable
                elif source.endswith('.txt'):
                    source_url = self._create_source_url(source)
                    is_citable = True

                # Mark source appropriately
                if is_citable:
                    source_to_url[i+1] = source_url
                    valid_citation_sources.append(i+1)
                    citation_mark = "[CITABLE]"
                else:
                    citation_mark = "[REFERENCE ONLY]"

                source_index_mapping[source] = i+1

                # Add to context with URL and citation mark
                context += f"{citation_mark} [Source {i+1}: {source}]\n"
                context += f"URL: {source_url}\n{doc['content']}\n\n"

        # Include chat history if session_id is provided and history exists
        chat_history = ""
        if session_id and session_id in self.chat_histories and self.chat_histories[session_id]:
            chat_history = self.format_chat_history(session_id)

        # Use custom or default prompt template
        if prompt_template is None:
            valid_sources_list = ", ".join(
                [str(src) for src in valid_citation_sources])

            prompt = f"""Answer the following question based on the information in the provided context and the conversation history (if any).
If you cannot answer the question based solely on the context, say "I don't have enough information in the provided context to answer this question."

{chat_history}

Context information (including source URLs):
{context}

Current question: {query}

Your answer should be thorough but concise, providing specific information from the context that directly addresses the question.
IMPORTANT: You may ONLY cite sources marked as [CITABLE] in your answer. These are sources: {valid_sources_list}.
You can use information from [REFERENCE ONLY] sources to inform your answer, but do not cite them directly. Don't even say [REFERENCE ONLY] or [Information from Context] in your answer.
When you reference a citable source, include the source number and the clickable URL like this: [X](URL).
For example, if you're referencing Source 1 and it's citable, write: [Source 1]({source_to_url.get(1, "#")}).
If the question is a follow-up to a previous question in the chat history, consider that context in your answer."""
        else:
            # If using a custom prompt template, we still need to pass the valid_citation_sources
            valid_sources_list = ", ".join(
                [str(src) for src in valid_citation_sources])
            prompt = prompt_template.format(
                context=context,
                query=query,
                chat_history=chat_history,
                source_url_example=f"[Source 1]({source_to_url.get(1, '#')})",
                valid_sources_list=valid_sources_list
            )

        # Generate answer
        answer = self._claude_generate(prompt, temperature=0.0)

        # Process the answer to ensure all source references have URLs and remove invalid citations
        answer = self._add_missing_source_links(
            answer, source_to_url, valid_citation_sources)

        return answer

    def process_query(
        self,
        query: str,
        session_id: str = None,
        use_query_translation: bool = False,
        top_k: int = 5,
        top_sources: int = 7,
        include_clusters: bool = True,
        prompt_template: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Process a query using the OpenTitan RAG system
        """
        logger.info(f"Processing query: {query}")
        results = {"query": query}

        if use_query_translation:
            # Step 1: Translate query into sub-queries
            sub_queries = self.translate_query(query)
            results["sub_queries"] = sub_queries

            # Step 2: Retrieve and aggregate documents
            aggregated_results = self.aggregate_documents(
                sub_queries, top_k, include_clusters)

            retrieval_stats = {
                "total_docs": len(aggregated_results["all_docs"]),
                "unique_sources": len(aggregated_results["unique_sources"]),
                "source_frequencies": dict(aggregated_results["source_counter"])
            }
            results["retrieval_stats"] = retrieval_stats

            # Step 3: Generate answer with chat history
            answer = self.generate_answer(
                query,
                session_id=session_id,
                aggregated_results=aggregated_results,
                top_sources=top_sources,
                prompt_template=prompt_template
            )
        else:
            # Standard retrieval and generation
            documents = self.retrieve(
                query, top_k, include_clusters, verbose)
            results["documents"] = documents

            # Generate answer with chat history
            answer = self.generate_answer(
                query,
                session_id=session_id,
                documents=documents,
                top_sources=top_sources,
                prompt_template=prompt_template
            )

        results["answer"] = answer

        # Add this exchange to chat history if session_id is provided
        if session_id:
            self.add_to_chat_history(session_id, query, answer)

        return results


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="OpenTitan RAG System")
    parser.add_argument("--docs_dir", default="raptor_output",
                        help="Directory containing document index files")
    parser.add_argument("--query", required=True,
                        help="The user query to process")
    parser.add_argument("--translate", action="store_true",
                        help="Use query translation for complex queries")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of documents to retrieve per query")
    parser.add_argument("--top_sources", type=int, default=7,
                        help="Number of top sources to include in final context")
    parser.add_argument("--include_clusters", action="store_true", default=True,
                        help="Include cluster summaries in retrieval")
    parser.add_argument("--model", default="claude-3-opus-20240229",
                        help="Claude model to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed retrieval results")
    parser.add_argument("--session_id", default=None,
                        help="Session ID for maintaining chat history")
    parser.add_argument("--base_url", default="https://opentitan.org/book/",
                        help="Base URL for source document links")

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY=your-api-key")
        return

    # Initialize and run
    rag = OpenTitanRAG(args.docs_dir, model=args.model, base_url=args.base_url)
    results = rag.process_query(
        query=args.query,
        session_id=args.session_id,
        use_query_translation=args.translate,
        top_k=args.top_k,
        top_sources=args.top_sources,
        include_clusters=args.include_clusters,
        verbose=args.verbose
    )

    # Print results
    print("\n" + "="*80)

    if args.translate:
        print("QUERY BREAKDOWN:")
        for i, sub_query in enumerate(results["sub_queries"]):
            print(f"{i+1}. {sub_query}")

        print("\n" + "="*80)
        print("RETRIEVAL STATISTICS:")
        print(
            f"- Total documents retrieved: {results['retrieval_stats']['total_docs']}")
        print(
            f"- Unique sources: {results['retrieval_stats']['unique_sources']}")
        print("\nTop sources by frequency:")

        for source, count in sorted(
            results['retrieval_stats']['source_frequencies'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            print(f"- {source}: {count} occurrences")
    else:
        if args.verbose:
            print("RETRIEVED DOCUMENTS:")
            for i, doc in enumerate(results["documents"][:5]):
                source = doc['metadata'].get(
                    'source', doc['metadata'].get('node_id', 'Unknown'))
                print(f"{i+1}. Source: {source} (Score: {doc['score']:.4f})")
                print(f"   {doc['content'][:100]}...")

    print("\n" + "="*80)
    print("FINAL ANSWER:")
    print(results["answer"])
    print("="*80)


if __name__ == "__main__":
    main()
