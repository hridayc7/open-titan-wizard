#!/usr/bin/env python3
import os
import torch
import pickle
import logging
import argparse
import re
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


class CodebaseRAG:
    """
    Codebase RAG system with optional chat history and source hyperlinking.
    """

    def __init__(
        self,
        docs_dir: str,
        api_key: str = None,
        model: str = "claude-3-7-sonnet-20250219",
        embedding_model: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",  # For codebase
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        base_url: str = "https://github.com/lowRISC/opentitan/tree/master/",
        max_history_length: int = 20
    ):
        """Initialize the Codebase RAG system"""
        self.docs_dir = docs_dir
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.embedding_model_name = embedding_model
        self.device = device
        self.base_url = base_url
        self.max_history_length = max_history_length

        # Initialize chat history storage
        self.chat_histories = {}

        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set as ANTHROPIC_API_KEY env variable")

        # Load embedding model for codebase vector store
        logger.info(f"Loading embedding model for codebase: {embedding_model}")
        self.embedding_model = SentenceTransformer(
            embedding_model, device=device)
        self.lc_embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
        )

        # Load embedding model for HTML/documentation vector store
        self.html_embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        logger.info(
            f"Loading embedding model for HTML/docs: {self.html_embedding_model_name}")
        self.html_embedding_model = SentenceTransformer(
            self.html_embedding_model_name, device=device)
        self.html_lc_embeddings = HuggingFaceEmbeddings(
            model_name=self.html_embedding_model_name,
            model_kwargs={"device": device},
        )

        # Load vector stores
        self.vector_store = self._load_vector_store(
            self.lc_embeddings, custom_dir=self.docs_dir)
        self.html_vector_store = self._load_vector_store(
            self.html_lc_embeddings, custom_dir="raptor_output")

        # Load metadata (optional)
        try:
            self.code_chunks = self._load_code_chunks()
            self.cluster_summaries = self._load_cluster_summaries()
        except Exception as e:
            logger.warning(f"Could not load document metadata: {e}")
            self.code_chunks = None
            self.cluster_summaries = {}

        logger.info("Codebase RAG system initialized successfully")

    def _load_code_chunks(self):
        """Load the code chunks from pickle file."""
        chunks_path = os.path.join(self.docs_dir, 'code_chunks.pkl')
        if not os.path.exists(chunks_path):
            logger.warning(f"Code chunks not found at {chunks_path}")
            return None

        logger.info(f"Loading code chunks from {chunks_path}")
        with open(chunks_path, 'rb') as f:
            return pickle.load(f)

    def _load_cluster_summaries(self):
        """Load the cluster summaries from pickle file."""
        summaries_path = os.path.join(self.docs_dir, 'cluster_summaries.pkl')
        node_summaries_path = os.path.join(self.docs_dir, 'node_summaries.pkl')

        if os.path.exists(summaries_path):
            logger.info(f"Loading cluster summaries from {summaries_path}")
            with open(summaries_path, 'rb') as f:
                return pickle.load(f)
        elif os.path.exists(node_summaries_path):
            logger.info(f"Loading node summaries from {node_summaries_path}")
            with open(node_summaries_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def _load_vector_store(self, embeddings, custom_dir: Optional[str] = None):
        """
        Load the FAISS vector store from a given directory using the provided embeddings.
        """
        docs_dir = custom_dir or self.docs_dir
        index_path = os.path.join(docs_dir, 'faiss_index')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        logger.info(f"Loading vector store from {index_path}")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    def _claude_generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text using Claude API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=temperature,
            system="You are a helpful expert assistant that provides accurate, concise answers based solely on the provided code context.",
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
        Create a URL for a source file based on its path.
        For all valid sources in the repository, create a proper GitHub URL.
        Skip .txt files entirely.
        """
        if source == "Unknown" or not source or source.endswith('.txt'):
            return "#"

        # Clean up the source path if needed
        if source.startswith('./') or source.startswith('/'):
            source = source.lstrip('./')

        # Create the GitHub URL - use 'tree/master' for better directory navigation
        return f"{self.base_url}tree/master/{source}"

    def _is_citable_source(self, source: str, doc_type: str) -> bool:
        """
        Determine if a source is citable. 
        Exclude .txt files from being citable.
        """
        # Immediately return False for .txt files
        if source.endswith('.txt'):
            return False

        # If it's a code result, it's always citable
        if doc_type == "code":
            return True

        # For HTML results, only cite if it has a proper source path
        if source and source != "Unknown" and not source.startswith('#'):
            return True

        return False

    def _add_missing_source_links(self, answer: str, source_to_url: Dict[int, str], valid_citation_sources: List[int]) -> str:
        """
        Add clickable URLs to any source references that don't have them.
        Remove references to non-citable sources.
        """
        # Find all source references without links: Source X or [Source X]
        source_pattern = r'(?:\[)?Source\s+(\d+)(?:\])?(?!\()'

        def replace_with_link(match):
            source_num = int(match.group(1))
            if source_num in valid_citation_sources and source_num in source_to_url:
                return f"[Source {source_num}]({source_to_url[source_num]})"
            else:
                # Replace references to non-citable sources with generic text
                return "[Information from context]"

        # Replace all source references with proper links or generalize invalid ones
        linked_answer = re.sub(source_pattern, replace_with_link, answer)
        return linked_answer

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_clusters: bool = True,
        verbose: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant documents from both codebase and HTML/documentation.
        """
        if self.vector_store is None or self.html_vector_store is None:
            raise ValueError("Vector stores not properly loaded")

        logger.info(f"Retrieving documents for query: {query}")

        def search(store):
            return store.similarity_search_with_score(
                query,
                k=top_k
            )

        code_results = [
            {'content': doc.page_content,
                'metadata': doc.metadata, 'score': float(score)}
            for doc, score in search(self.vector_store)
        ]

        html_results = [
            {'content': doc.page_content,
                'metadata': doc.metadata, 'score': float(score)}
            for doc, score in search(self.html_vector_store)
        ]

        if verbose:
            print(f"\nQuery: {query}")
            print("\n--- Code Results ---")
            for i, result in enumerate(code_results):
                source = result['metadata'].get('source', 'N/A')
                print(f"{i+1}. {source} (Score: {result['score']:.4f})")
                # Print URL that would be generated
                print(f"   URL: {self._create_source_url(source)}")

            print("\n--- HTML Results ---")
            for i, result in enumerate(html_results):
                source = result['metadata'].get('source', 'N/A')
                print(f"{i+1}. {source} (Score: {result['score']:.4f})")
                # Print URL that would be generated
                print(f"   URL: {self._create_source_url(source)}")

        return {"code": code_results, "html": html_results}

    def generate_answer(
        self,
        query: str,
        session_id: str = None,
        documents: Dict[str, List[Dict[str, Any]]] = None,
        top_sources: int = 7,
        prompt_template: Optional[str] = None
    ) -> str:
        logger.info(f"Generating answer for query: {query}")
        source_to_url = {}
        valid_citation_sources = []
        context = ""

        def add_docs(docs, label, doc_type):
            nonlocal context
            context += f"\n### {label} Context\n"
            for i, doc in enumerate(docs[:top_sources]):
                source = doc['metadata'].get(
                    'source', doc['metadata'].get('cluster_id', 'Unknown'))

                # Check if this source is citable
                is_citable = self._is_citable_source(source, doc_type)

                source_index = len(source_to_url) + 1
                source_url = self._create_source_url(source)
                source_to_url[source_index] = source_url

                if is_citable:
                    valid_citation_sources.append(source_index)
                    citation_mark = "[CITABLE]"
                else:
                    citation_mark = "[REFERENCE ONLY]"

                context += f"{citation_mark} [Source {source_index}: {source}]\n"
                context += f"URL: {source_url}\n{doc['content']}\n\n"

        add_docs(documents.get("code", []), "Codebase", "code")
        add_docs(documents.get("html", []), "Documentation", "html")

        chat_history = self.format_chat_history(
            session_id) if session_id else ""

        # List of valid citation sources
        valid_sources_list = ", ".join(
            [str(src) for src in valid_citation_sources])

        if prompt_template is None:
            prompt = f"""Answer the following question using both the codebase and the documentation context below. 

IMPORTANT: You may ONLY cite sources marked as [CITABLE] in your answer. These are sources: {valid_sources_list}.
You can use information from [REFERENCE ONLY] sources to inform your answer, but do not cite them directly or mention them at all.
When you reference a citable source, include the source number like this: [X].

{chat_history}

{context}

Current question: {query}
"""
        else:
            prompt = prompt_template.format(
                context=context,
                query=query,
                chat_history=chat_history,
                valid_sources_list=valid_sources_list
            )

        answer = self._claude_generate(prompt)

        # Add hyperlinks to any source references that don't have them
        answer = self._add_missing_source_links(
            answer, source_to_url, valid_citation_sources)

        return answer

    def process_query(
        self,
        query: str,
        session_id: str = None,
        top_k: int = 5,
        top_sources: int = 7,
        include_clusters: bool = True,
        prompt_template: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Process a query using the Codebase RAG system
        """
        logger.info(f"Processing query: {query}")
        results = {"query": query}

        # Step 1: Retrieve documents
        documents = self.retrieve(query, top_k, include_clusters, verbose)
        results["documents"] = documents

        answer = self.generate_answer(
            query=query,
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
        description="Codebase RAG System")
    parser.add_argument("--docs_dir", default="code_raptor_output",
                        help="Directory containing document index files")
    parser.add_argument("--query", required=True,
                        help="The user query to process")
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
    parser.add_argument("--base_url", default="https://github.com/lowRISC/opentitan/tree/master/",
                        help="Base URL for source document links")

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY=your-api-key")
        return

    # Initialize and run
    rag = CodebaseRAG(
        docs_dir=args.docs_dir,
        model=args.model,
        base_url=args.base_url
    )

    results = rag.process_query(
        query=args.query,
        session_id=args.session_id,
        top_k=args.top_k,
        top_sources=args.top_sources,
        include_clusters=args.include_clusters,
        verbose=args.verbose
    )

    # Print results
    print("\n" + "="*80)
    print("RETRIEVED DOCUMENTS:")
    print("\n--- Code Results ---")
    for i, doc in enumerate(results["documents"]["code"][:5]):
        source = doc['metadata'].get(
            'source', doc['metadata'].get('cluster_id', 'Unknown'))
        print(f"{i+1}. Source: {source} (Score: {doc['score']:.4f})")
        print(f"   URL: {rag._create_source_url(source)}")
        print(f"   {doc['content'][:100]}...")

    print("\n--- HTML Results ---")
    for i, doc in enumerate(results["documents"]["html"][:5]):
        source = doc['metadata'].get(
            'source', doc['metadata'].get('cluster_id', 'Unknown'))
        print(f"{i+1}. Source: {source} (Score: {doc['score']:.4f})")
        print(f"   URL: {rag._create_source_url(source)}")
        print(f"   {doc['content'][:100]}...")

    print("\n" + "="*80)
    print("FINAL ANSWER:")
    print(results["answer"])
    print("="*80)


if __name__ == "__main__":
    main()
