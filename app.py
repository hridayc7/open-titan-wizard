from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import uuid
from opentitan_rag import OpenTitanRAG
from codebase_rag import CodebaseRAG

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For secure sessions
CORS(app, supports_credentials=True)  # Enable CORS with credentials

# Initialize the RAG systems
# Directory for OpenTitan docs (conceptual content)
opentitan_docs_dir = "raptor_output"
codebase_docs_dir = "raptor_output_codebase"  # Directory for code documentation

# Base URLs for source links
opentitan_base_url = "https://opentitan.org/book/"
# Update this to your repo
codebase_base_url = "https://github.com/lowRISC/opentitan/"

# Initialize both RAG systems
lumos_rag = OpenTitanRAG(docs_dir=opentitan_docs_dir,
                         base_url=opentitan_base_url)

# Only initialize Revelio (CodebaseRAG) if the directory exists
try:
    revelio_rag = CodebaseRAG(
        docs_dir=codebase_docs_dir, base_url=codebase_base_url)
    has_revelio = True
except FileNotFoundError:
    has_revelio = False
    print(
        f"Warning: CodebaseRAG directory '{codebase_docs_dir}' not found. Revelio model disabled.")

# Store session_ids for users
sessions = {}


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    use_translation = data.get('use_translation', False)
    model = data.get('model', 'lumos')  # Default to Lumos if not specified

    # Get or create session_id
    session_id = data.get('session_id')

    # If no session_id provided or invalid, create a new one
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {'messages': []}

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Select the appropriate RAG system based on model
        if model.lower() == 'revelio':
            if not has_revelio:
                return jsonify({
                    "error": "Revelio model is not available. Please check if the code_raptor_output directory exists."
                }), 400

            # Use the codebase RAG for technical details
            results = revelio_rag.process_query(
                query=query,
                session_id=session_id,
                top_k=5,
                top_sources=7,
                verbose=False
            )
        else:  # Default to Lumos model (OpenTitan conceptual RAG)
            results = lumos_rag.process_query(
                query=query,
                session_id=session_id,
                use_query_translation=use_translation,
                top_k=5,
                top_sources=7
            )

        # Store messages in session
        sessions[session_id]['messages'].append({
            'query': query,
            'answer': results["answer"],
            'model': model
        })

        response = {
            "answer": results["answer"],
            "session_id": session_id,  # Return session_id to client
            "model": model  # Return which model was used
        }

        # Include sub-queries if translation was used (only available for Lumos/OpenTitan)
        if use_translation and model.lower() == 'lumos' and "sub_queries" in results:
            response["sub_queries"] = results["sub_queries"]
            response["retrieval_stats"] = results["retrieval_stats"]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    data = request.json
    session_id = data.get('session_id')
    model = data.get('model', 'lumos')  # Which model's history to clear

    if session_id and session_id in sessions:
        # Clear session data
        sessions[session_id]['messages'] = []

        # Clear chat history in both RAG systems to ensure consistency
        lumos_rag.clear_chat_history(session_id)

        if has_revelio:
            revelio_rag.clear_chat_history(session_id)

        return jsonify({"status": "success", "message": "Chat history cleared"})

    return jsonify({"status": "success", "message": "Session not found, created new session"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5001)
