# OpenTitan RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) chatbot application designed to answer questions about the OpenTitan project through an intelligent dual-query system that searches both conceptual documentation and source code.


## Features

- **Dual RAG System**: Two specialized models for different types of queries
  - **Lumos**: Conceptual model for documentation-based questions
  - **Revelio**: Technical model for codebase-specific queries
- **Conversation Memory**: Maintains chat history for context-aware responses
- **Clean UI**: Modern React interface with light/dark mode support
- **Session Management**: Create, rename, and organize multiple conversations

## System Architecture

### Backend (Python/Flask)

- **OpenTitanRAG Class**: Processes conceptual queries with OpenTitan documentation
- **CodebaseRAG Class**: Specialized for source code and technical implementation questions
- **Vector Storage**: FAISS indices for efficient semantic search
- **REST API**: Flask server with endpoints for chat and session management
- **Embedding Models**: Multiple embedding models optimized for different content types

### Frontend (React)

- **Modern UI**: Clean, responsive single-page application
- **State Management**: React hooks for state and localStorage for persistence
- **Markdown Support**: Real-time rendering of formatted text and code blocks
- **Chat Management**: History, session tracking, and conversation organization
- **Model Selection**: Easy switching between Lumos and Revelio models

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- Anthropic API key

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/opentitan-rag.git
   cd opentitan-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

5. Ensure you have the vector stores in the proper directories:
   - `raptor_output/` - For the OpenTitan documentation
   - `raptor_output_codebase/` - For the OpenTitan codebase

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file (optional, for custom API URL):
   ```
   REACT_APP_API_URL=http://localhost:5001
   ```

## Usage

1. Start the backend server:
   ```bash
   python app.py
   ```

2. In a separate terminal, start the frontend development server:
   ```bash
   cd frontend
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`

4. Choose between models:
   - **Lumos**: For conceptual questions about OpenTitan architecture, features, etc.
   - **Revelio**: For technical inquiries about code implementation details

## API Endpoints

- `POST /api/chat`: Send a query and receive an answer
  - Parameters: `query`, `session_id`, `use_translation`, `model`
  - Returns: Answer with optional sub-queries and session information

- `POST /api/clear-chat`: Clear the chat history for a session
  - Parameters: `session_id`, `model`

## Project Structure

```
.
├── app.py                       # Main Flask application
├── opentitan_rag.py             # Lumos RAG implementation (documentation)
├── codebase_rag.py              # Revelio RAG implementation (source code)
├── requirements.txt             # Python dependencies
├── raptor_output/               # Vector stores for documentation
│   ├── faiss_index/             # FAISS index files
│   ├── document_tree.pkl        # Document tree data
│   └── node_summaries.pkl       # Document summary data
├── raptor_output_codebase/      # Vector stores for code (optional)
│   └── faiss_index/             # FAISS index files
└── frontend/                    # React frontend application
    ├── public/                  # Static assets
    └── src/                     # React source code
        ├── App.js               # Main application component
        └── App.css              # Styling
```

## Note to David and Jefferey
Hey guys, I had a lot of fun and learned quite a bit with this project! Hope you have as much fun using it as I did building it!
