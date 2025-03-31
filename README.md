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

- **Modern UI**: Clean, responsive single-page application (Inspired by Claude)
- **State Management**: React hooks for state and localStorage for persistence
- **Markdown Support**: Real-time rendering of formatted text and code blocks
- **Chat Management**: History, session tracking, and conversation organization

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

3. Create a `.env` file (optional, for custom API URL): [OPTIONAL]
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

## 🔧 Things That Didn’t Work as Well (Needs Iteration)

### Sourcing

In both modes (Lumos and Revelio) — i.e. conceptual and codebase query modes — sourcing was not entirely accurate.

#### Lumos (Conceptual)
The reason sourcing fell off here was because I downloaded the files from the HTML rather than from the GitHub repository. When I did this, I extracted the raw text from each page into text files and converted links in the following format:

https://opentitan.org/book/hw/index.html → hw_index.txt
https://opentitan.org/book/hw/ip/otbn/doc/developing_otbn.html → hw_ip_otbn_doc_developing_otbn.txt


The issue here was when I had the thought of sourcing the data in my answers, my metadata only contained the file path directories I broke each link into, which was problematic.

If I had thought about sourcing in the beginning, I would have paid more attention to storing the URL in metadata when chunking the documents.

What I tried to do was build the source back up based on the file path, but I ran into the issue where the program wasn’t able to determine differences between terms that had organically had underscores and terms that manually had underscores (added in by me when downloading files). For instance, `developing_otbn` would become `developing/otbn` when reconstructing the path.

#### Revelio (Codebase)
This one hallucinated at some points by producing images for sources. I’m not entirely sure why. I had manually, naively cleaned up the GitHub repo before embedding it, and I thought I removed all image files prior to embedding, but I suppose I missed out on some. Still not sure why the image was returned — will have to take a look at the code.

Also, since Revelio still makes use of the HTML database, sourcing issues from the HTML database persist in this chat.

### Multi-Query / Advanced Reasoning

The idea behind Multi-Query / Advanced Reasoning was that we take our input query, break it up into multiple questions, and then we should hit more diverse parts — i.e. hit different leaves of the hierarchical database we built — when retrieving documents.

However, a mistake I made was that I weighed documents seen more frequently higher than others. This would be good if we had a question that was vague and needed consensus, but for questions that are very specific and would require diversity in answers, the weightage system at the end was not the best idea.

---

## Conclusion

I think there may be some more changes I would make to this project, based on my talk with David and some further introspection. Plus, I can always go back to the codebase and see points that could be easily sped up.  
But, I’m still proud of the work I put into this project, and it does get a lot of things correct as well. Only way is up 🙂


## Note to David and Jefferey
Hey guys, I really enjoyed working on this project and learned quite a bit. Hope you have as much fun using it as I did building it!
