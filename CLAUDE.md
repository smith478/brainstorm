# Project Overview

This project, "Brainstorm," is an AI-powered learning assistant designed to help users keep their technical skills sharp, particularly in the fast-evolving field of AI. It runs locally, leveraging Ollama to interact with various large language models. The primary interface is a command-line application (`brainstorm.py`) that allows users to search for recent research papers, take on coding challenges, and discuss AI topics.

# Key Features

*   **Paper Discovery:** Fetches and ranks recent AI papers from arXiv based on user interests.
*   **Coding Challenges:** Provides a set of coding challenges with varying difficulty levels (easy, medium, hard) and categories. It can also provide hints and evaluate user-submitted solutions using an LLM.
*   **AI Topic Discussion:** Utilizes a Retrieval-Augmented Generation (RAG) pipeline to discuss AI concepts. It indexes the fetched papers into a ChromaDB vector store to provide contextually relevant answers.
*   **Local First:** Everything runs locally using Ollama, ensuring privacy and offline capabilities.
*   **Configurable:** The application uses a `config.json` file to allow users to customize the LLM model, database paths, and other settings.

# Technology Stack

*   **Language:** Python 3
*   **AI Framework:** LangChain
*   **LLM Serving:** Ollama
*   **Vector Store:** ChromaDB
*   **Package Management:** uv
*   **Core Libraries:** `requests`, `beautifulsoup4`, `numpy`, `arxiv`

# How to Run

1.  **Install Ollama:** Follow the instructions in the `README.md` to install Ollama for your operating system.
2.  **Pull an Ollama Model:** Start the Ollama service and pull a model (e.g., `ollama pull llama3.2`).
3.  **Set up Python Environment:**
    *   Install `uv`: `pip install uv`
    *   Create a virtual environment: `uv venv`
    *   Activate the environment: `source .venv/bin/activate` (on macOS/Linux)
    *   Install dependencies: `uv pip install -r requirements.txt`
4.  **Run the Application:** `python brainstorm.py`

# Project Structure

```
brainstorm/
├── brainstorm.py          # Main agent code
├── requirements.txt        # Python dependencies
├── config.json            # Configuration (auto-created)
├── brainstorm_db/        # Vector database storage
│   └── papers_vectors/    # Indexed papers
├── challenges/            # Coding challenges
├── papers_cache/          # Cached paper data
└── README.md             # Project README
└── CLAUDE.md             # Notes for Claude CLI
```

# Important Notes for Claude CLI

*   The user's main goal is to have a tool that helps them stay up-to-date with AI and practice their skills.
*   The project is in its early stages, so the user might be interested in adding new features. The `README.md` contains a "Future Enhancements" section that you can refer to.
*   The user has chosen `uv` as the package manager. Stick to this choice for any future dependency management tasks.
*   The core logic is in the `brainstorm.py` file, which is well-structured with different classes for different functionalities. When adding new features, try to follow this modular approach.
