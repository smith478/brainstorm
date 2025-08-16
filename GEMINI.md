# Project Overview

This project, "Brainstorm," is an AI-powered learning assistant designed to help users keep their technical skills sharp, particularly in the fast-evolving field of AI. It runs locally, leveraging Ollama to interact with various large language models. The primary interface is a command-line application (`main.py`) that allows users to search for recent research papers, take on coding challenges, and discuss AI topics.

# Key Features

*   **Impactful Paper Discovery:** Fetches and ranks recent AI papers from arXiv based on user interests and social media mentions.
*   **Trend Discovery:** Discovers trending AI techniques and tools by analyzing web search results.
*   **Dynamic Coding Challenges:** Generates new coding challenges on demand using an LLM, with the ability to get hints and evaluate solutions.
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
4.  **Run the Application:** `python main.py`

# Project Structure

```
brainstorm/
├── main.py                 # CLI entry point
├── agent.py                # BrainstormAgent orchestrator
├── config.py               # Config class
├── llm_provider.py         # LLM initialization
├── tools/                  # Tool modules
│   ├── __init__.py
│   ├── paper_discovery.py
│   ├── coding_challenge.py
│   └── discussion.py
├── challenges/             # Directory for coding challenges
│   └── .gitkeep
├── papers_cache/           # Cached paper data
├── brainstorm_db/          # Vector database storage
├── requirements.txt        # Python dependencies
├── config.json             # Configuration (auto-created)
└── README.md               # Project README
└── GEMINI.md               # Notes for Gemini CLI
```

# Important Notes for Gemini CLI

*   The user's main goal is to have a tool that helps them stay up-to-date with AI and practice their skills.
*   The project has been refactored into a modular structure. The core logic is now split across several files:
    *   `main.py`: The command-line interface.
    *   `agent.py`: The main agent that orchestrates the tools. This file also contains the logic for trend discovery and social score calculation.
    *   `tools/`: Each file in this directory represents a specific capability (e.g., `paper_discovery.py`, `coding_challenge.py`).
*   The user is interested in expanding the project with new features, such as a multi-agent system and new learning domains (e.g., MLOps, infrastructure). The `README.md` contains a "Future Enhancements" section for ideas.
*   The user has chosen `uv` as the package manager. Stick to this choice for any future dependency management tasks.