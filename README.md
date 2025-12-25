# Lexi-Agent ⚖️

A private, on-premise AI assistant designed for legal professionals. Lexi-Agent is a **sovereign AI** solution that ensures all sensitive data remains confidential. It is capable of drafting contracts, summarizing complex legal documents, and identifying relevant case law to assist in legal research.

-----

## ✨ Features

  * **On-Premise Privacy:** All operations are conducted locally, guaranteeing that sensitive legal data never leaves your environment.
  * **Comprehensive Legal Assistance:** Acts as a complete legal assistant, handling tasks from document analysis to drafting and research.
  * **Specialized AI Agents:** Utilizes a multi-agent system with a **Document Analyst Agent** for summaries, a **Case Law Agent** for research, and a **Drafting Agent** for creating legal documents.
  * **Domain-Specific Embeddings:** Employs a highly specialized legal embedding model fine-tuned on extensive case law and legal texts to ensure high accuracy in searches and analysis.
  * **Efficient Local-First Architecture:** Leverages an embedded database like **LanceDB** for fast, local data access, combined with the performance of **Rust** for core logic.

-----

## ⚙️ Tech Stack

  * **Core Logic:** Rust
  * **Local Database:** LanceDB
  * **Legal Reasoning Model:** Gemma-2 or a similarly capable open-source model
  * **Vector Search:** LanceDB's built-in capabilities

-----

## 🚀 Getting Started

### Prerequisites

  * Rust
  * Python 3.10+ (for model inference)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/saadsalmanakram/Lexi-Agent.git
    cd Lexi-Agent
    ```
2.  **Build the Rust core:**
    ```bash
    cargo build --release
    ```
3.  **Set up the local model:**
    Download and configure the fine-tuned Gemma-2 model and ensure the environment is set up for local inference.

### Configuration

Follow the configuration guides in the project's documentation to set up the data directories and model paths.

### Usage

Run the main executable to start the Lexi-Agent server. You can then interact with the assistant via a local API or CLI interface.

-----

