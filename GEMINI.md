# Project Changes and Key Information (GEMINI.md)

This document summarizes the modifications made to the `mcp-crawl4ai-rag` project and provides essential information for setting up and running the modified application.

## 1. Database Migration: Supabase to Local PostgreSQL

The project's database backend has been migrated from Supabase to a local PostgreSQL instance. This change was implemented to provide more control over the database environment and to allow for local development without relying on external services.

### Key Changes:

- Replaced `supabase-python` client with `psycopg2-binary` (for synchronous operations) and `asyncpg` (for asynchronous operations).
- Updated database connection logic in `src/utils.py` and `src/crawl4ai_mcp.py`.
- Modified environment variable names for database configuration to be more generic (e.g., `DB_HOST`, `DB_NAME`).

## 2. Database Setup for Local PostgreSQL

To use the project with your local PostgreSQL database, follow these steps:

### 2.1. Install `pgvector` Extension

Ensure that the `pgvector` extension is installed and enabled in your PostgreSQL database. This extension is crucial for vector similarity search, which is a core component of the RAG functionality.

You can check and attempt to install it using the provided script:

```bash
python check_pgvector.py
```

### 2.2. Create Database Schema

Execute the SQL commands found in `crawled_pages.sql` against your local PostgreSQL database. This will create the necessary tables (`sources`, `crawled_pages`, `code_examples`) and functions (`match_crawled_pages`, `match_code_examples`).

```bash
psql -h <DB_HOST> -p <DB_PORT> -U <DB_USER> -d <DB_NAME> -f crawled_pages.sql
```

(Replace `<DB_HOST>`, `<DB_PORT>`, `<DB_USER>`, `<DB_NAME>` with your actual database credentials.)

## 3. Environment Configuration (`.env` file)

Update your `.env` file (or create one from `.env.example`) with your local PostgreSQL database connection details. The relevant variables are:

```
# PostgreSQL Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
```

Make sure to also configure your `OPENAI_API_KEY` and other RAG strategy flags as needed.

## 4. Python Dependencies

Install the updated Python dependencies using `uv`:

```bash
uv pip install .
```

## 5. `check_pgvector.py` Script

A utility script `check_pgvector.py` has been created to help you verify the `pgvector` extension installation and attempt to install it if missing. This script uses your `.env` file for database connection details.

## 6. New MCP Tool: `insert_local_document`

A new MCP tool `insert_local_document` has been added to allow for the ingestion of content from local files into the RAG database or, with certain limitations, the knowledge graph.

### Usage:

```python
await mcp.tool().insert_local_document(
    file_path="/path/to/your/document.txt",
    document_type="text", # or "code"
    source_id="my_local_docs" # Optional custom source ID
)
```

### Important Considerations for `document_type='code'`:

Currently, the knowledge graph implementation is designed to parse entire GitHub _repositories_. Direct insertion of individual local code files into the knowledge graph is **not supported** by this tool. If you wish to add code to the knowledge graph, you should use the `parse_github_repository` tool with a GitHub repository URL.

## Next Steps

After completing the above setup, you should be able to run the `mcp-crawl4ai-rag` server, which will now connect to your local PostgreSQL database.

```bash
uv run src/crawl4ai_mcp.py
```

Remember to ensure your PostgreSQL server is running before starting the MCP server.

## Test Content from Session

### Knowledge Graph Insertion Test

The `pydantic-ai` repository was successfully parsed and inserted into the knowledge graph using the `parse_github_repository` tool.

```
{
  "success": true,
  "repo_url": "https://github.com/pydantic/pydantic-ai.git",
  "repo_name": "pydantic-ai",
  "message": "Successfully parsed repository 'pydantic-ai' into knowledge graph",
  "statistics": {
    "repository": "pydantic-ai",
    "files_processed": 115,
    "classes_created": 296,
    "methods_created": 525,
    "functions_created": 128,
    "attributes_created": 732,
    "sample_modules": [
      "pydantic_ai.models.gemini",
      "pydantic_graph.persistence.in_mem",
      "pydantic_graph.persistence._utils",
      "pydantic_graph.persistence.file",
      "pydantic_graph.persistence.__init__"
    ]
  },
  "ready_for_validation": true,
  "next_steps": [
    "Repository is now available for hallucination detection",
    "Use check_ai_script_hallucinations to validate scripts against pydantic-ai",
    "The knowledge graph contains classes, methods, and functions from this repository"
  ]
}
```

The `repos` command confirmed the presence of `pydantic-ai` in the knowledge graph.

```
{
  "success": true,
  "command": "repos",
  "data": {
    "repositories": [
      "pydantic-ai"
    ]
  },
  "metadata": {
    "total_results": 1,
    "limited": false
  }
}
```

### Key Leakage Risk Check

A search for common patterns indicating API keys, secrets, passwords, and tokens within `.py`, `.js`, `.ts`, `.env`, `.json`, `.yml`, `.yaml`, and `.sh` files in the repository yielded no matches. The `.env` file is correctly listed in `.gitignore`, preventing it from being committed to the repository.

### RAG Query Troubleshooting

Attempts to run a RAG query using `perform_rag_query` resulted in the error: `"could not determine data type of parameter $2"`. This error persisted even after modifying `src/crawl4ai_mcp.py` to adjust parameter indexing. The issue is likely related to how the embedding is passed to the PostgreSQL function, which expects a vector type, and the current setup might be converting it to a string.

A change was made to `src/utils.py` to pass the embedding directly as a list of floats to the PostgreSQL function. However, for this change to take effect, the MCP server needs to be restarted.