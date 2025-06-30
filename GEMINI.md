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
    file_paths=["/path/to/your/document1.txt", "/path/to/your/document2.txt"],
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
