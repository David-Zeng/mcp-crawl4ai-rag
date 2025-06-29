
"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
import psycopg2
import asyncpg
from urllib.parse import urlparse
import openai
import re
import time

# Load OpenAI API key for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_db_conn():
    """
    Get a PostgreSQL database connection.
    
    Returns:
        psycopg2 connection object
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

async def get_async_db_conn():
    """
    Get an async PostgreSQL database connection.
    
    Returns:
        asyncpg connection object
    """
    return await asyncpg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay
    
    for retry in range(max_retries):
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small", # Hardcoding embedding model for now, will change this later to be more dynamic
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                # Try creating embeddings one by one as fallback
                print("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0
                
                for i, text in enumerate(texts):
                    try:
                        individual_response = openai.embeddings.create(
                            model="text-embedding-3-small",
                            input=[text]
                        )
                        embeddings.append(individual_response.data[0].embedding)
                        successful_count += 1
                    except Exception as individual_error:
                        print(f"Failed to create embedding for text {i}: {individual_error}")
                        # Add zero embedding as fallback
                        embeddings.append([0.0] * 1536)
                
                print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                return embeddings

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536

def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Extract the generated context
        context = response.choices[0].message.content.strip()
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)

def add_documents_to_db(
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> None:
    """
    Add documents to the crawled_pages table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.
    
    Args:
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))
    
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Delete existing records for these URLs in a single operation
            try:
                if unique_urls:
                    cur.execute("DELETE FROM crawled_pages WHERE url = ANY(%s)", (unique_urls,))
            except Exception as e:
                print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
                conn.rollback()
                # Fallback: delete records one by one
                for url in unique_urls:
                    try:
                        cur.execute("DELETE FROM crawled_pages WHERE url = %s", (url,))
                    except Exception as inner_e:
                        print(f"Error deleting record for URL {url}: {inner_e}")
                        conn.rollback()
                        # Continue with the next URL even if one fails
            
            # Check if MODEL_CHOICE is set for contextual embeddings
            use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
            print(f"\n\nUse contextual embeddings: {use_contextual_embeddings}\n\n")
            
            # Process in batches to avoid memory issues
            for i in range(0, len(contents), batch_size):
                batch_end = min(i + batch_size, len(contents))
                
                # Get batch slices
                batch_urls = urls[i:batch_end]
                batch_chunk_numbers = chunk_numbers[i:batch_end]
                batch_contents = contents[i:batch_end]
                batch_metadatas = metadatas[i:batch_end]
                
                # Apply contextual embedding to each chunk if MODEL_CHOICE is set
                if use_contextual_embeddings:
                    # Prepare arguments for parallel processing
                    process_args = []
                    for j, content in enumerate(batch_contents):
                        url = batch_urls[j]
                        full_document = url_to_full_document.get(url, "")
                        process_args.append((url, content, full_document))
                    
                    # Process in parallel using ThreadPoolExecutor
                    contextual_contents = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        # Submit all tasks and collect results
                        future_to_idx = {executor.submit(process_chunk_with_context, arg): idx 
                                        for idx, arg in enumerate(process_args)}
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            try:
                                result, success = future.result()
                                contextual_contents.append(result)
                                if success:
                                    batch_metadatas[idx]["contextual_embedding"] = True
                            except Exception as e:
                                print(f"Error processing chunk {idx}: {e}")
                                # Use original content as fallback
                                contextual_contents.append(batch_contents[idx])
                    
                    # Sort results back into original order if needed
                    if len(contextual_contents) != len(batch_contents):
                        print(f"Warning: Expected {len(batch_contents)} results but got {len(contextual_contents)}")
                        # Use original contents as fallback
                        contextual_contents = batch_contents
                else:
                    # If not using contextual embeddings, use original contents
                    contextual_contents = batch_contents
                
                # Create embeddings for the entire batch at once
                batch_embeddings = create_embeddings_batch(contextual_contents)
                
                batch_data = []
                for j in range(len(contextual_contents)):
                    # Extract metadata fields
                    chunk_size = len(contextual_contents[j])
                    
                    # Extract source_id from URL
                    parsed_url = urlparse(batch_urls[j])
                    source_id = parsed_url.netloc or parsed_url.path
                    
                    # Prepare data for insertion
                    data = (
                        batch_urls[j],
                        batch_chunk_numbers[j],
                        contextual_contents[j],
                        json.dumps({
                            "chunk_size": chunk_size,
                            **batch_metadatas[j]
                        }),
                        source_id,
                        batch_embeddings[j]
                    )
                    
                    batch_data.append(data)
                
                # Insert batch into the database
                try:
                    cur.executemany(
                        "INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
                        batch_data
                    )
                    conn.commit()
                except Exception as e:
                    print(f"Error inserting batch into the database: {e}")
                    conn.rollback()
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            cur.execute(
                                "INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
                                record
                            )
                            conn.commit()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record for URL {record[0]}: {individual_error}")
                            conn.rollback()
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually")

async def search_documents(
    query: str, 
    match_count: int = 10, 
    match_threshold: float = 0.5, # Added match_threshold
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in the database using vector similarity.
    
    Args:
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        
    Returns:
        List of matching documents
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)
    
    conn = await get_async_db_conn()
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        # Prepare the filter (jsonb) and source_filter (text) arguments for the SQL function
        sql_filter_jsonb = json.dumps(filter_metadata) if filter_metadata else '{}'
        sql_source_filter_text = None # Default to None (SQL NULL)

        if filter_metadata and 'source' in filter_metadata:
            sql_source_filter_text = filter_metadata['source']

        # The params list for asyncpg.fetch should match the SQL function's arguments
        # match_crawled_pages(query_embedding vector, match_count int, filter jsonb, source_filter text)
        params = [str(query_embedding), match_count, sql_filter_jsonb, sql_source_filter_text]
        sql_query = 'SELECT * FROM match_crawled_pages($1, $2, $3, $4)'

        
        rows = await conn.fetch(sql_query, *params)
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []
    finally:
        await conn.close()


def extract_code_blocks(markdown_content: str, min_length: int = 1000) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.
    
    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)
        
    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []
    
    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith('```'):
        # Skip the first triple backticks
        start_offset = 3
        print("Skipping initial triple backticks")
    
    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find('```', pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3
    
    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]
        
        # Extract the content between backticks
        code_section = markdown_content[start_pos+3:end_pos]
        
        # Check if there's a language specifier on the first line
        lines = code_section.split('\n', 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and not ' ' in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()
        
        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue
        
        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()
        
        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3:context_end].strip()
        
        code_blocks.append({
            'code': code_content,
            'language': language,
            'context_before': context_before,
            'context_after': context_after,
            'full_context': f"{context_before}\n\n{code_content}\n\n{context_after}"
        })
        
        # Move to next pair (skip the closing backtick we just processed)
        i += 2
    
    return code_blocks


def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    """
    Generate a summary for a code example using its surrounding context.
    
    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        
    Returns:
        A summary of what the code example demonstrates
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""
    
    try:
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."


def add_code_examples_to_db(
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20
):
    """
    Add code examples to the code_examples table in batches.
    
    Args:
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return
        
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Delete existing records for these URLs
            unique_urls = list(set(urls))
            for url in unique_urls:
                try:
                    cur.execute("DELETE FROM code_examples WHERE url = %s", (url,))
                except Exception as e:
                    print(f"Error deleting existing code examples for {url}: {e}")
                    conn.rollback()
            
            # Process in batches
            total_items = len(urls)
            for i in range(0, total_items, batch_size):
                batch_end = min(i + batch_size, total_items)
                batch_texts = []
                
                # Create combined texts for embedding (code + summary)
                for j in range(i, batch_end):
                    combined_text = f"{code_examples[j]}\n\nSummary: {summaries[j]}"
                    batch_texts.append(combined_text)
                
                # Create embeddings for the batch
                embeddings = create_embeddings_batch(batch_texts)
                
                # Check if embeddings are valid (not all zeros)
                valid_embeddings = []
                for embedding in embeddings:
                    if embedding and not all(v == 0.0 for v in embedding):
                        valid_embeddings.append(embedding)
                    else:
                        print(f"Warning: Zero or invalid embedding detected, creating new one...")
                        # Try to create a single embedding as fallback
                        single_embedding = create_embedding(batch_texts[len(valid_embeddings)])
                        valid_embeddings.append(single_embedding)
                
                # Prepare batch data
                batch_data = []
                for j, embedding in enumerate(valid_embeddings):
                    idx = i + j
                    
                    # Extract source_id from URL
                    parsed_url = urlparse(urls[idx])
                    source_id = parsed_url.netloc or parsed_url.path
                    
                    batch_data.append((
                        urls[idx],
                        chunk_numbers[idx],
                        code_examples[idx],
                        summaries[idx],
                        json.dumps(metadatas[idx]),
                        source_id,
                        embedding
                    ))
                
                # Insert batch into the database
                try:
                    cur.executemany(
                        "INSERT INTO code_examples (url, chunk_number, content, summary, metadata, source_id, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        batch_data
                    )
                    conn.commit()
                except Exception as e:
                    print(f"Error inserting batch into the database: {e}")
                    conn.rollback()
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            cur.execute(
                                "INSERT INTO code_examples (url, chunk_number, content, summary, metadata, source_id, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                                record
                            )
                            conn.commit()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record for URL {record[0]}: {individual_error}")
                            conn.rollback()
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually")
                print(f"Inserted batch {i//batch_size + 1} of {(total_items + batch_size - 1)//batch_size} code examples")


def update_source_info(source_id: str, summary: str, word_count: int):
    """
    Update or insert source information in the sources table.
    
    Args:
        source_id: The source ID (domain)
        summary: Summary of the source
        word_count: Total word count for the source
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            try:
                # Try to update existing source
                cur.execute(
                    "UPDATE sources SET summary = %s, total_word_count = %s, updated_at = now() WHERE source_id = %s",
                    (summary, word_count, source_id)
                )
                
                # If no rows were updated, insert new source
                if cur.rowcount == 0:
                    cur.execute(
                        "INSERT INTO sources (source_id, summary, total_word_count) VALUES (%s, %s, %s)",
                        (source_id, summary, word_count)
                    )
                    print(f"Created new source: {source_id}")
                else:
                    print(f"Updated source: {source_id}")
                conn.commit()
            except Exception as e:
                print(f"Error updating source {source_id}: {e}")
                conn.rollback()


def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using an LLM.
    
    This function uses the OpenAI API to generate a concise summary of the source content.
    
    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary
        
    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"
    
    if not content or len(content.strip()) == 0:
        return default_summary
    
    # Get the model choice from environment variables
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content
    
    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""
    
    try:
        # Call the OpenAI API to generate the summary
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise library/tool/framework summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        # Extract the generated summary
        summary = response.choices[0].message.content.strip()
        
        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary
    
    except Exception as e:
        print(f"Error generating summary with LLM for {source_id}: {e}. Using default summary.")
        return default_summary


async def search_code_examples(
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for code examples in the database using vector similarity.
    
    Args:
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID to filter results
        
    Returns:
        List of matching code examples
    """
    # Create a more descriptive query for better embedding match
    # Since code examples are embedded with their summaries, we should make the query more descriptive
    enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
    
    # Create embedding for the enhanced query
    query_embedding = create_embedding(enhanced_query)
    
    conn = await get_async_db_conn()
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = [str(query_embedding), match_count]
        sql_query = 'SELECT * FROM match_code_examples($1, $2, $3, $4)'
        
        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params.append(json.dumps(filter_metadata))
        else:
            params.append('{}')
            
        # Add source filter if provided
        if source_id:
            params.append(source_id)
        else:
            params.append(None)

        
        rows = await conn.fetch(sql_query, *params)
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"Error searching code examples: {e}")
        return []
    finally:
        await conn.close()
