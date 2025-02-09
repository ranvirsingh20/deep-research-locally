import asyncio
import aiohttp
import os
from pathlib import Path

# =======================
# Configuration Constants
# =======================
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:latest"  # Change to your preferred model
LOCAL_KNOWLEDGE_DIR = "local_knowledge"  # Directory containing text files for local "search"


# ============================
# Asynchronous Helper Functions
# ============================

async def call_ollama_async(session, messages, model=DEFAULT_MODEL):
    """Asynchronously call the local Ollama API"""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result['message']['content']
            else:
                text = await resp.text()
                print(f"Ollama API error: {resp.status} - {text}")
                return None
    except Exception as e:
        print("Error calling Ollama:", e)
        return None


async def generate_search_queries_async(session, user_query):
    """Generate search queries using local LLM"""
    prompt = (
        "You are an expert research assistant. Generate 3-4 distinct search queries "
        "that would help gather comprehensive information on the topic. "
        "Return only a Python list of strings."
    )
    messages = [
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
    ]
    response = await call_ollama_async(session, messages)

    if response:
        try:
            return eval(response)
        except:
            print("Failed to parse search queries, using fallback")
            return [user_query]
    return [user_query]


async def local_search_async(query):
    """Search local knowledge base (text files in directory)"""
    results = []
    knowledge_path = Path(LOCAL_KNOWLEDGE_DIR)

    if not knowledge_path.exists():
        print(f"Knowledge directory {LOCAL_KNOWLEDGE_DIR} not found!")
        return results

    for file_path in knowledge_path.glob("*.txt"):
        try:
            content = file_path.read_text(encoding="utf-8")
            if query.lower() in content.lower():
                results.append({
                    "path": str(file_path),
                    "content": content
                })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return results


async def is_content_useful_async(session, user_query, content):
    """Determine if content is useful using local LLM"""
    prompt = (
        "Determine if this content is relevant to the user's query. "
        "Respond with exactly 'Yes' or 'No'."
    )
    messages = [
        {"role": "user", "content": f"Query: {user_query}\nContent: {content[:20000]}\n{prompt}"}
    ]
    response = await call_ollama_async(session, messages)
    return "Yes" if response and "Yes" in response else "No"


async def extract_relevant_context_async(session, user_query, content):
    """Extract relevant context using local LLM"""
    prompt = (
        "Extract information relevant to the user's query from this content. "
        "Return only the relevant context as plain text."
    )
    messages = [
        {"role": "user", "content": f"Query: {user_query}\nContent: {content[:20000]}\n{prompt}"}
    ]
    return await call_ollama_async(session, messages)


async def get_new_search_queries_async(session, user_query, previous_queries, contexts):
    """Determine if new searches are needed using local LLM"""
    prompt = (
        "Based on the research so far, should we search more? "
        "Respond with a Python list of new queries or <done>."
    )
    messages = [
        {"role": "user",
         "content": f"Query: {user_query}\nPrevious Queries: {previous_queries}\nContext: {contexts}\n{prompt}"}
    ]
    response = await call_ollama_async(session, messages)

    if response:
        if response.startswith("["):
            try:
                return eval(response)
            except:
                return []
        elif "<done>" in response:
            return "<done>"
    return []


async def generate_final_report_async(session, user_query, contexts):
    """Generate final report using local LLM"""
    prompt = "Write a comprehensive report based on this research:"
    messages = [
        {"role": "user", "content": f"Query: {user_query}\nResearch: {' '.join(contexts)}\n{prompt}"}
    ]
    return await call_ollama_async(session, messages)


# =========================
# Main Processing Functions
# =========================

async def process_content(session, user_query, content_item):
    """Process a single content item"""
    print(f"Analyzing: {content_item['path']}")
    usefulness = await is_content_useful_async(session, user_query, content_item['content'])
    print(f"Usefulness: {usefulness}")

    if usefulness == "Yes":
        context = await extract_relevant_context_async(session, user_query, content_item['content'])
        if context:
            print(f"Extracted context (first 200 chars): {context[:200]}")
            return context
    return None


async def async_main():
    user_query = input("Enter your research query/topic: ").strip()
    iteration_limit = 3  # Maximum research iterations

    aggregated_contexts = []
    all_search_queries = []
    iteration = 0

    async with aiohttp.ClientSession() as session:
        # Initial search queries
        search_queries = await generate_search_queries_async(session, user_query)
        all_search_queries.extend(search_queries)

        while iteration < iteration_limit:
            print(f"\n=== Research Iteration {iteration + 1} ===")

            # Perform local searches for each query
            search_tasks = [local_search_async(query) for query in search_queries]
            search_results = await asyncio.gather(*search_tasks)

            # Process all found content
            content_items = [item for sublist in search_results for item in sublist]
            print(f"Found {len(content_items)} relevant documents")

            # Process content concurrently
            processing_tasks = [process_content(session, user_query, item) for item in content_items]
            results = await asyncio.gather(*processing_tasks)

            # Aggregate valid contexts
            new_contexts = [ctx for ctx in results if ctx]
            aggregated_contexts.extend(new_contexts)

            # Determine if more research is needed
            search_queries = await get_new_search_queries_async(
                session, user_query, all_search_queries, new_contexts
            )

            if search_queries == "<done>":
                print("Research complete!")
                break
            elif isinstance(search_queries, list) and search_queries:
                all_search_queries.extend(search_queries)
                print(f"New search queries: {search_queries}")
            else:
                print("No new search queries generated")
                break

            iteration += 1

        # Generate final report
        if aggregated_contexts:
            print("\nGenerating final report...")
            report = await generate_final_report_async(session, user_query, aggregated_contexts)
            print("\n==== FINAL REPORT ====\n")
            print(report)
        else:
            print("\nNo relevant information found for this query.")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    # Create knowledge directory if it doesn't exist
    if not os.path.exists(LOCAL_KNOWLEDGE_DIR):
        os.makedirs(LOCAL_KNOWLEDGE_DIR)
        print(f"Created {LOCAL_KNOWLEDGE_DIR} directory. Add some text files to it for research!")

    main()