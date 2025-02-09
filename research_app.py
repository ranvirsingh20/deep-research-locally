import streamlit as st
import aiohttp
import asyncio
from crawl4ai import AsyncWebCrawler
import ssl
import certifi
import ollama
import time
from datetime import datetime

try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False

ssl_context = ssl.create_default_context(cafile=certifi.where())

# Constants
SERPAPI_API_KEY = "your-searchapi-key"
SERPAPI_URL = "https://serpapi.com/search.json"

# Streamlit Configuration
st.set_page_config(page_title="AI Research Assistant", layout="wide", initial_sidebar_state="expanded")

# Custom CSS with animations
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Base styles */
        * {{
            font-family: 'Inter', sans-serif;
        }}

        /* Main container */
        .main {{
            background: #0F172A;
            color: white;
        }}

        /* Sidebar styling */
        .sidebar .sidebar-content {{
            background: #1E293B;
            padding: 2rem;
            border-right: 1px solid #334155;
            overflow-y: auto; /* Make sidebar scrollable */
            height: calc(100vh - 4rem); /* Adjust height to fit screen */
        }}

        /* Sidebar text adjustments */
        .sidebar .sidebar-content * {{
            word-wrap: break-word; /* Ensure text wraps */
            white-space: normal; /* Prevent text overflow */
            overflow-x: hidden; /* Prevent horizontal scrolling */
        }}

        /* Step indicators */
        .step-container {{
            display: flex;
            flex-wrap: wrap; /* Allow steps to wrap to the next line */
            gap: 0.5rem;
            margin: 1rem 0;
        }}

        .step {{
            padding: 0.5rem 1rem;
            border-radius: 8px;
            background: #334155;
            color: #94A3B8;
            transition: all 0.3s ease;
            flex: 1 1 auto; /* Allow steps to grow and shrink */
            text-align: center;
        }}

        .step.active {{
            background: #4F46E5;
            color: white;
            box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3);
        }}

        /* Pulsing animation */
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}

        /* Status indicator */
        .status-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4F46E5;
            animation: pulse 2s infinite;
            margin-right: 8px;
        }}

        /* Progress bar container */
        .progress-container {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: #1E293B;
            z-index: 999;
        }}

        .progress-bar {{
            height: 100%;
            background: #4F46E5;
            transition: width 0.3s ease;
        }}

        /* Response styling */
        .ai-response {{
            background: #1E293B;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border: 1px solid #334155;
        }}

        /* Typing effect */
        @keyframes typing {{
            from {{ width: 0; }}
            to {{ width: 100%; }}
        }}

        .typing-effect {{
            overflow: hidden;
            white-space: pre-wrap; /* Preserve line breaks */
            animation: typing 2s steps(40, end);
        }}

        /* Thinking animation */
        @keyframes thinking {{
            0% {{ opacity: 0; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0; }}
        }}

        .thinking-dots {{
            display: flex;
            gap: 4px;
        }}

        .thinking-dots span {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4F46E5;
            animation: thinking 1.5s infinite;
        }}

        .thinking-dots span:nth-child(2) {{
            animation-delay: 0.5s;
        }}

        .thinking-dots span:nth-child(3) {{
            animation-delay: 1s;
        }}

        /* Copy button styling */
        .copy-button {{
            background: #4F46E5;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}

        .copy-button:hover {{
            background: #4338CA;
        }}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.update({
        'current_step': 0,
        'progress_messages': [],
        'start_time': None,
        'scraped_data': {},
        'found_urls': [],
        'final_answer': "",
        'is_generating': False
    })

# Sidebar setup
with st.sidebar:
    st.markdown("## Research Progress")
    progress_placeholder = st.empty()

    if st.session_state.found_urls:
        with st.expander("Discovered Links", expanded=True):
            st.markdown("\n".join([f"• [{url}]({url})" for url in st.session_state.found_urls]))

    if st.session_state.scraped_data:
        with st.expander("Scraped Content Preview", expanded=False):
            for url, content in st.session_state.scraped_data.items():
                st.markdown(f"**{url}**")
                st.code(content['full_content'], language="markdown")

    st.markdown("### Model Activity")
    status_placeholder = st.empty()

# Main content
st.title("🔍 AI Research Assistant")
query = st.text_input("Research topic", placeholder="The future of AI in healthcare...")

# Progress bar
st.markdown("""
    <div class="progress-container">
        <div class="progress-bar" id="progress" style="width: 0%"></div>
    </div>
""", unsafe_allow_html=True)


def update_progress(percentage):
    st.write(f"""
        <script>
            document.getElementById('progress').style.width = "{percentage}%";
        </script>
    """, unsafe_allow_html=True)


def add_progress_message(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.progress_messages.append(f"{timestamp} - {message}")
    update_status()


def update_status():
    with st.sidebar:
        status_placeholder.markdown("""
            <div style='border-left: 2px solid #4F46E5; padding-left: 1rem; margin: 1rem 0;'>
                <h4 style='color: #94A3B8; margin-bottom: 0.5rem;'>Process Log</h4>
                {}
            </div>
        """.format("\n".join([
            f"<div style='margin: 0.5rem 0;'><span class='status-indicator'></span>{msg}</div>"
            for msg in reversed(st.session_state.progress_messages[-5:])
        ])), unsafe_allow_html=True)


async def perform_search_async(session, query):
    params = {"q": query, "api_key": SERPAPI_API_KEY, "engine": "google", "num": 10}
    async with session.get(SERPAPI_URL, params=params) as resp:
        results = await resp.json()
        return [item.get("link") for item in results.get("organic_results", [])][:10]


async def scrape_websites(urls):
    results = {}
    async with AsyncWebCrawler(
            extract_blocks=True,  # Extract structured blocks
            parse_tags=["html", "body", "p", "div", "article", "main", "section", "span", "h1", "h2", "h3", "h4", "h5",
                        "h6", "header", "footer", "ul", "ol", "li", "blockquote"],
            include_images=False,
            include_links=False,
            render_js=True,
            timeout=60,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }
    ) as crawler:
        for idx, url in enumerate(urls):
            try:
                add_progress_message(f"🌐 Scraping {url}...")
                start = time.time()

                # Run the crawler
                result = await crawler.arun(url=url)
                elapsed = time.time() - start

                # Extract full content
                content = result.markdown

                # Store the full content
                results[url] = {
                    "full_content": content,
                }

                add_progress_message(f"✅ Scraped {url} ({len(content)} chars, {elapsed:.1f}s)")
                update_progress(40 + int((idx + 1) / len(urls) * 30))

            except Exception as e:
                results[url] = {
                    "full_content": f"❌ Failed to scrape: {str(e)}",
                }
                add_progress_message(f"❌ Failed {url}: {str(e)}")

            await asyncio.sleep(1)  # Delay to avoid rate limits
    return results


def generate_answer(prompt):
    response = ollama.chat(
        model="llama3.2:latest",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7}
    )
    return response["message"]["content"]


def generate_research_paper(scraped_data):

    introduction = generate_answer(f"Write a detailed introduction for a research paper about {query}. Include relevant background information and context.")
    methodology = generate_answer(f"Describe the methodology used to gather data for a research paper about {query}. Be specific about the sources and techniques.")
    results = generate_answer(f"Summarize the key findings from the scraped data about {query}. Include specific data points, quotes, and references.")
    discussion = generate_answer(f"Discuss the implications of the findings about {query}. Analyze the significance and potential impact.")
    conclusion = generate_answer(f"Write a comprehensive conclusion for a research paper about {query}. Summarize the findings and suggest future research directions.")

    appendix = f"""
        ## Appendix

        ### References
        {"\n".join([f"- [{url}]({url})" for url in scraped_data.keys()])}

        ### Author
        - **Generated by**: AI Research Assistant
        - **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """

    research_paper = f"""
        # Research Paper: {query}

        ## Introduction
        {introduction}

        ## Methodology
        {methodology}

        ## Results
        {results}

        ## Discussion
        {discussion}

        ## Conclusion
        {conclusion}

        {appendix}
    """
    return research_paper


def simulate_typing_effect(text, placeholder):
    placeholder.markdown(f"""
        <div class='ai-response'>
            <h3 style='color: #4F46E5; margin-bottom: 1rem;'>Research Report: {query}</h3>
            <div style='color: #94A3B8; margin-bottom: 1rem;'>
                ⏱️ Generated in {time.time() - st.session_state.start_time:.1f} seconds from {len(st.session_state.found_urls)} sources
            </div>
            <div class='typing-effect'>{text}</div>
        </div>
    """, unsafe_allow_html=True)


def copy_to_clipboard(text):
    if HAS_PYPERCLIP:
        pyperclip.copy(text)
        st.toast("Copied to clipboard!", icon="✅")
    else:
        st.code(text, language="markdown")
        st.toast("Copy the text manually from the code block above.", icon="ℹ️")


if st.button("Start Research"):
    if query.strip():
        st.session_state.update({
            'current_step': 0,
            'progress_messages': [],
            'start_time': time.time(),
            'scraped_data': {},
            'found_urls': [],
            'final_answer': "",
            'is_generating': False
        })

        async def main():
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                # Phase 1: Search
                st.session_state.current_step = 1
                add_progress_message("🔍 Starting web search...")
                update_progress(10)

                urls = await perform_search_async(session, query)
                st.session_state.found_urls = urls

                if not urls:
                    add_progress_message("⚠️ No results found")
                    return

                add_progress_message(f"✅ Found {len(urls)} URLs")
                update_progress(30)

                # Phase 2: Scraping
                st.session_state.current_step = 2
                scraped_data = await scrape_websites(urls)
                st.session_state.scraped_data = scraped_data
                update_progress(70)

                st.session_state.current_step = 3
                add_progress_message("🧠 Starting content analysis...")

                # Generate research paper
                research_paper = generate_research_paper(scraped_data)
                st.session_state.final_answer = research_paper
                add_progress_message("✅ Research paper generated")
                update_progress(95)

                st.session_state.current_step = 5
                total_time = time.time() - st.session_state.start_time
                add_progress_message(f"⏱️ Total processing time: {total_time:.1f} seconds")
                update_progress(100)

                placeholder = st.empty()
                simulate_typing_effect(research_paper, placeholder)

        asyncio.run(main())
    else:
        st.warning("Please enter a research topic")

if 'current_step' not in st.session_state:
    st.session_state.update({
        'current_step': 0,
        'progress_messages': [],
        'start_time': None,
        'scraped_data': {},
        'found_urls': [],
        'final_answer': "",
        'is_generating': False
    })

with st.sidebar:
    st.markdown("## Research Progress")
    progress_placeholder = st.empty()

    if st.session_state.found_urls:
        with st.expander("Discovered Links", expanded=True):
            st.markdown("\n".join([f"• [{url}]({url})" for url in st.session_state.found_urls]))

    if st.session_state.scraped_data:
        with st.expander("Scraped Content Preview", expanded=False):
            for url, content in st.session_state.scraped_data.items():
                st.markdown(f"**{url}**")
                st.code(content['full_content'], language="markdown")

    st.markdown("### Model Activity")
    status_placeholder = st.empty()
