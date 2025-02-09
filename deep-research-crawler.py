import asyncio
from crawl4ai import *
from pathlib import Path


async def crawl_urls(urls, output_dir="results"):
    """Crawl multiple URLs asynchronously and save results"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    async with AsyncWebCrawler() as crawler:
        tasks = []
        for url in urls:
            task = crawler.arun(
                url=url.strip(),
                # Optional: Add custom extraction config
                extraction_strategy=ExtractionStrategy(
                    strategy=CombinedStrategy(
                        strategies=[
                            RegexExtraction(regex=r".*", content_group=0),
                            SemanticHtmlExtraction()
                        ]
                    )
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (url, result) in enumerate(zip(urls, results)):
            if isinstance(result, Exception):
                print(f"Error crawling {url}: {str(result)}")
                continue

            # Save raw content
            filename = f"result_{i + 1}.txt"
            with open(Path(output_dir) / filename, "w", encoding="utf-8") as f:
                f.write(f"URL: {url}\n")
                f.write(result.text)

            # Save markdown
            md_filename = f"result_{i + 1}.md"
            with open(Path(output_dir) / md_filename, "w", encoding="utf-8") as f:
                f.write(result.markdown)

            print(f"Saved results for {url} to {filename} and {md_filename}")


def read_urls_from_file(filename="urls.txt"):
    """Read URLs from a text file"""
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]


async def main():
    urls = read_urls_from_file()

    if not urls:
        print("No URLs found in urls.txt")
        return

    print(f"Found {len(urls)} URLs to crawl:")
    for url in urls:
        print(f" - {url}")

    await crawl_urls(urls)


if __name__ == "__main__":
    asyncio.run(main())