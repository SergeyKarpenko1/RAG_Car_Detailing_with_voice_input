import asyncio
import os.path

from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv

from config import BASE_URL, CSS_SELECTOR, REQUIRED_KEYS
from utils.data_utils import (
    save_venues_to_csv,
    save_venues_to_markdown,
)
from utils.scraper_utils import (
    fetch_and_process_page,
    get_browser_config,
    get_llm_strategy,
)

load_dotenv()


async def crawl_single_article():
    """
    Function to crawl a single article page without pagination.
    """
    # Initialize configurations
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy()
    session_id = "article_crawl_session"

    # Initialize state variables
    all_venues = []
    seen_names = set()

    # Start the web crawler context
    # https://docs.crawl4ai.com/api/async-webcrawler/#asyncwebcrawler
    async with AsyncWebCrawler(config=browser_config) as crawler:
        print("Processing article page...")
        
        # Process the article page once without pagination
        venues, _ = await fetch_and_process_page(
            crawler,
            1,  # page_number is always 1 for single articles
            BASE_URL,
            CSS_SELECTOR,
            llm_strategy,
            session_id,
            REQUIRED_KEYS,
            seen_names,
        )
        
        if venues:
            all_venues.extend(venues)
        else:
            print("No content extracted from the article.")

    # Generate filename based on the URL
    article_name = os.path.basename(BASE_URL.rstrip('/'))
    base_filename = article_name if article_name else "article_content"

    # Save the collected content to markdown and CSV files
    if all_venues:
        # Save to markdown
        md_filename = f"{base_filename}.md"
        save_venues_to_markdown(all_venues, md_filename)
        print(f"Saved {len(all_venues)} entries to '{md_filename}'.")
        
        # Also save to CSV for backward compatibility
        csv_filename = f"{base_filename}.csv"
        save_venues_to_csv(all_venues, csv_filename)
        print(f"Saved {len(all_venues)} entries to '{csv_filename}'.")
    else:
        print("No content was found during the crawl.")

    # Display usage statistics for the LLM strategy
    llm_strategy.show_usage()


async def main():
    """
    Entry point of the script.
    """
    await crawl_single_article()


if __name__ == "__main__":
    asyncio.run(main())