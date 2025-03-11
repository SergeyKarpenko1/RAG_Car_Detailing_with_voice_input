import asyncio
import os
import os.path

from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv

from config import (
    BLOG_BASE_URL,
    CSS_SELECTOR,
    REQUIRED_KEYS,
    MAX_BLOG_PAGES,
)
from utils.data_utils import (
    save_venues_to_csv,
    save_venues_to_markdown,
)
from utils.scraper_utils import (
    extract_article_links,
    process_single_article,
    fetch_and_process_page,
    get_browser_config,
    get_llm_strategy,
)

load_dotenv()


async def crawl_blog():
    """
    Main function to crawl all articles from the blog.
    """
    # Initialize configurations
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy()
    session_id = "blog_crawl_session"

    # Create output directory if it doesn't exist
    output_dir = "articles"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize state variables
    all_articles = []
    processed_urls = set()  # Track processed URLs to avoid duplicates
    seen_names = set()  # Track seen article names to avoid duplicates

    # Start the web crawler context
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Process each page of the blog listing
        for page_number in range(1, MAX_BLOG_PAGES + 1):
            # Use the correct pagination parameter PAGEN_1 instead of page
            page_url = f"{BLOG_BASE_URL}?PAGEN_1={page_number}"
            print(f"\n--- Processing blog page {page_number}/{MAX_BLOG_PAGES} ---")
            
            # Extract article links from the current page
            article_links = await extract_article_links(crawler, page_url, session_id)
            
            if not article_links:
                print(f"No articles found on page {page_number}. Stopping.")
                break
                
            # Process each article
            for article in article_links:
                article_url = article["url"]
                
                # Skip if already processed - but be more specific about what constitutes a duplicate
                # Extract the unique part of the URL for comparison
                article_path = article_url.split('/')[-2] if article_url.endswith('/') else article_url.split('/')[-1]
                
                if article_path in processed_urls:
                    print(f"Article already processed: {article_path}. Skipping.")
                    continue
                
                # Debug output
                print(f"Processing new article path: {article_path}")
                processed_urls.add(article_path)
                
                # Process the article using fetch_and_process_page with LLM strategy
                article_data, _ = await fetch_and_process_page(
                    crawler,
                    1,  # page_number is always 1 for single articles
                    article_url,
                    CSS_SELECTOR,
                    llm_strategy,
                    session_id,
                    REQUIRED_KEYS,
                    seen_names,
                )
                
                if article_data:
                    # Generate filename from URL
                    article_filename = os.path.basename(article_url.rstrip('/'))
                    if not article_filename:
                        article_filename = f"article_{len(all_articles) + 1}"
                        
                    # Save individual article
                    md_path = os.path.join(output_dir, f"{article_filename}.md")
                    csv_path = os.path.join(output_dir, f"{article_filename}.csv")
                    
                    save_venues_to_markdown(article_data, md_path)
                    save_venues_to_csv(article_data, csv_path)
                    print(f"Saved article to {md_path} and {csv_path}")
                    
                    # Add to all articles
                    all_articles.extend(article_data)
                
                # Pause between requests to be polite
                await asyncio.sleep(2)
            
            # Pause between pages
            await asyncio.sleep(3)

    # Save the collected articles to a combined file
    if all_articles:
        # Save to markdown
        save_venues_to_markdown(all_articles, "all_articles.md")
        print(f"Saved {len(all_articles)} articles to 'all_articles.md'.")
        
        # Also save to CSV for backward compatibility
        save_venues_to_csv(all_articles, "all_articles.csv")
        print(f"Saved {len(all_articles)} articles to 'all_articles.csv'.")
    else:
        print("No articles were found during the crawl.")

    # Display usage statistics for the LLM strategy
    llm_strategy.show_usage()


async def main():
    """
    Entry point of the script.
    """
    await crawl_blog()


if __name__ == "__main__":
    asyncio.run(main())