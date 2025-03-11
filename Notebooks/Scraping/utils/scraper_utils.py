import json
import os
import re
from typing import List, Set, Tuple, Dict
from bs4 import BeautifulSoup

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
)

from models.venue import Venue
from utils.data_utils import is_complete_venue, is_duplicate_venue


def get_browser_config() -> BrowserConfig:
    """
    Returns the browser configuration for the crawler.

    Returns:
        BrowserConfig: The configuration settings for the browser.
    """
    # https://docs.crawl4ai.com/core/browser-crawler-config/
    return BrowserConfig(
        browser_type="chromium",  # Type of browser to simulate
        headless=False,  # Whether to run in headless mode (no GUI)
        verbose=True,  # Enable verbose logging
    )


def get_llm_strategy() -> LLMExtractionStrategy:
    """
    Returns the configuration for the language model extraction strategy.

    Returns:
        LLMExtractionStrategy: The settings for how to extract data using LLM.
    """
    # https://docs.crawl4ai.com/api/strategies/#llmextractionstrategy
    return LLMExtractionStrategy(
        # provider="gpt-4o-mini",  # Name of the LLM provider
        # api_token=os.getenv("OPENAI_API_KEY"),  # API token for authentication
        provider="openrouter/google/gemini-2.0-flash-001",
        api_token=os.getenv("OPENROUTER_API_KEY"),
        schema=Venue.model_json_schema(),  # JSON schema of the data model
        extraction_type="schema",  # Type of extraction to perform
        instruction=(
            "Extract the complete information from the content. "
            "For each article, extract: "
            "1. The full title as 'name' "
            "2. The COMPLETE text content as 'content_text' without any truncation or summarization. "
            "Do not abbreviate or shorten the text with ellipses (...). "
            "Include ALL paragraphs and sections in their entirety. "
            "3. All links mentioned in the article as 'links'. "
            "Ensure that no part of the text is omitted or truncated."

        ),  # Instructions for the LLM
        input_format="markdown",  # Format of the input content
        verbose=True,  # Enable verbose logging
    )


async def check_no_results(
    crawler: AsyncWebCrawler,
    url: str,
    session_id: str,
) -> bool:
    """
    Checks if the "No Results Found" message is present on the page.

    Args:
        crawler (AsyncWebCrawler): The web crawler instance.
        url (str): The URL to check.
        session_id (str): The session identifier.

    Returns:
        bool: True if "No Results Found" message is found, False otherwise.
    """
    # Fetch the page without any CSS selector or extraction strategy
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            session_id=session_id,
        ),
    )

    if result.success:
        if "No Results Found" in result.cleaned_html:
            return True
    else:
        print(
            f"Error fetching page for 'No Results Found' check: {result.error_message}"
        )

    return False


async def extract_article_links(
    crawler: AsyncWebCrawler,
    page_url: str,
    session_id: str,
) -> List[Dict[str, str]]:
    """
    Extracts links to articles from a blog listing page.

    Args:
        crawler (AsyncWebCrawler): The web crawler instance.
        page_url (str): The URL of the blog listing page.
        session_id (str): The session identifier.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing article titles and URLs.
    """
    print(f"Extracting article links from {page_url}...")
    
    # Fetch the page without any extraction strategy
    result = await crawler.arun(
        url=page_url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            session_id=session_id,
        ),
    )

    if not result.success:
        print(f"Error fetching page {page_url}: {result.error_message}")
        return []

    # Debug: Print a sample of the HTML to understand its structure
    html_sample = result.cleaned_html[:5000] + "..." if len(result.cleaned_html) > 5000 else result.cleaned_html
    print(f"HTML sample from page:\n{html_sample}\n")
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(result.cleaned_html, 'html.parser')
    
    # Try different selectors to find article blocks
    selectors_to_try = [
        'div.item.bg-white.bordered.box-shadow.rounded3',
        'div.item.bg-white',
        'div.item',
        'div.title a',  # Direct link to articles
        'a[href*="/blog/"]',  # Any link containing "/blog/"
        'a'  # All links as a last resort
    ]
    
    article_blocks = []
    for selector in selectors_to_try:
        article_blocks = soup.select(selector)
        print(f"Selector '{selector}' found {len(article_blocks)} elements")
        if article_blocks:
            print(f"Using selector: {selector}")
            break
    
    articles = []
    
    # If we're using a selector that directly finds links
    if selector in ['div.title a', 'a[href*="/blog/"]', 'a']:
        # Debug: Print all links found
        print("\nAll links found:")
        for i, link in enumerate(article_blocks):
            url = link.get('href')
            title = link.get_text(strip=True)
            print(f"{i+1}. URL: {url}, Title: {title}")
        
        for link in article_blocks:
            url = link.get('href')
            title = link.get_text(strip=True)
            
            # Debug: Print each URL and why it's accepted or rejected
            if not url:
                print(f"Rejected: {title} - No URL")
                continue
                
            if '/blog/' not in url:
                print(f"Rejected: {url} - Not a blog URL")
                continue
                
            # Only reject category URLs, not article URLs
            if url.endswith('/') and url.count('/') < 4:
                print(f"Rejected: {url} - Category URL")
                continue
                
            if '?' in url:
                print(f"Rejected: {url} - Contains ?")
                continue
            
            # Make sure URL is absolute
            if not url.startswith('http'):
                url = f"https://sappo.ru{url}"
            
            # Debug output
            print(f"Accepted article: {title} at {url}")
                
            articles.append({
                "title": title or f"Article at {url}",
                "url": url
            })
    else:
        # Original approach for article blocks
        for block in article_blocks:
            # Find the title element
            title_element = block.select_one('div.title a')
            if title_element:
                title = title_element.get_text(strip=True)
                url = title_element.get('href')
                
                # Make sure URL is absolute
                if url and not url.startswith('http'):
                    url = f"https://sappo.ru{url}"
                    
                articles.append({
                    "title": title,
                    "url": url
                })
    
    print(f"Found {len(articles)} articles on page {page_url}")
    return articles


async def fetch_and_process_page(
    crawler: AsyncWebCrawler,
    page_number: int,
    base_url: str,
    css_selector: str,
    llm_strategy: LLMExtractionStrategy,
    session_id: str,
    required_keys: List[str],
    seen_names: Set[str],
) -> Tuple[List[dict], bool]:
    """
    Fetches and processes a single page of venue data.

    Args:
        crawler (AsyncWebCrawler): The web crawler instance.
        page_number (int): The page number to fetch.
        base_url (str): The base URL of the website.
        css_selector (str): The CSS selector to target the content.
        llm_strategy (LLMExtractionStrategy): The LLM extraction strategy.
        session_id (str): The session identifier.
        required_keys (List[str]): List of required keys in the venue data.
        seen_names (Set[str]): Set of venue names that have already been seen.

    Returns:
        Tuple[List[dict], bool]:
            - List[dict]: A list of processed venues from the page.
            - bool: A flag indicating if the "No Results Found" message was encountered.
    """
    url = f"{base_url}?page={page_number}"
    print(f"Loading page {page_number}...")
    # В функции fetch_and_process_page добавить:
    # print("HTML content sent to LLM:", result.cleaned_html[:500] + "..." if len(result.cleaned_html) > 500 else result.cleaned_html)

    # Check if "No Results Found" message is present
    no_results = await check_no_results(crawler, url, session_id)
    if no_results:
        return [], True  # No more results, signal to stop crawling

    # Fetch page content with the extraction strategy
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Do not use cached data
            extraction_strategy=llm_strategy,  # Strategy for data extraction
            css_selector=css_selector,  # Target specific content on the page
            session_id=session_id,  # Unique session ID for the crawl
        ),
    )

    if not (result.success and result.extracted_content):
        print(f"Error fetching page {page_number}: {result.error_message}")
        return [], False

    # Parse extracted content
    extracted_data = json.loads(result.extracted_content)
    if not extracted_data:
        print(f"No venues found on page {page_number}.")
        return [], False

    # After parsing extracted content
    print("Extracted data:", extracted_data)

    # Process venues
    complete_venues = []
    for venue in extracted_data:
        # Debugging: Print each venue to understand its structure
        print("Processing venue:", venue)

        # Ignore the 'error' key if it's False
        if venue.get("error") is False:
            venue.pop("error", None)  # Remove the 'error' key if it's False

        if not is_complete_venue(venue, required_keys):
            continue  # Skip incomplete venues

        if is_duplicate_venue(venue["name"], seen_names):
            print(f"Duplicate venue '{venue['name']}' found. Skipping.")
            continue  # Skip duplicate venues

        # Add venue to the list
        seen_names.add(venue["name"])
        complete_venues.append(venue)

    if not complete_venues:
        print(f"No complete venues found on page {page_number}.")
        return [], False

    print(f"Extracted {len(complete_venues)} venues from page {page_number}.")
    return complete_venues, False  # Continue crawling


async def process_single_article(
    crawler: AsyncWebCrawler,
    article_url: str,
    css_selector: str,
    llm_strategy: LLMExtractionStrategy,
    session_id: str,
    required_keys: List[str],
) -> List[dict]:
    """
    Processes a single article page and extracts its content.

    Args:
        crawler (AsyncWebCrawler): The web crawler instance.
        article_url (str): The URL of the article to process.
        css_selector (str): The CSS selector to target the content.
        llm_strategy (LLMExtractionStrategy): The LLM extraction strategy.
        session_id (str): The session identifier.
        required_keys (List[str]): List of required keys in the article data.

    Returns:
        List[dict]: A list containing the processed article data.
    """
    print(f"Processing article: {article_url}")
    
    # Try different CSS selectors for article content
    selectors_to_try = [
        css_selector,  # Original selector
        "div.content-text",  # Common content class
        "div.article-content",  # Another common content class
        "div.blog-post-text",  # Another possible content class
        "article",  # Generic article tag
        "div.content",  # Generic content class
        "body"  # Last resort - entire body
    ]
    
    for selector in selectors_to_try:
        print(f"Trying selector: {selector}")
        
        # Fetch page content with the extraction strategy
        result = await crawler.arun(
            url=article_url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,  # Do not use cached data
                extraction_strategy=llm_strategy,  # Strategy for data extraction
                css_selector=selector,  # Target specific content on the page
                session_id=session_id,  # Unique session ID for the crawl
            ),
        )

        if not result.success:
            print(f"Error fetching article with selector {selector}: {result.error_message}")
            continue
            
        if not result.extracted_content:
            print(f"No extracted content found with selector {selector}")
            continue
            
        # Parse extracted content
        try:
            extracted_data = json.loads(result.extracted_content)
            
            if not extracted_data:
                print(f"Empty extracted data with selector {selector}")
                continue
                
            # Process article data
            complete_articles = []
            for article in extracted_data:
                # Debugging: Print article data
                print(f"Processing article content with selector {selector}:", article)

                # Ignore the 'error' key if it's False
                if article.get("error") is False:
                    article.pop("error", None)  # Remove the 'error' key if it's False

                if not is_complete_venue(article, required_keys):
                    print(f"Incomplete article data for {article_url} with selector {selector}. Missing keys: {[key for key in required_keys if key not in article]}")
                    continue  # Skip incomplete articles

                # Add article to the list
                complete_articles.append(article)

            if complete_articles:
                print(f"Successfully extracted content from article {article_url} with selector {selector}.")
                return complete_articles
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error with selector {selector}: {e}")
            continue
    
    # If we get here, we've tried all selectors and none worked
    print(f"Failed to extract content from article {article_url} with any selector.")
    
    # As a fallback, create a minimal article entry with just the URL as content
    article_name = os.path.basename(article_url.rstrip('/'))
    fallback_article = {
        "name": f"Article: {article_name}",
        "content_text": f"Could not extract content from {article_url}",
        "links": article_url
    }
    
    # Check if this minimal entry satisfies the required keys
    if is_complete_venue(fallback_article, required_keys):
        print(f"Created fallback entry for {article_url}")
        return [fallback_article]
        
    return []


async def extract_article_content_with_bs4(
    crawler: AsyncWebCrawler,
    article_url: str,
    session_id: str,
) -> List[dict]:
    """
    Extracts article content using BeautifulSoup instead of LLM.
    
    Args:
        crawler (AsyncWebCrawler): The web crawler instance.
        article_url (str): The URL of the article to process.
        session_id (str): The session identifier.
        
    Returns:
        List[dict]: A list containing the processed article data.
    """
    print(f"Processing article with BeautifulSoup: {article_url}")
    
    # Fetch the page without any extraction strategy
    result = await crawler.arun(
        url=article_url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            session_id=session_id,
        ),
    )
    
    if not result.success:
        print(f"Error fetching article {article_url}: {result.error_message}")
        return []
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(result.cleaned_html, 'html.parser')
    
    # Extract article title
    title = None
    title_candidates = [
        soup.select_one('h1'),  # Most common heading
        soup.select_one('header h1'),  # Header heading
        soup.select_one('article h1'),  # Article heading
        soup.select_one('.article-title'),  # Common article title class
        soup.select_one('.post-title'),  # Common post title class
        soup.select_one('title')  # Page title as last resort
    ]
    
    for candidate in title_candidates:
        if candidate and candidate.get_text(strip=True):
            title = candidate.get_text(strip=True)
            break
    
    if not title:
        # Use filename from URL as fallback title
        title = os.path.basename(article_url.rstrip('/'))
        title = title.replace('-', ' ').replace('_', ' ').title()
    
    # Extract article content
    content = ""
    
    # For sappo.ru, we know the article content is in the div with class "detail-text"
    article_content = soup.select_one('div.detail-text')
    
    if article_content:
        # Extract headings and paragraphs from the article content
        content_parts = []
        
        # Process headings (h2, h3, etc.) and paragraphs
        for element in article_content.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol']):
            if element.name.startswith('h'):
                # Add heading with markdown formatting
                heading_level = int(element.name[1])
                heading_text = element.get_text(strip=True)
                content_parts.append(f"{'#' * heading_level} {heading_text}")
            elif element.name == 'p':
                # Add paragraph
                paragraph_text = element.get_text(strip=True)
                if paragraph_text:  # Skip empty paragraphs
                    content_parts.append(paragraph_text)
            elif element.name in ['ul', 'ol']:
                # Process lists
                for li in element.find_all('li'):
                    li_text = li.get_text(strip=True)
                    if li_text:
                        content_parts.append(f"  * {li_text}")
        
        # Join all content parts with double newlines
        content = "\n\n".join(content_parts)
    
    if not content:
        # Try alternative selectors for sappo.ru
        article_content = soup.select_one('div.blog-detail-text')
        if article_content:
            # Extract headings and paragraphs from the article content
            content_parts = []
            
            # Process headings (h2, h3, etc.) and paragraphs
            for element in article_content.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol']):
                if element.name.startswith('h'):
                    # Add heading with markdown formatting
                    heading_level = int(element.name[1])
                    heading_text = element.get_text(strip=True)
                    content_parts.append(f"{'#' * heading_level} {heading_text}")
                elif element.name == 'p':
                    # Add paragraph
                    paragraph_text = element.get_text(strip=True)
                    if paragraph_text:  # Skip empty paragraphs
                        content_parts.append(paragraph_text)
                elif element.name in ['ul', 'ol']:
                    # Process lists
                    for li in element.find_all('li'):
                        li_text = li.get_text(strip=True)
                        if li_text:
                            content_parts.append(f"  * {li_text}")
            
            # Join all content parts with double newlines
            content = "\n\n".join(content_parts)
    
    if not content:
        # Fallback to generic content extraction if specific selectors don't work
        main_content = None
        
        # Try to find the main content container
        content_candidates = [
            soup.select_one('div.content-text'),  # Common content class
            soup.select_one('div.article-content'),  # Another common content class
            soup.select_one('article'),  # Article tag
            soup.select_one('div.content'),  # Generic content class
            soup.select_one('main'),  # Main content area
        ]
        
        for candidate in content_candidates:
            if candidate:
                main_content = candidate
                break
        
        if main_content:
            # Extract headings and paragraphs from the main content
            content_parts = []
            
            # Process headings (h2, h3, etc.) and paragraphs
            for element in main_content.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol']):
                if element.name.startswith('h'):
                    # Add heading with markdown formatting
                    heading_level = int(element.name[1])
                    heading_text = element.get_text(strip=True)
                    content_parts.append(f"{'#' * heading_level} {heading_text}")
                elif element.name == 'p':
                    # Add paragraph
                    paragraph_text = element.get_text(strip=True)
                    if paragraph_text:  # Skip empty paragraphs
                        content_parts.append(paragraph_text)
                elif element.name in ['ul', 'ol']:
                    # Process lists
                    for li in element.find_all('li'):
                        li_text = li.get_text(strip=True)
                        if li_text:
                            content_parts.append(f"  * {li_text}")
            
            # Join all content parts with double newlines
            content = "\n\n".join(content_parts)
        
        if not content:
            # Fallback to extracting all paragraphs if no structured content was found
            paragraphs = soup.select('p')
            if paragraphs:
                content = "\n\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
    
    if not content:
        content = f"Could not extract content from {article_url}"
    
    # Extract only relevant links from the article content
    links = []
    article_content = article_content or main_content
    
    if article_content:
        for link in article_content.select('a[href]'):
            href = link.get('href')
            if href and not href.startswith('#') and not href.startswith('javascript:'):
                # Make absolute URL if relative
                if not href.startswith('http'):
                    if href.startswith('/'):
                        # Domain-relative URL
                        domain = '/'.join(article_url.split('/')[:3])  # Get domain from article URL
                        href = f"{domain}{href}"
                    else:
                        # Path-relative URL
                        base_path = '/'.join(article_url.split('/')[:-1])  # Get base path from article URL
                        href = f"{base_path}/{href}"
                
                links.append(href)
    
    # Create article data
    article_data = {
        "name": title,
        "content_text": content,
        "links": ", ".join(links) if links else article_url
    }
    
    print(f"Successfully extracted content from article {article_url} using BeautifulSoup.")
    return [article_data]