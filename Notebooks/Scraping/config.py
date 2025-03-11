# config.py

# URL for blog listing page
BLOG_BASE_URL = "https://sappo.ru/blog/"

# URL for individual article (used in main_one_page.py)
ARTICLE_BASE_URL = "https://sappo.ru/blog/sam-sebe-detailer/kak-snyat-tonirovku-so-stekla-avtomobilya/"

# For backward compatibility
BASE_URL = ARTICLE_BASE_URL

# CSS selector for article content
CSS_SELECTOR = "[class^='content-text muted777']"

# Required keys in extracted data
REQUIRED_KEYS = [
    "name",
    "content_text",
    "links",
]

# Maximum number of blog pages to crawl
MAX_BLOG_PAGES = 9
