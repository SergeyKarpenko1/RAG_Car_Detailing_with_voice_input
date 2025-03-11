import csv
import os

from models.venue import Venue


def is_duplicate_venue(venue_name: str, seen_names: set) -> bool:
    return venue_name in seen_names


def is_complete_venue(venue: dict, required_keys: list) -> bool:
    return all(key in venue for key in required_keys)


def save_venues_to_csv(venues: list, filename: str):
    if not venues:
        print("No venues to save.")
        return

    # Use field names from the Venue model
    fieldnames = Venue.model_fields.keys()

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(venues)
    print(f"Saved {len(venues)} venues to '{filename}'.")


def save_venues_to_markdown(venues: list, filename: str):
    """
    Save venues to a markdown file.
    
    Args:
        venues (list): List of venue dictionaries.
        filename (str): Name of the markdown file to save to.
    """
    if not venues:
        print("No venues to save.")
        return
    
    # Ensure the filename has .md extension
    if not filename.endswith('.md'):
        filename = f"{os.path.splitext(filename)[0]}.md"
    
    with open(filename, mode="w", encoding="utf-8") as file:
        for venue in venues:
            # Write the name as a header if it doesn't already start with #
            if not venue['name'].startswith('#'):
                file.write(f"# {venue['name']}\n\n")
            else:
                file.write(f"{venue['name']}\n\n")
            
            # Write the content text
            # Check if content already has markdown formatting
            content_text = venue['content_text']
            
            # Write the content text, preserving any existing markdown formatting
            file.write(f"{content_text}\n\n")
            
            # Write the links only if they're not already included in the content
            if venue.get('links') and "## Ссылки" not in content_text:
                file.write("## Ссылки\n\n")
                # Check if links is a list or a string
                if isinstance(venue['links'], list):
                    for link in venue['links']:
                        file.write(f"- {link}\n")
                else:
                    # If it's a string, split by commas or just write as is
                    links = venue['links'].split(',') if ',' in venue['links'] else [venue['links']]
                    for link in links:
                        link = link.strip()
                        if link:  # Only write non-empty links
                            file.write(f"- {link}\n")
            
            # Add a separator between venues
            file.write("\n---\n\n")
    
    print(f"Saved {len(venues)} venues to '{filename}'.")