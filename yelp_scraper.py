#!/usr/bin/env python3
"""
Yelp Restaurant Review Scraper
Scrapes restaurant information and reviews from Yelp pages
Optimized for archived Yelp pages from web.archive.org

Usage: python3 yelp_scraper.py <url>

Default URL: http://www.yelp.ca/biz/pai-northern-thai-kitchen-toronto-5
"""

import sys
import csv
import re
from urllib.parse import quote
import requests
from bs4 import BeautifulSoup
import time


class YelpScraper:
    """Scraper for Yelp restaurant pages"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def debug_print(self, message):
        """Print debug messages"""
        if self.debug:
            print(f"[DEBUG] {message}")
    
    def get_archive_url(self, original_url, timestamp="20141202"):
        """Convert original URL to Wayback Machine URL"""
        # Use 2014 snapshot by default (best compatibility)
        return f"https://web.archive.org/web/{timestamp}/{original_url}"
    
    def get_page(self, url):
        """Fetch page content"""
        try:
            # If not already an archive URL, convert it
            if 'web.archive.org' not in url:
                archive_url = self.get_archive_url(url)
                print(f"Using Wayback Machine archive (2014)...")
                print(f"Archive URL: {archive_url}\n")
                url = archive_url
            
            print(f"Fetching page...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            print(f"✓ Page fetched successfully ({len(response.text)} chars)")
            return response.text
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print("⚠️  Access Denied (403)")
            else:
                print(f"HTTP Error: {e}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove newlines
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text.strip()
    
    def extract_restaurant_info(self, soup):
        """Extract restaurant information"""
        restaurant_data = {
            'name': '',
            'total_reviews': 0,
            'reviews': []
        }
        
        # Extract restaurant name (2014 Yelp structure)
        self.debug_print("Extracting restaurant name...")
        
        # Try h1 with class "biz-page-title"
        name_elem = soup.find('h1', class_='biz-page-title')
        if name_elem:
            restaurant_data['name'] = self.clean_text(name_elem.get_text())
            self.debug_print(f"Name: {restaurant_data['name']}")
        
        # Fallback to meta tag
        if not restaurant_data['name']:
            meta = soup.find('meta', property='og:title')
            if meta:
                restaurant_data['name'] = self.clean_text(meta.get('content', ''))
        
        # Extract total review count
        self.debug_print("Extracting review count...")
        
        # Look for review count span
        review_count_elem = soup.find('span', class_='review-count')
        if review_count_elem:
            count_text = review_count_elem.get_text()
            match = re.search(r'(\d+)', count_text.replace(',', ''))
            if match:
                restaurant_data['total_reviews'] = int(match.group(1))
                self.debug_print(f"Review count: {restaurant_data['total_reviews']}")
        
        # Extract reviews
        restaurant_data['reviews'] = self.extract_reviews(soup)
        
        return restaurant_data
    
    def extract_reviews(self, soup):
        """Extract individual reviews (2014 Yelp structure)"""
        reviews = []
        
        self.debug_print("Extracting reviews...")
        
        # 2014 Yelp uses div with class "review review--with-sidebar"
        review_divs = soup.find_all('div', class_=re.compile(r'review'))
        self.debug_print(f"Found {len(review_divs)} review divs")
        
        for i, review_div in enumerate(review_divs[:10], 1):  # Limit to 10
            review_data = self.extract_single_review(review_div)
            
            if review_data['review_text']:
                reviews.append(review_data)
                self.debug_print(f"Review {i}: ✓ ({len(review_data['review_text'])} chars)")
            else:
                self.debug_print(f"Review {i}: ✗ (no text)")
        
        return reviews
    
    def extract_single_review(self, review_elem):
        """Extract data from single review element"""
        review = {
            'review_text': '',
            'reviewer': '',
            'rating': ''
        }
        
        # Extract review text (2014 structure uses <p> with lang="en")
        # Try multiple selectors
        text_elem = review_elem.find('p', lang='en')
        if not text_elem:
            text_elem = review_elem.find('p', class_=re.compile(r'review-content'))
        if not text_elem:
            # Look for any p tag with substantial text
            p_tags = review_elem.find_all('p')
            for p in p_tags:
                text = self.clean_text(p.get_text())
                if len(text) > 50:
                    text_elem = p
                    break
        
        if text_elem:
            review['review_text'] = self.clean_text(text_elem.get_text())
        
        # Extract reviewer name
        # 2014 uses <a> with class "user-display-name"
        name_elem = review_elem.find('a', class_='user-display-name')
        if not name_elem:
            name_elem = review_elem.find('a', class_=re.compile(r'user'))
        if name_elem:
            review['reviewer'] = self.clean_text(name_elem.get_text())
        
        # Extract rating (2014 uses <meta> itemprop="ratingValue")
        rating_meta = review_elem.find('meta', itemprop='ratingValue')
        if rating_meta:
            review['rating'] = rating_meta.get('content', '')
        
        # Alternative: look for div with title containing "star"
        if not review['rating']:
            rating_div = review_elem.find('div', class_=re.compile(r'rating'))
            if rating_div:
                title = rating_div.get('title', '')
                match = re.search(r'(\d+(?:\.\d+)?)\s*star', title, re.I)
                if match:
                    review['rating'] = match.group(1)
        
        # Alternative: aria-label
        if not review['rating']:
            rating_elem = review_elem.find(attrs={'aria-label': re.compile(r'star', re.I)})
            if rating_elem:
                aria = rating_elem.get('aria-label', '')
                match = re.search(r'(\d+(?:\.\d+)?)', aria)
                if match:
                    review['rating'] = match.group(1)
        
        return review
    
    def save_to_csv(self, restaurant_data, filename='restaurant_reviews.csv'):
        """Save data to CSV"""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'Restaurant_Name', 'Total_Reviews', 
                    'Review_Text', 'Reviewer', 'Rating'
                ])
                
                writer.writeheader()
                
                if restaurant_data['reviews']:
                    for review in restaurant_data['reviews']:
                        writer.writerow({
                            'Restaurant_Name': restaurant_data['name'],
                            'Total_Reviews': restaurant_data['total_reviews'],
                            'Review_Text': review['review_text'],
                            'Reviewer': review['reviewer'],
                            'Rating': review['rating']
                        })
                else:
                    # Write header row with restaurant info
                    writer.writerow({
                        'Restaurant_Name': restaurant_data['name'],
                        'Total_Reviews': restaurant_data['total_reviews'],
                        'Review_Text': '',
                        'Reviewer': '',
                        'Rating': ''
                    })
            
            print(f"\n{'='*60}")
            print(f"✓ SUCCESS! Data saved to {filename}")
            print(f"✓ Restaurant: {restaurant_data['name']}")
            print(f"✓ Total Reviews: {restaurant_data['total_reviews']}")
            print(f"✓ Reviews Scraped: {len(restaurant_data['reviews'])}")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False
    
    def scrape_restaurant(self, url):
        """Main scraping method"""
        print(f"\n{'='*60}")
        print(f"Yelp Restaurant Review Scraper")
        print(f"{'='*60}")
        print(f"Target: {url}\n")
        
        html = self.get_page(url)
        if not html:
            return None
        
        if self.debug:
            with open('debug_page.html', 'w', encoding='utf-8') as f:
                f.write(html)
            print("[DEBUG] Saved to debug_page.html\n")
        
        soup = BeautifulSoup(html, 'html.parser')
        restaurant_data = self.extract_restaurant_info(soup)
        
        return restaurant_data
    
    def scrape_multiple_restaurants(self, urls):
        """Scrape multiple restaurants (BONUS)"""
        all_reviews = []
        
        print(f"\n{'='*60}")
        print(f"BONUS: Scraping {len(urls)} restaurants")
        print(f"{'='*60}\n")
        
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Processing...")
            restaurant_data = self.scrape_restaurant(url)
            
            if restaurant_data and restaurant_data['reviews']:
                for review in restaurant_data['reviews']:
                    all_reviews.append({
                        'restaurant_name': restaurant_data['name'],
                        'total_reviews': restaurant_data['total_reviews'],
                        'review': review
                    })
            
            # Delay between requests
            if i < len(urls):
                time.sleep(2)
        
        if all_reviews:
            self.save_multiple_to_csv(all_reviews)
        
        return all_reviews
    
    def save_multiple_to_csv(self, all_reviews, filename='all_restaurants_reviews.csv'):
        """Save multiple restaurants to CSV"""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'Restaurant_Name', 'Total_Reviews',
                    'Review_Text', 'Reviewer', 'Rating'
                ])
                writer.writeheader()
                
                for item in all_reviews:
                    writer.writerow({
                        'Restaurant_Name': item['restaurant_name'],
                        'Total_Reviews': item['total_reviews'],
                        'Review_Text': item['review']['review_text'],
                        'Reviewer': item['review']['reviewer'],
                        'Rating': item['review']['rating']
                    })
            
            print(f"\n✓ BONUS: Saved {len(all_reviews)} reviews to {filename}")
            
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point"""
    
    # Default URL
    default_url = "http://www.yelp.ca/biz/pai-northern-thai-kitchen-toronto-5"
    
    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("Yelp Restaurant Review Scraper")
        print("="*60)
        print("\nUsage:")
        print(f"  python3 {sys.argv[0]} <url>")
        print(f"  python3 {sys.argv[0]} --debug <url>")
        print(f"  python3 {sys.argv[0]} <url1> <url2> ... (multiple)")
        print("\nExamples:")
        print(f'  python3 {sys.argv[0]} "{default_url}"')
        print(f'  python3 {sys.argv[0]} --debug "{default_url}"')
        print("\nNote: Script uses 2014 Wayback Machine archive by default")
        print("="*60 + "\n")
        sys.exit(1)
    
    # Parse arguments
    debug = '--debug' in sys.argv or '-d' in sys.argv
    urls = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    
    if not urls:
        print("Error: No URL provided")
        sys.exit(1)
    
    # Create scraper
    scraper = YelpScraper(debug=debug)
    
    if len(urls) == 1:
        # Single restaurant
        restaurant_data = scraper.scrape_restaurant(urls[0])
        if restaurant_data:
            success = scraper.save_to_csv(restaurant_data)
            if not success:
                sys.exit(1)
        else:
            print("\n✗ Failed to scrape data")
            sys.exit(1)
    else:
        # Multiple restaurants (BONUS)
        scraper.scrape_multiple_restaurants(urls)


if __name__ == "__main__":
    main()
