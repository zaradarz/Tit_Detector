import os
from duckduckgo_search import DDGS
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import time
from duckduckgo_search.exceptions import RatelimitException, DuckDuckGoSearchException

# Settings
NUM_IMAGES = 500
IMG_SIZE = (128, 128)
DATASET_DIR = 'dataset'
CATEGORIES = {
    'tits': [
        'tit bird', 'parus major', 'great tit bird', 'blue tit bird', 'titmouse bird'
    ],
    'not_tits': [
        'landscape', 'people', 'tree', 'building', 'mountain', 'river', 'city'
        # Notice that we're not downloading images of birds that are not tits. 
    ]
}

# Ensure directories exist
def make_dirs():
    for category in CATEGORIES:
        dir_path = os.path.join(DATASET_DIR, category)
        os.makedirs(dir_path, exist_ok=True)

# Download and save images
def download_images(category, queries, num_images):
    save_dir = os.path.join(DATASET_DIR, category)
    downloaded = 0
    seen_urls = set()
    with DDGS() as ddgs:
        for query in queries:
            # Retry with exponential backoff if rate limited
            attempt = 0
            backoff = 10  # seconds
            while True:
                try:
                    results = ddgs.images(query, max_results=num_images*2)
                    break
                except RatelimitException:
                    attempt += 1
                    wait = backoff * attempt
                    print(f"Rate limit hit. Waiting {wait}s before retrying...")
                    time.sleep(wait)
                except DuckDuckGoSearchException as e:
                    print(f"Search error: {e}. Skipping query '{query}'.")
                    results = []
                    break

            for result in tqdm(results, desc=f"{category} - {query}", leave=False):
                url = result['image']
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                try:
                    response = requests.get(url, timeout=5)
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    img = img.resize(IMG_SIZE)
                    img.save(os.path.join(save_dir, f'{category}_{downloaded:04d}.jpg'))
                    downloaded += 1
                except Exception:
                    continue  # Skip failed downloads or corrupt images
                time.sleep(0.2)  # small delay between downloads
                if downloaded >= num_images:
                    break
            if downloaded >= num_images:
                break
    print(f"Downloaded {downloaded} images for '{category}'")

if __name__ == '__main__':
    print("Creating dataset directories...")
    make_dirs()
    for category, queries in CATEGORIES.items():
        print(f"\nDownloading images for '{category}'...")
        download_images(category, queries, NUM_IMAGES)
    print("\nAll done! Your dataset is ready in the 'dataset/' folder.") 