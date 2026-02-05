import argparse
import requests
from pathlib import Path
from urllib.parse import urlparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_from_links(links_file: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(links_file, 'r') as f:
        lines = []
        for line in f:
            line = line.strip()
            if 'http' in line:
                url_start = line.find('http')
                url = line[url_start:].strip()
                if url.endswith('.'):
                    url = url[:-1]
                lines.append(url)
    
    logger.info(f"Downloading {len(lines)} images to {output_dir}")
    
    for idx, url in enumerate(tqdm(lines, desc="Downloading")):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            parsed = urlparse(url)
            ext = Path(parsed.path).suffix or ".png"
            
            filename = output_path / f"ref_{idx:04d}{ext}"
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
    
    logger.info(f"Download complete. {len(list(output_path.glob('*')))} files in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download images from links.txt")
    parser.add_argument("--links", type=str, default="links.txt", help="File with image URLs")
    parser.add_argument("--output", type=str, default="training_data", help="Output directory")
    
    args = parser.parse_args()
    
    download_from_links(args.links, args.output)


if __name__ == "__main__":
    main()
