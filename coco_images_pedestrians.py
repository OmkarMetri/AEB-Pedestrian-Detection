import os
import json
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


def download_image(im, session):
    file_path = f"data/images/{im['file_name']}"

    if not os.path.isfile(file_path):
        img_data = session.get(im["coco_url"]).content
        with open(file_path, "wb") as handler:
            handler.write(img_data)
    return file_path


def downloadImages(images, title):
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        with tqdm(total=len(images), desc=title, ncols=100, unit="image") as pbar:
            for im in images:
                futures.append(executor.submit(download_image, im, session))

            for future in as_completed(futures):
                pbar.update(1)


if __name__ == "__main__":
    pedestrian_image_urls = "data/labels/pedestrian_image_urls.json"
    with open(pedestrian_image_urls, "r") as f:
        imgs = json.load(f)["images"]

    print(f"Number of pedestrian images (COCO): {len(imgs)}")
    Path("data/images").mkdir(parents=True, exist_ok=True)
    downloadImages(imgs, "Downloading Images")
    print("Download Completed")
