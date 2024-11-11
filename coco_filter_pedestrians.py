import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO


def cocoJson(images):
    arrayIds = np.array([img["id"] for img in images])
    annIds = coco.getAnnIds(imgIds=arrayIds, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    annotations = {
        "description": "pedestrians images and annotations",
        "images": images,
        "annotations": anns,
    }

    return annotations


if __name__ == "__main__":
    annotationFile = "annotations/instances_train2017.json"
    coco = COCO(annotationFile)
    catNms = ["person"]
    catIds = coco.getCatIds(catNms)
    imgIds = coco.getImgIds(catIds=catIds)

    imgOriginals = coco.loadImgs(imgIds)
    annotations = cocoJson(imgOriginals)

    Path("data/labels").mkdir(parents=True, exist_ok=True)

    jsonFile = "data/labels/pedestrian_image_urls.json"
    with open(jsonFile, "w") as outfile:
        json.dump(annotations, outfile)

    print(f"Number of pedestrian images (COCO): {len(imgOriginals)}")
    print(f"Image URLs can be found: {jsonFile}")
