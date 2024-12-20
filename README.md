## Project: Autonomous Emergency Braking (AEB) on Pedestrian Detection
**Goal**: This AEB system will detect pedestrians and alert or stop the vehicle to avoid collisions.

---

### Setup

1. **Install Dependencies**  
   Run the following command to set up all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### Guide

#### 1. Dataset Preparation and Filtering

1. **Annotations for Class Filtering**  
   - The `annotations` directory contains the necessary class annotations to filter pedestrian images from the COCO dataset.
   - Download the [2017 Train/Val Annotations (241 MB)](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) from the [official COCO website](https://cocodataset.org/#download).
   - After downloading, unzip the file and place it in the root directory of the repository.

2. **Filter Pedestrian Images**  
   - Execute the following command to generate the list of pedestrian images:
     ```bash
     python coco_filter_pedestrians.py
     ```
   - **Output**: A file named `pedestrian_image_urls.json` will be created in `data/labels`, containing URLs of filtered images.

3. **Download Pedestrian Images**  
   - To download the pedestrian images, run:
     ```bash
     python coco_images_pedestrians.py
     ```
   - **Note**: Uses multithreading to accelerate the downloading process.
   - **Output**: All images will be saved in `data/labels`.

---

#### 2. YOLOv8 Model for Pedestrian Detection

1. **Load YOLOv8 Pre-trained Model**  
   - The `models` folder contains YOLOv8 model weights, trained on the COCO dataset for pedestrian detection.

2. **Demo Images**:
   - The `demo_test_images` folder contains sample images to test the detection.

3. **Perform Pedestrian Detection**  
   - To detect pedestrians in test images, execute:
     ```bash
     python yolov8_detect_pedestrians.py
     ```
   - **Output**: Each image will return either `Pedestrian Detected` or `No Pedestrian Detected`.

---

#### 3. CARLA Simulation

### Notes
- **Pedestrian Detection**: When pedestrians are detected by the sensor or camera, their images are saved locally in the directory `images/sceneX/*`. For example, for Scene `0`, the images will be stored in `images/scene0/*`.

- **Traffic Lights**: The ego vehicle disregards any traffic light detections and focuses solely on pedestrians.


**Scene 0 - Braking on pedestrian detection**  

   - **Running the Script**: ```python scene_basic.py```

   - **Video**: [Scene Basic Video](videos/Scene0.mp4)


**Scene 1 - Braking on pedestrian detection in low light and fog**  

   - **Running the Script**: ```python scene_1.py```

   - **Video**: [Scene 1 Video](videos/Scene1.mp4)


**Scene 2 - Braking when pedestrian occluded behind another car**  

   - **Running the Script**: ```python scene_2.py```

   - **Video**: [Scene 2 Video](videos/Scene2.mp4)


**Scene 3 - Braking when pedestrian is visible, becomes occluded and again visible**  

   - **Running the Script**: ```python scene_3.py```

   - **Video**: [Scene 3 Video](videos/Scene3.mp4)


**Scene 4 - Braking during right turn when the light is green and pedestrian is crossing**  

   - **Running the Script**: ```python scene_4.py```

   - **Video**: [Scene 4 Video](videos/Scene4.mp4)


**Scene 5 - Braking when pedestrian is having strange object, such as shopping cart or trolley**  

   - **Running the Script**: ```python scene_5.py```

   - **Video**: [Scene 5 Video](videos/Scene5.mp4)
