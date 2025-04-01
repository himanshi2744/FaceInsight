# FaceInsight: Gender & Age Estimator

## Objective
To build a gender and age detection system that can estimate the gender and age range of a person from a single image or via webcam.

## About the Project
This project utilizes deep learning techniques to predict gender and age from facial images. The model used is based on the pre-trained models by [Tal Hassner and Gil Levi](https://talhassner.github.io/home/projects/Adience/Adience-data.html). Given an input image, the model classifies the gender as either 'Male' or 'Female' and predicts the age range from one of the following categories:
- (0 – 2)
- (4 – 6)
- (8 – 12)
- (15 – 20)
- (25 – 32)
- (38 – 43)
- (48 – 53)
- (60 – 100)

Age prediction is treated as a classification problem rather than regression due to variations caused by lighting, expressions, and obstructions.

## Dataset
The project is based on the publicly available [Adience dataset](https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification). This dataset is a benchmark for face recognition tasks, containing images under real-world conditions such as different lighting, poses, and occlusions. The dataset consists of 26,580 images of 2,284 individuals categorized into eight age ranges. All images are collected from Flickr albums and fall under the Creative Commons (CC) license.

The dataset is particularly useful for age and gender classification tasks as it includes a diverse range of images that reflect real-world variability in human faces. It is annotated with both age and gender labels, making it a valuable resource for training deep learning models.

### Additional Image Source
To supplement the dataset, images have also been sourced from **[Pexels](https://www.pexels.com/search/person/)**, a free stock photo website. Pexels provides high-quality images under the **Pexels License**, which allows usage for commercial and personal projects without attribution.

## Caffe Model Details
This project uses models trained with the Caffe deep learning framework. Caffe (Convolutional Architecture for Fast Feature Embedding) is an open-source deep learning framework developed by Berkeley AI Research (BAIR). It is known for its speed and efficiency, particularly in image classification tasks.

### Model Files Used:
- `age_deploy.prototxt` - Defines the architecture of the age classification model.
- `age_net.caffemodel` - Contains the trained weights for the age classification model.
- `gender_deploy.prototxt` - Defines the architecture of the gender classification model.
- `gender_net.caffemodel` - Contains the trained weights for the gender classification model.

These models use Convolutional Neural Networks (CNNs) trained on the Adience dataset. The final layer of each model is a softmax classifier, which assigns probabilities to each age group or gender category. The models are optimized using stochastic gradient descent (SGD) and have been pre-trained to recognize common facial features associated with different age groups and genders.

## Requirements
Install the following Python libraries before running the project:

```bash
pip install opencv-python argparse
```

## Project Files
- `opencv_face_detector.pbtxt` - Configuration file for face detection model
- `opencv_face_detector_uint8.pb` - Pre-trained TensorFlow model for face detection
- `age_deploy.prototxt` - Configuration file for age prediction model
- `age_net.caffemodel` - Pre-trained Caffe model for age classification
- `gender_deploy.prototxt` - Configuration file for gender classification model
- `gender_net.caffemodel` - Pre-trained Caffe model for gender classification
- `detect.py` - Python script for performing gender and age detection

## Usage
### 1. Detect Gender and Age from an Image
Ensure that the image is in the same directory as the project files and use the following command:

```bash
python detect.py --image <image_name>
```

### 2. Detect Gender and Age via Webcam
Run the following command:

```bash
python detect.py
```
Press `Ctrl + C` to stop the execution.

## Example Output
```bash
> python detect.py --image sample.jpg
Gender: Female
Age: 25-32 years
```

## License
This project is open-source and licensed under the MIT License. Please ensure that any modifications and usage comply with the original dataset and model licensing terms.

## Acknowledgments
- **Dataset:** Adience dataset by Tal Hassner and Gil Levi
- **Pre-trained Models:** Caffe models trained on the Adience dataset
- **Deep Learning Frameworks:** OpenCV DNN module
- **Image Source:** Additional images used from **[Pexels](https://www.pexels.com/search/person/)** under their free-to-use license.

---
### Notes:
- The images used in examples should be sourced from public domain or personal datasets to avoid copyright violations.
- This project is intended for educational purposes only.
