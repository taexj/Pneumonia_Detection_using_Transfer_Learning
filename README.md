# Pneumonia_Detection_using_Transfer_Learning
This project aims to leverage the power of transfer learning in convolutional neural networks (CNNs) to accurately detect pneumonia from chest X-ray images. By utilizing pre-trained models and adapting them to our specific problem, we strive to create a robust and efficient solution that can assist healthcare professionals in diagnosing pneumonia more effectively.
## Project Overview
Pneumonia is a significant public health problem and one of the leading causes of death in children and the elderly worldwide. Early and accurate detection is crucial for effective treatment. This project uses deep learning and transfer learning techniques to develop a model that can classify chest X-ray images as either showing signs of pneumonia or not.
## Dataset
The dataset used for this project consists of chest X-ray images from pediatric patients, sourced from the publicly available [ Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset. 
## Methodology
We employ transfer learning using pre-trained VGG16 as a starting point. The model has been pre-trained on the ImageNet dataset, which includes a wide variety of images. We then fine-tune this model on our specific dataset of chest X-ray images to detect pneumonia.
## Requirements
- Python 3.6+
- TensorFlow 2.x
- Keras
- scipy
- glob2

