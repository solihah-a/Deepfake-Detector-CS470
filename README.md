# Deepfake Image Detector - Tiny CNN

This project trains a lightweight CNN from scratch to classify images as:

- real
- AI-generated

It uses the Hugging Face dataset:

Rajarshi-Roy-research/Defactify_Image_Dataset

## Setup

### 1. Create virtual environment

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
## How the CNN works 
This model uses a small image-recognition system called a convolutional neural network, or CNN, to learn the visual patterns that separate real images from AI-generated ones. To prepare the data, each image is resized to the same size so the model can process them consistently, converted into RGB format so every image has the same 3 color channels, and turned into numbers the computer can work with. During training, I also used a simple technique called horizontal flipping, which means some images are reversed left-to-right so the model can learn from slightly varied versions of the same data and not memorize one exact appearance. Inside the model, the image passes through 3 layers that gradually look for useful visual details, starting with simpler patterns like edges and textures and then building up to more complex patterns. After that, the model sends what it learned through a small decision-making section that chooses between 2 classes: real or AI-generated. The final output is turned into probabilities, and the probability for the AI-generated class becomes the confidence score. In terms of results, the model did a good job recognizing AI-generated images and showed that the overall approach works, which is a strong starting point. However, it had a harder time correctly identifying real images and often labeled them as fake. This happened mainly because the dataset had many more AI-generated images than real ones, so the model became biased toward the larger group. Overall, the model worked well as a first baseline and proved that the pipeline is functional, but it also showed that the next step is to improve the training setup so the system becomes more balanced and reliable.

## Results
The model reached 73.23% overall accuracy on the test set, with a strong F1-score of 0.83 for AI-generated images. However, it performed much worse on real images, with an F1-score of 0.31, showing that the model was biased toward predicting the AI-generated class.


For Real images: 7,500 and AI-generated images: 37,500 (45,000 total test images) :

2,645 real images were correctly predicted as real
4,855 real images were incorrectly predicted as AI-generated
30,309 AI-generated images were correctly predicted as AI-generated
7,191 AI-generated images were incorrectly predicted as real
