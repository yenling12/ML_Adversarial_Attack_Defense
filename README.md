# Adversarial Machine Learning: Attacking the MNIST Dataset


## Problem Statement

Deep Neural Networks (DNNs) have achieved impressive performance in various image classification tasks. Despite their capabilities, DNNs are vulnerable to slight, often invisible alterations to input images that can lead to incorrect classifications. Our project aims to highlight this vulnerability by manipulating images from the MNIST dataset to mislead DNN classifiers.

## Dataset

We utilize the MNIST dataset, which comprises 70,000 grayscale images of handwritten digits (0-9) in a 28x28 pixel format. The dataset is split into 60,000 training images and 10,000 testing images.

## Models

### LeNet-5

- **Introduction**: A CNN designed for character recognition, introduced in 1998. Our project adapts the LeNet-5 architecture, using ReLU activation functions for better performance.
- **Architecture**: Consists of 7 layers, including 3 convolutional layers, 2 average pooling layers, and 2 fully connected layers.

### VGG-16

- **Introduction**: Known for its performance on the ImageNet dataset, introduced in 2015. We modify the last few layers to accommodate the MNIST dataset.
- **Architecture**: 16 weighted layers, including 13 convolutional layers and 3 fully connected layers, plus 5 max pooling layers.

### InceptionV1 (GoogLeNet)

- **Introduction**: Winner of the ILSVRC-2014 challenge, known for its efficient computation and depth.
- **Architecture**: 22 layers deep, utilizing 9 inception modules to capture features at various scales.

## Proposed Solutions

Our approach includes two adversarial attack methods:

1. **Fast Gradient Signed Method (FGSM)**: Generates adversarial images by applying calculated perturbations to the original images, aiming to deceive the DNN classifier while keeping changes visually undetectable.

2. **Deep Fool**: An iterative method that finds the minimal perturbation needed to mislead the classifier, focusing on staying close to the decision boundary with minimal changes.

## Implementation

- **Milestones**:
  - Download/pre-train models, select images, calculate loss and gradients, create attack method functions.
  - Apply attack methods, input manipulated images, analyze results, and adjust parameters as needed.
  - Perform model performance and evaluations.
  - Apply defense methods on FGSM and Randomized Smoothing on DeepFool.

## Experimental Results and Analysis

FGSM: 
- FGSM attack reduces the test accuracy from 96.53% to 21.36% with epsilon from 0 to 0.3.
- Distillation defense can achieve model robustness under the five experimental perturbations and achieve an accuracy of 98% on the test set.

Deepfool:
- The perturbations applied on the images by this attack method with the iterative approach were effective than the FGSM since the perturbation's were less visible to the naked eye.
- The accuracy of Deepfool was also better compared to the FGSM, but it required more computational resources to perturb the images.

Randomized Smoothing:
- It was discovered through experimentation that the the optimal sigma value typicall falls within the range of 0.5 to 1.
- A lower sigma value indicates that the model is more vulnerable to adversarial perturbations, while higher noise results in inferior performance in classifying images due to increased deviations.

## Future Work

The future work for this project consists of advancing the attack algorithms more efficiently in terms of parallel computations to compare the accuracies between the VGG16 and LeNet-5 models. Due to computation resource limitations, the VGG16 model could not be fully compiled due to the large number of convolution layers involved and the large size of the dataset it trains through.  

Additionally, our current implementation for this mainly focused on white box attacks, therefore, the future goal would primarily need to be focused on black box attacks where the hyperparameters dictating the internal state of the models are unknown. The next step after that would be to compare accuracies with our present implementation.

Further research is required in terms of random smoothing implementation to determine the effects of different sigma values on VGG16 model performance, which can provide deeper insights into how to enhance the model's robustness.


## Roles and Responsibilities

Linda Yen:
- Project proposal
- Trained LeNet and VGG16 models
- DeepFool Attack Implementation for both LeNet and VGG16 models
- Adversarial Training Implementation and Results for LeNet
- Test and evaluate models and visualize results
- Final project report & ppt
- README.md

Wenzheng Zhao: 
- Project proposal
- Implement FGSM based on LeNet and VGG16
- Implement distillation defense methods to improve model robustness
- Test and evaluate models and visualize results
- Final project report & ppt
- README.md, debug

Ishan Sharma:
- Project proposal
- DeepFool Attack Implementation for both LeNet and VGG16 models
- Adversarial Training Implementation and Results for VGG16
- Final project report & ppt
- README.md

Ty Roth:
- Project proposal
- Implemented Randomized Smoothing Implementation and Results
- Test and evaluate models and visualize results for Random Smoothing
- Final project report & ppt
- README.md

## Code Repository

[Project GitHub Repository](#)

