# IMAGE CLASSIFIER USING PYTORCH (END TO END PROJECT)

# Image Classifier for Cats and Dogs using PyTorch and Comet.ml

## Introduction

This project implements an image classifier for distinguishing between cats and dogs using PyTorch, Comet.ml for experiment tracking, and Gradio for deploying the model as a web application. The README provides a detailed overview of the project, including its purpose, architecture, training process, and deployment.

### Comet.ml Integration

Comet.ml was integrated into the project using an API key to track and monitor machine learning experiments. The Comet.ml project link for this specific project can be found [here](https://www.comet.com/talw98/image-classifier-of-dogs-and-cats/view/new/panels). Comet.ml offers comprehensive experiment tracking, visualization, and model monitoring capabilities, making it a valuable tool for managing machine learning projects.

### Dataset

The dataset used for training the image classifier was obtained from Kaggle. The dataset link is available [here](https://www.kaggle.com/datasets/tongpython/cat-and-dog). It contains a collection of images of cats and dogs, which are used to train and evaluate the image classifier model.

### Model Architecture

The image classifier model architecture is a convolutional neural network (CNN) designed to extract features from input images and classify them into cat or dog categories. The architecture consists of the following layers:

- Three convolutional layers with ReLU activation and batch normalization.
- Max-pooling layers for downsampling.
- A fully connected layer for classification.

The total number of parameters in the model is 2,668,418. The detailed layer-wise architecture and parameter count can be found below:

```
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ImageClassifier                          [1, 2]                    --
├─Sequential: 1-1                        [1, 64, 112, 112]         --
│    └─Conv2d: 2-1                       [1, 64, 224, 224]         1,792
│    └─ReLU: 2-2                         [1, 64, 224, 224]         --
│    └─BatchNorm2d: 2-3                  [1, 64, 224, 224]         128
│    └─MaxPool2d: 2-4                    [1, 64, 112, 112]         --
├─Sequential: 1-2                        [1, 512, 56, 56]          --
│    └─Conv2d: 2-5                       [1, 512, 112, 112]        295,424
│    └─ReLU: 2-6                         [1, 512, 112, 112]        --
│    └─BatchNorm2d: 2-7                  [1, 512, 112, 112]        1,024
│    └─MaxPool2d: 2-8                    [1, 512, 56, 56]          --
├─Sequential: 1-3                        [1, 512, 28, 28]          --
│    └─Conv2d: 2-9                       [1, 512, 56, 56]          2,359,808
│    └─ReLU: 2-10                        [1, 512, 56, 56]          --
│    └─BatchNorm2d: 2-11                 [1, 512, 56, 56]          1,024
│    └─MaxPool2d: 2-12                   [1, 512, 28, 28]          --
├─Sequential: 1-4                        [1, 512, 14, 14]          (recursive)
│    └─Conv2d: 2-13                      [1, 512, 28, 28]          (recursive)
│    └─ReLU: 2-14                        [1, 512, 28, 28]          --
│    └─BatchNorm2d: 2-15                 [1, 512, 28, 28]          (recursive)
│    └─MaxPool2d: 2-16                   [1, 512, 14, 14]          --
├─Sequential: 1-5                        [1, 512, 7, 7]            (recursive)
│    └─Conv2d: 2-17                      [1, 512, 14, 14]          (recursive)
│    └─ReLU: 2-18                        [1, 512, 14, 14]          --
│    └─BatchNorm2d: 2-19                 [1, 512, 14, 14]          (recursive)
│    └─MaxPool2d: 2-20                   [1, 512, 7, 7]            --
├─Sequential: 1-6                        [1, 512, 3, 3]            (recursive)
│    └─Conv2d: 2-21                      [1, 512, 7, 7]            (recursive)
│    └─ReLU: 2-22                        [1, 512, 7, 7]            --
│    └─BatchNorm2d: 2-23                 [1, 512, 7, 7]            (recursive)
│    └─MaxPool2d: 2-24                   [1, 512, 3, 3]            --
├─Sequential: 1-7                        [1, 2]                    --
│    └─Flatten: 2-25                     [1, 4608]                 --
│    └─Linear: 2-26                      [1, 2]                    9,218
==========================================================================================
Total params: 2,668,418
Trainable params: 2,668,418
Non-trainable params: 0
Total mult-adds (G): 13.62
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 188.26
Params size (MB): 10.67
Estimated Total Size (MB): 199.54
```

### Training Process

The model was trained using the provided dataset with the following hyperparameters:

- Batch size: 32
- Number of epochs: 20
- Learning rate: 1e-4

After 20 epochs of training, the model achieved a training loss of 0.1495 and a test loss of 0.1815. The accuracy on both the training and test sets was 93%.

### Deployment using Gradio

The trained image classifier model is deployed as a web application using Gradio. Gradio simplifies the deployment process by providing a user-friendly interface for interacting with machine learning models. Users can upload images and receive predictions from the deployed model in real-time via the web application.

### Benefits of Comet.ml
Comet.ml offers several benefits for managing machine learning projects:

**Experiment Tracking**: Comet.ml tracks and logs experiment details, including hyperparameters, metrics, and model versions. This enables reproducibility and facilitates collaboration among team members.

**Visualisation**: Comet.ml provides interactive visualizations of experiment results, including loss curves, accuracy metrics, and model performance over time. These visualizations aid in understanding the behavior of the model during training and testing phases.

**Model Monitoring**: Comet.ml allows for real-time monitoring of model performance and health. Teams can set up alerts and notifications based on predefined thresholds to ensure the stability and reliability of deployed models.

**Hyperparameter Optimization**: Comet.ml supports hyperparameter optimization techniques, such as grid search and random search. It helps in finding the optimal set of hyperparameters for improving model performance.

Overall, Comet.ml enhances the efficiency and effectiveness of machine learning projects by providing comprehensive tools for experiment management and monitoring.

## Conclusion

This project demonstrates the end-to-end process of developing an image classifier for cats and dogs using PyTorch, Comet.ml, and Gradio. By leveraging Comet
