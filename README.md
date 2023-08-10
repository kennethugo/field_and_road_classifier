# Two class classifier : Field & Road

This repository shows the step by step process of creating a binary classifier that classifiers images of field and road. This is a technical exercise.  The solution provided in this repository tackles the problem of working with very small and higly imbalance image dataset.

# Table of Contents
* [**Introduction**](##Introduction)
* [**Getting Started**](##Getting-Started)
* [**Methodology and Results**](##Methodology-and-Results)
* [**Conclusion** ](##Conclusion)
* [**Further work** ](##Further-work)

## Introduction
This project aims to classify images of roads and fields using computer vision techniques. The main task in this exercise is not just about creating a two class road and filed image classifier but also the ability to work with highly imbalanced small dataset. In addition to applying techniques that handles data imbalance in this exercise,  I also applied transfer learning and fine tuning technique because of the small number of the dataset. The weights of a pre-trained MobilenetV2 that has been pre-trained on ImageNet's dataset was used to customize and train the field and road classifier model. I used MobilenetV2 for transfer learning because of it's designed to provide fast and computationally efficient performance, even good for lightweight deep learning tasks. MobileNetv2's depthwise separable convolutions, which reduce the computational complexity and the number of parameters in the network is a good factor to consider when looking for a convolutional neural network architecture that can provide fast Inference and suitable for real-time applications.  

The dataset that has been made available for this computer vision task has two classes namely field and road. Originally, there are 45 image files in the field image folder and 108 image files in the road folders, also a test folder with 10 images of both fields and roads have been provided. There is a significant level of data imbalance in the original dataset, which will make the success of the exercise dependent on the preprocessing of this dataset that is not just small  but also highly imbalance. I will explored techniques like data augmentation, oversampling and class weighting to try to reduce the effect of the data imbalance on the training dataset and results. During finetuning regularization techniques like dropout and l2 regularization were applied to check  overfitting of the model. At the end, metrics like accuracy, precision and recall were used to evaluate the performance of the model. 


## Getting Started
The  following steps will help you get started :

1. Open the notebook in Goggle colab
https://colab.research.google.com/drive/1YRQlMInx7nfsf2qsBLDw9lx33a7FetnB#scrollTo=Zipuq1NHtqDq
3. Codes and comments are available to explain the different steps :

* Importing packages
* Downloading dataset
* Creating and spliting the dataset 
* Calculating class weights
* Data Preprocessing and Augmentation 
* Metrics
* Model Building
* Model training
* Fine tuning
* Evaluation and validatio tests
* Inference

## Methodology and Results
### Dataset

The Road and Field Binary Classification Project involves the categorization of images into two classes: roads and fields. The dataset used for this project consists of a diverse collection of road and field images provided by Trimble / Bilberry. The dataset show 45 images of fields and 108 images of road 

### Data Preprocessing

Prior to training the classification model, extensive data preprocessing was performed. The images were resized to a consistent resolution, and data augmentation techniques were applied to increase the diversity of the training dataset. This step included random rotations, flips, and brightness adjustments to improve the model's robustness.

### Model Architecture

Transfer learning approach was adopted using the MobileNetV2 architecture, a pre-trained convolutional neural network (CNN) model. The pre-trained MobileNetV2 was fine-tuned for the binary classification task. To enhance the model's performance, additional layers were appended, including global average pooling, dropout layers, and dense layers with L2 regularization.

### Training and Evaluation

The model was trained using a combination of the fine-tuned architecture and the Adam optimizer. A Binary Cross-Entropy loss function was employed, and class weights were utilized to address the class imbalance between roads and fields. The training process was monitored using metrics such as accuracy, precision, recall.

The model's performance was evaluated on a validation dataset that was distinct from the training data. Early stopping was implemented to prevent overfitting, and the best weights were restored to ensure optimal generalization.

### Results

Considering the size of overall dataset, the model achieved impressive results in classifying road and field images. Although overfitting was not completely  removed, but was reduced and controlled during finetuning using dropout and l2 regularization  The evaluation metrics showcased the model's effectiveness in distinguishing between the two classes. The precision, recall, and F1-score demonstrated the model's ability to provide balanced and reliable predictions.



## Conclusion

The Road and Field Binary Classification Project demonstrates the application of transfer learning and advanced computer vision techniques to the task of classifying road and field images. By leveraging a pre-trained model, fine-tuning, and implementing data augmentation,  impressive classification results were achieved. The project's methodology serves as a valuable foundation for similar image classification tasks and can be extended to other domains. There are rooms for Improvement too.

For detailed code implementation and usage instructions, refer to the [**Getting Started**](##Getting-Started) section above.


## Further work
While this project has yielded promising results, there is ample room for future improvements and enhancements.

