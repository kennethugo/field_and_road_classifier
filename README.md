# Two class classifier: Field & Road

This repository shows the step by step process of creating a binary classifier that classifiers image of field and road. This is a technical exercise.  The solution provided in this repository tackles the problem of working with very small and highly imbalance image dataset.

# Table of Contents
* [**Introduction**](##Introduction)
* [**Getting Started**](##Getting-Started)
* [**Methodology and Results**](##Methodology-and-Results)
* [**Conclusion** ](##Conclusion)
* [**Further work** ](##Further-work)

## Introduction
This project aims to classify images of roads and fields using computer vision techniques. It also looks at ways to solve data imbalance while training with a small dataset. The main task in this exercise is not just about creating a two class road and filed image classifier but also the ability to work with significantly imbalanced small dataset in a way to have good prediction results. In addition to applying techniques that handles data imbalance in this exercise, I also applied transfer learning and fine tuning technique because of the small number of the dataset. The weight of a pre-trained MobilenetV2 that has been pre-trained on ImageNet’s dataset was used to customize and train the field and road classifier model. I used MobilenetV2 for transfer learning because of it’s designed to provide fast and computationally efficient performance, even good for lightweight deep learning tasks and can provide faster inference on resource-constrained edge devices. MobileNetv2's depth-wise separable convolutions, which reduce the computational complexity is a good factor to consider when looking for a Convolutional neural network architecture that can provide fast Inference and suitable for real-time applications. 

The dataset that has been made available for this computer vision task has two classes namely field and road. Originally, there are 45 image files in the field image folder and 108 image files in the road folders, also a test folder with 10 images of both fields and roads have been provided. There is a significant level of data imbalance in the original dataset, which will make the success of the exercise dependent on the preprocessing of this dataset that is not just small but also highly imbalance. I will explore techniques like data augmentation, oversampling and class weighting to try to reduce the effect of the data imbalance on the training dataset and results. During fine-tuning regularization techniques like dropout and l2 regularization were tried out to avoid overfitting of the model. At the end, metrics like accuracy, precision and recall were used to evaluate the performance of the model. 


## Getting Started
The following steps will help you get started:

1. Open the notebook in Goggle colab
https://colab.research.google.com/drive/1YRQlMInx7nfsf2qsBLDw9lx33a7FetnB#scrollTo=Zipuq1NHtqDq
3. Codes and comments are available to explain the different steps:

* Importing packages
* Downloading dataset
* Creating and splitting the dataset 
* Calculating class weights
* Data Preprocessing and Augmentation 
* Metrics
* Model Building
* Model training
* Fine tuning
* Evaluation and validation tests
* Inference

## Methodology and Results
### Dataset

The Road and Field Binary Classification Project involve the categorization of images into two classes: roads and fields. The dataset used for this project consists of a diverse collection of road and field images provided by Trimble / Bilberry. The dataset show 45 images of fields and 108 images of road. 

### Data Preprocessing

Prior to training the classification model, extensive data preprocessing was performed. First of all, in solving the problem of data imbalance, I considered techniques like oversampling and class weighting in addition to data augmentation. Due to the small number of the dataset, applying data augmentation techniques to artificially increase the size and diversity of the training dataset was inevitable. So I used common augmentations like random rotations, flips, zooms to help augment and improve diversity and model's robustness. The images were resized to a consistent resolution, maintaining same size of (244, 244) mobileNETv2 was preprocessed with. 

After oversampling the dataset using the Borderline SMOTE method, both classes had equal number of examples but going through the oversampled images, I realized the qualities were poor and there may be no new information for the model to learn and this can further increase over fitting. This left me with the option of class weighting. Class weighting will assign different weights to classes during training, giving higher weight to the minority class, which effectively increases its importance during gradient updates. So I used the class weighting technique by calculating class weight of the two classes and applied this during training. This reduced over fitting and bias towards the majority Class and also reduce poor generalization to the minority class.. The best solution to handle the data imbalance could have been to generate or collect more data to add to the minority field class, but It was not stated if this option was allowed in this exercise. 
Going forward, the dataset was split into training and validation sets using the tf.keras image_dataset_from_directory. As best practice, the same preprocessing that was used during training of mobilenetV2 was also applied.  This helps achieve desired result since transfer learning of the mobile will be used.

### Model Architecture

Transfer learning approach was adopted using the MobileNetV2 architecture, a pre-trained Convolutional neural network (CNN) model. I used MobilenetV2 for transfer learning because of its designed to provide fast and computationally efficient performance, even good for lightweight deep learning tasks. The pre-trained MobileNetV2 was fine-tuned for the binary classification task. To enhance the model's performance, additional layers were appended, including global average pooling, dropout layers were added, and a dense layer. During fine-tuning, more dropout values and dense layers with L2 regularization were experimented with to reduce overfitting and improve models performance on validation datasets.

### Training and Evaluation

The model was trained using a combination of the fine-tuned architecture and the Adam optimizer. A Binary Cross-Entropy loss function was employed, and class weights were utilized to address the class imbalance between roads and fields. The training process was monitored using metrics such as accuracy, precision, recall. Tensorboard was used to visualize the curves and graphs of the various metrics. class weighting was used to assign different weights to different classes during training.  This is to reduce bias toward the Majority class when predicting that majority class. This will also stop the tendency of the model not generalize well to the minority class.

The model's performance was evaluated on a validation dataset that was distinct from the training data. Early stopping was implemented to prevent overfitting, and the best weights were restored to ensure optimal generalization.

### Results

Considering the size of the overall dataset, the model achieved impressive results in classifying road and field images. Although overfitting was not completely removed, bias towards predicting the majority class (field class) was eliminated using class weighting. Poor generalization of the model to the minority class (the road class) was also reduced; there was no overfittiing to the majority class for a model like this.  Overfitting was generally reduced and controlled during fine-tuning using dropout and l2 regularization. The evaluation metrics showcased the model's effectiveness in distinguishing between the two classes. The precision, recall, and F1-score demonstrated the model's ability to provide reliable predictions.



## Conclusion

The Road and Field Binary Classification Project demonstrates the application of transfer learning and advanced computer vision techniques to the task of classifying road and field images. This exercise also provides solution to solve the problems that could be caused by small and imbalance datasets like overfitting and bias toward a majority Class, or poor generalization of the model to the minority class etc. By leveraging a pre-trained model, fine-tuning, and implementing data augmentation and class weighting impressive classification results were achieved. The project's methodology serves as a valuable foundation for future work or similar image classification tasks and can be extended to other domains. There are rooms for Improvement too.

For detailed code implementation and usage instructions, refer to the [**Getting Started**](##Getting-Started) section above.


## Further work
While this project has yielded promising results, there is ample room for future improvements and enhancements.

