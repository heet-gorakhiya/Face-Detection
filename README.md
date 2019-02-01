# Face-Detection

### Face detection using Support Vector Machines on small dataset: 

The dataset used here is a custom "emotions" dataset, collected from 43 people, collecting 12 images of everyone - 6 images displaying different emotions and 6 images derived from the former by applying "sketch" filter.

## Overview - Short Description

This is an attempt on solving the very well known face recognition problem in the Machine Learning paradigm. Our approach is based on the Support Vector Machine (SVM) Algorithm which is a classifying algorithm for ML framework. In this document we discuss our approach and its outcomes as well as our journey through the assignment, i.e., what different methods we used before we decided our final approach as well as the challenges we faced implementing them. In this assignment, we take a data set of 588 images, each with one of the 43 people’s faces and on that we apply our SVM Machine Learning algorithm to classify and identify each person uniquely. To test the results and accuracy of face detection, we give a different picture of a person’s face as an input and find out how many times it identifies the person correctly.

## Algorithm

![Workflow](https://github.com/Grimmjaw6/Face-Detection/blob/master/Workflow.png)

### Pre-processing Image data 

The data is located in the "data" directory in the same folder. 

1. Load thae data in "files" object
2. For every entry in the object:
  2a: Resized to 100 x 100 size 
  2b. Convert to grayscale
  2c. Flatten into numpy array
3. The numpy arrays for each image is appended into one "NN_data.npy" file

### Main Model

1. Load data from "NN_data.npy" file
2. Load into dataframe for one-hot encoding
3. Perform PCA by specifying number of features
4. Split Train and Test data
5. Train SVM model on Train set
6. To test the model:
  6a. Apply Haar-cascade on validation image
  ![Haar-cascade Visualisation](https://github.com/Grimmjaw6/Face-Detection/blob/master/Haar-cascade.png)
  6b. Predict.
  
  
### Fine Tuning PCA features

![PCA optimum features](https://github.com/Grimmjaw6/Face-Detection/blob/master/PCA_optimum_features.png)
