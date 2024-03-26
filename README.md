Building and Deploying CIFAR-10 Image Classification Model on Amazon SageMaker
Introduction
This repository contains code to train, deploy, and evaluate a CIFAR-10 image classification model using Amazon SageMaker. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The objective is to train a deep learning model that can accurately classify these images into their respective categories.

Contents
Training Code: The cifar10_keras_main.py file contains the main training script for the CIFAR-10 model using TensorFlow and Keras.
Source Directory: The source_dir directory includes additional source code and dependencies required for training and deployment.
Notebook: The notebook contains code to interact with Amazon SageMaker for training, deployment, and evaluation of the CIFAR-10 model.
Setup
Amazon SageMaker: Ensure you have an active AWS account and access to Amazon SageMaker services.
Data Preparation: Prepare the CIFAR-10 dataset in the required format and store it in an accessible location.
Environment Setup: Set up your Python environment with necessary libraries, including sagemaker, tensorflow, keras, and numpy.
Usage
Training Locally: Train the model locally using TensorFlow on your local machine with instance_type = "local".
Training on SageMaker: Train the model on Amazon SageMaker by setting up a SageMaker TensorFlow estimator and specifying instance configurations.
Evaluation: Evaluate the trained model's performance using various metrics, including accuracy and confusion matrix.
Deployment: Deploy the trained model as an endpoint on Amazon SageMaker for real-time inference.
Prediction: Make predictions using the deployed endpoint on new data or test datasets.
Cleanup: Delete the SageMaker endpoint after use to avoid unnecessary charges.
Note
Ensure proper data preprocessing and model optimization for best results.
Experiment with different hyperparameters and model architectures to improve performance.
Monitor training jobs and endpoint metrics using Amazon CloudWatch for better insights and optimization.
