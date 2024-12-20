# Face_recommendation
Face Recognition with PCA and SVM

This project demonstrates a face recognition pipeline using Principal Component Analysis (PCA) for dimensionality reduction and a Support Vector Machine (SVM) for classification. The dataset used is the Labeled Faces in the Wild (LFW) dataset.

Features

Dataset:

Utilizes the Labeled Faces in the Wild dataset, which contains labeled face images of various individuals.

Filters the dataset to include individuals with at least 60 images.

Data Visualization:

Displays face images in a grid using Matplotlib for easy inspection.

Dimensionality Reduction:

Applies PCA to reduce the dimensionality of the data and normalize the images.

Classification:

Uses a Support Vector Machine (SVM) with an RBF kernel to classify faces.

Hyperparameter Optimization:

Employs GridSearchCV to optimize the C and gamma parameters of the SVM model.

Evaluation:

Provides a classification report with metrics such as precision, recall, and F1-score.

Requirements

To run this project, you need the following libraries:

scikit-learn

matplotlib

numpy

Install the required libraries using pip:

pip install scikit-learn matplotlib numpy

Steps to Run

Clone the repository:

git clone https://github.com/yourusername/face-recognition.git
cd face-recognition

Open the Jupyter Notebook:

jupyter notebook face_recognition.ipynb

Follow the steps in the notebook:

Load and preprocess the dataset.

Visualize face images.

Perform dimensionality reduction using PCA.

Train the SVM classifier.

Evaluate the model using a classification report.

Explore the results:

View the grid of face images with predictions.

Analyze the detailed classification metrics.

Project Structure

face-recognition/
|-- face_recognition.ipynb  # Jupyter Notebook containing the implementation
|-- README.md               # Project documentation

Example Output

Face Image Grid:
A visualization of face images with corresponding predictions.

Classification Report:
Detailed metrics for each class, including precision, recall, and F1-score.

Acknowledgements

Scikit-learn for providing machine learning tools.

The LFW dataset creators for making the dataset publicly available.

