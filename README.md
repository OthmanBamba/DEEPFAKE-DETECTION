DeepFake Detection with Deep Learning
Description
This project aims to detect DeepFake videos using a deep learning model based on a Convolutional Neural Network (CNN). Original and DeepFake videos are extracted, preprocessed, and used to train a binary classification model.
Features
* Extraction and preprocessing of original and DeepFake videos
* Visualization of images extracted from videos
* Construction and training of a CNN model for classification
* Prediction on new videos
* Saving and loading the trained model
Prerequisites
Before running this project, ensure you have installed the following libraries:
bash
CopyEdit
pip install numpy pandas matplotlib seaborn nltk scikit-learn joblib tensorflow opencv-python wordcloud
Project Structure
bash
CopyEdit
/AI Project
│── /DEEP
│   ├── /original
│   ├── /deepfake
│── /images (to store extracted images)
│── deepfake_data.csv (video metadata file)
│── model.h5 (saved model after training)
│── script.py (main script for training and prediction)
Usage
1. Data Preprocessing
The script analyzes original and DeepFake videos, extracts images, and stores them in a CSV file:
python
CopyEdit
videos_data = "/Users/benothmane/Desktop/Projet IA/DEEP"
classes = ["original", "deepfake"]
output_folder = "/Users/benothmane/Desktop/Projet IA/images"
The data is then split into training and test sets.
2. Training the CNN Model
A CNN model is defined and trained on the extracted images:
python
CopyEdit
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
Training is performed over 10 epochs using Adam as the optimizer.
3. Model Evaluation
After training, the model is evaluated on the test set:
python
CopyEdit
loss, accuracy = model.evaluate(X_test, y_test)
print("Test data accuracy:", accuracy)
4. Prediction on New Videos
A video is analyzed frame by frame to predict whether it is original or DeepFake:
python
CopyEdit
model = load_model('model.h5')
predict_video("path_to_video.mp4", "model.h5")
Results
* Display of images extracted from videos
* Distribution of classes in the dataset
* Model accuracy on the test set
* Real-time detection of DeepFake videos
Authors
* Bamba Ben Othmane
License
This project is licensed under the MIT License.
