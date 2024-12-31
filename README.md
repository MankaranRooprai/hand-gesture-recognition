# Real-Time Hand Gesture Recognition using Deep Learning

This project implements a real-time hand gesture recognition system using deep learning. It uses a convolutional neural network (CNN) to classify hand gestures in video frames captured via a webcam. The model is trained on the ASL (American Sign Language) alphabet, with additional classes for space and delete gestures. The project demonstrates the end-to-end pipeline of training, model deployment, and real-time recognition.

## Features

- **Real-time Gesture Recognition**: The system captures video from a webcam and detects hand gestures in real-time.
- **Deep Learning Model**: A CNN model trained on 29 classes, including A-Z (ASL Alphabet), space, and delete.
- **Preprocessing**: The system uses OpenCV for preprocessing and segmenting the hand region in each frame.
- **GPU Support**: The model is optimized to run on CUDA-enabled GPUs for faster training and inference.

## Requirements

- Python 3.x
- PyTorch 1.8+ (with GPU support for faster training)
- OpenCV
- tqdm
- numpy
- torchvision

### Installation

**Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```
   
![image](https://github.com/user-attachments/assets/5db52b74-1ea6-49e0-8d5b-0ca7a30b46df)

## Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # For Linux/MacOS
.venv\Scripts\activate     # For Windows
```

### Install the dependencies:

```bash
pip install -r requirements.txt
```

### Training the Model

The model can be trained using the following command:
```bash
python hand-gesture-recognition.py
```

This will:

Load the dataset (make sure you have the ASL Alphabet dataset in the correct directory).
Train the CNN model on the dataset.
Save the trained model as gesture_model.pth.


### Real-time Gesture Recognition

Once the model is trained, you can use it for real-time hand gesture recognition by running:
```bash
python hand-gesture-recognition.py
```

This will open a webcam window where you can show hand gestures to classify them. Press q to quit the program.

## Data

This model is trained on the **ASL Alphabet Dataset**, which contains images of hand gestures for each letter in the American Sign Language alphabet. In addition to the 26 letters, the dataset includes gestures for "space" and "delete."

## Model Architecture

The model is a simple Convolutional Neural Network (CNN) with:

- Two convolutional layers with ReLU activations
- Fully connected layers for classification (29 output classes)

## Performance

- **Training Time**: The model was trained for 10 epochs with a batch size of 32.
- **Accuracy**: The model achieves a high accuracy on the validation dataset for recognizing the 29 gestures.

## Contributions

- Developed the CNN architecture for gesture classification.
- Implemented preprocessing pipeline using OpenCV.
- Trained the model on a large dataset of hand gesture images.
- Enabled real-time gesture recognition with a live video feed.

## Future Work

- Implement a more advanced model with better accuracy.
- Add support for dynamic gestures (not just static hand shapes).
- Enhance real-time performance using optimization techniques.

