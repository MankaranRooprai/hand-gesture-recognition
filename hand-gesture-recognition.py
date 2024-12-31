import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import os
from collections import Counter
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Define the CNN model with dropout and extra layers
class GestureModel(nn.Module):
    def __init__(self):
        super(GestureModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Added another convolution layer
        self.fc1 = nn.Linear(128 * 64 * 64, 128)  # Assuming 64x64 image input
        self.fc2 = nn.Linear(128, 29)  # 29 classes (A-Z + nothing, space, del)
        self.dropout = nn.Dropout(p=0.5)  # Dropout to prevent overfitting
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))  # Third convolution layer
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

# 2. Preprocessing (Resize and normalize)
def preprocess_image(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Ensure this normalization is applied correctly
    ])
    return transform(frame).unsqueeze(0)  # Add batch dimension

# 3. Train the model
def train_model():
    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder('gesture_dataset/asl_alphabet_train/asl_alphabet_train', transform=transform)

    # Debugging: Print class distribution to ensure all gestures (A-Z) are present
    class_counts = Counter(dataset.targets)
    print(f"Class distribution: {class_counts}")

    # Calculate class weights based on the dataset class distribution
    total_samples = len(dataset)
    class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]

    # Convert to tensor and use them in the CrossEntropyLoss
    weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Create the dataloader with a larger batch size (e.g., 32 or 64)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = GestureModel().to(device)  # Move model to GPU or CPU
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for stable training

    # Train the model
    num_epochs = 5  # Increase epochs for better training
    for epoch in tqdm(range(num_epochs), desc="Training..."):
        print("Epoch:", epoch)
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader, desc="Loading..."):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU or CPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'gesture_model.pth')
    print("Model trained and saved as gesture_model.pth")

# 4. Load the pre-trained model
def load_model():
    model = GestureModel()
    model.load_state_dict(torch.load('gesture_model.pth', weights_only=True))
    model.eval()
    return model

# 5. Capture video and classify gestures
def recognize_gesture(model):
    # Move the model to the correct device (GPU or CPU)
    model = model.to(device)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        hand_region = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("Hand Region", hand_region)

        hand_region_resized = cv2.resize(hand_region, (64, 64))  # Resize to match training size
        input_tensor = preprocess_image(hand_region_resized).to(device)  # Ensure input tensor is on the correct device

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            gesture_label = predicted.item()

        # The gestures list will now include A-Z + nothing, space, and delete (29 classes total)
        gestures = [chr(i) for i in range(65, 91)] + ['nothing', 'space', 'del']  # A-Z + additional labels
        gesture_name = gestures[gesture_label]

        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 6. Main function to choose between training or recognizing gestures
def main():
    if not os.path.exists('gesture_model.pth'):
        # Train the model if it doesn't exist
        train_model()
    
    # Load the trained model
    model = load_model()
    print('Model loaded')

    print('Now recognizing hand gestures')
    # Recognize hand gestures
    recognize_gesture(model)

if __name__ == "__main__":
    main()
