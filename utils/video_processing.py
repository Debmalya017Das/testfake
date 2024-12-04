import cv2
import os

def split_video_to_frames(video_path, output_folder):
    """Splits a video into frames."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

def compile_frames_to_video(frame_folder, output_video_path, fps=30):
    """Compiles frames into a video."""
    frames = sorted(os.listdir(frame_folder))
    if not frames:
        print("No frames found in folder.")
        return

    frame_path = os.path.join(frame_folder, frames[0])
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_file in frames:
        frame = cv2.imread(os.path.join(frame_folder, frame_file))
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video_path}")
    
    
#     import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models, transforms
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import RandomOverSampler
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# from utils.video_processing import split_video_to_frames
# from utils.face_detection import extract_faces_from_frames

# # Constants
# DATASET_DIR = "dataset"
# FRAME_DIR = "frames"
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 30
# TIME_STEPS = 10
# EPOCHS = 10
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class VideoDataset(Dataset):
#     def __init__(self, X, y, transform=None):
#         self.X = X
#         self.y = y
#         self.transform = transform

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         frames = self.X[idx]
#         label = self.y[idx]
#         if self.transform:
#             frames = torch.stack([self.transform(frame) for frame in frames])
#         return frames, label

# def preprocess_frames(frame_folder):
#     """Loads and preprocesses frames."""
#     frames = []
#     for img_file in sorted(os.listdir(frame_folder))[:TIME_STEPS]:
#         img_path = os.path.join(frame_folder, img_file)
#         if not img_file.endswith(".jpg"):
#             continue
#         img = plt.imread(img_path)
#         img = transforms.ToPILImage()(img)
#         img = transforms.Resize(IMG_SIZE)(img)
#         img = transforms.ToTensor()(img)
#         frames.append(img)
#     if len(frames) < TIME_STEPS:
#         print(f"Insufficient frames in {frame_folder}. Padding...")
#         while len(frames) < TIME_STEPS:
#             frames.append(torch.zeros(3, *IMG_SIZE))
#     return torch.stack(frames)

# def load_dataset():
#     """Loads the dataset and prepares it for training."""
#     X, y = [], []
#     for label, category in enumerate(["real", "fake"]):
#         category_folder = os.path.join(FRAME_DIR, category)
#         for video_folder in os.listdir(category_folder):
#             frame_folder = os.path.join(category_folder, video_folder)
#             if not os.path.isdir(frame_folder):
#                 continue
#             video_frames = preprocess_frames(frame_folder)
#             X.append(video_frames.numpy())
#             y.append(label)
#     X = np.array(X)
#     y = np.array(y)

#     # Oversample minority class
#     X_flat = X.reshape(X.shape[0], -1)
#     ros = RandomOverSampler(random_state=42)
#     X_resampled, y_resampled = ros.fit_resample(X_flat, y)
#     X_resampled = X_resampled.reshape(-1, TIME_STEPS, 3, *IMG_SIZE)
#     return torch.tensor(X_resampled), torch.tensor(y_resampled)

# class ResNetLSTM(nn.Module):
#     def __init__(self):
#         super(ResNetLSTM, self).__init__()
#         self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
#         self.resnet.fc = nn.Identity()  # Remove the fully connected layer
#         self.lstm = nn.LSTM(2048, 128, batch_first=True)
#         self.fc = nn.Linear(128, 1)

#     def forward(self, x):
#         batch_size, time_steps, C, H, W = x.size()
#         x = x.view(batch_size * time_steps, C, H, W)
#         features = self.resnet(x)
#         features = features.view(batch_size, time_steps, -1)
#         _, (hidden, _) = self.lstm(features)
#         out = self.fc(hidden[-1])
#         return torch.sigmoid(out)

# def train_model():
#     """Trains the ResNet-LSTM model and evaluates its performance."""
#     # Load dataset
#     X, y = load_dataset()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Data loaders
#     transform = transforms.Compose([
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     train_dataset = VideoDataset(X_train, y_train, transform=transform)
#     test_dataset = VideoDataset(X_test, y_test, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#     # Model, loss, and optimizer
#     model = ResNetLSTM().to(DEVICE)
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Training loop
#     for epoch in range(EPOCHS):
#         model.train()
#         train_loss = 0
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(DEVICE), y_batch.float().to(DEVICE)
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs.squeeze(), y_batch)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#         print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss / len(train_loader)}")

#     # Save the trained model
#     os.makedirs("models", exist_ok=True)
#     torch.save(model.state_dict(), "models/resnet_lstm_model.pth")
#     print("Model saved to models/resnet_lstm_model.pth")

#     # Evaluation
#     model.eval()
#     y_pred, y_true = [], []
#     with torch.no_grad():
#         for X_batch, y_batch in test_loader:
#             X_batch = X_batch.to(DEVICE)
#             outputs = model(X_batch)
#             y_pred.extend(outputs.cpu().numpy())
#             y_true.extend(y_batch.numpy())

#     y_pred = (np.array(y_pred).squeeze() > 0.5).astype("int")
#     accuracy = (y_pred == y_true).mean() * 100
#     print(f"Accuracy: {accuracy:.2f}%")

#     # Classification report and confusion matrix
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=["Real", "Fake"], zero_division=1))

#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(6, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")
#     plt.show()

# if __name__ == "__main__":
#     # Step 1: Process dataset (split videos into frames and extract faces)
#     for category in ["real", "fake"]:
#         for video_file in os.listdir(os.path.join(DATASET_DIR, category)):
#             video_path = os.path.join(DATASET_DIR, category, video_file)
#             output_folder = os.path.join(FRAME_DIR, category, os.path.splitext(video_file)[0])
#             split_video_to_frames(video_path, output_folder)
#             extract_faces_from_frames(output_folder, output_folder)

#     # Step 2: Train the model
#     train_model()