# import os
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
# BATCH_SIZE = 16
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
#     # for category in ["real", "fake"]:
#     #     for video_file in os.listdir(os.path.join(DATASET_DIR, category)):
#     #         video_path = os.path.join(DATASET_DIR, category, video_file)
#     #         output_folder = os.path.join(FRAME_DIR, category, os.path.splitext(video_file)[0])
#     #         split_video_to_frames(video_path, output_folder)
#     #         extract_faces_from_frames(output_folder, output_folder)

#     # Step 2: Train the model
#     train_model()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Constants
DATASET_DIR = "dataset"
FRAME_DIR = "frames"
IMG_SIZE = (224, 224)
BATCH_SIZE = 20
TIME_STEPS = 10
EPOCHS = 15  # Slightly increased epochs
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

def split_video_to_frames(video_path, output_folder, max_frames=100):
    """Splits a video into frames with a maximum frame limit."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

def preprocess_frames(frame_folder):
    """Loads and preprocesses frames with robust handling."""
    frames = []
    all_frames = [f for f in sorted(os.listdir(frame_folder)) if f.endswith('.jpg')]
    
    # Improved frame selection strategy
    if len(all_frames) > TIME_STEPS:
        selected_indices = np.linspace(0, len(all_frames) - 1, TIME_STEPS, dtype=int)
        selected_frames = [all_frames[i] for i in selected_indices]
    else:
        selected_frames = all_frames

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMG_SIZE),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for img_file in selected_frames:
        img_path = os.path.join(frame_folder, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img)
        frames.append(img_tensor)

    # Padding with last frame if insufficient
    while len(frames) < TIME_STEPS:
        frames.append(frames[-1])

    return torch.stack(frames)

class VideoDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        frames = self.X[idx]
        label = self.y[idx]
        return frames, label

def load_dataset():
    """Loads the dataset with improved balancing and validation."""
    X, y = [], []
    for label, category in enumerate(["real", "fake"]):
        category_folder = os.path.join(FRAME_DIR, category)
        
        for video_folder in os.listdir(category_folder):
            frame_folder = os.path.join(category_folder, video_folder)
            
            if not os.path.isdir(frame_folder):
                continue
            
            try:
                video_frames = preprocess_frames(frame_folder)
                X.append(video_frames.numpy())
                y.append(label)
            except Exception as e:
                print(f"Error processing {frame_folder}: {e}")
    
    if len(X) == 0:
        raise ValueError("No frames were loaded from the dataset. Check dataset preparation.")
    
    X = np.array(X)
    y = np.array(y)

    # Advanced resampling with stratification
    X_flat = X.reshape(X.shape[0], -1)
    ros = RandomOverSampler(random_state=RANDOM_SEED)
    X_resampled, y_resampled = ros.fit_resample(X_flat, y)
    X_resampled = X_resampled.reshape(-1, TIME_STEPS, 3, *IMG_SIZE)
    
    return torch.tensor(X_resampled), torch.tensor(y_resampled)

class EnhancedResNetLSTM(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(EnhancedResNetLSTM, self).__init__()
        # Pretrained ResNet50 with more extensive fine-tuning
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        
        # More comprehensive fine-tuning strategy
        # Gradually unfreeze more layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze multiple stages for more adaptability
        for module in [self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            for param in module.parameters():
                param.requires_grad = True
        
        # Remove the fully connected layer
        self.resnet.fc = nn.Identity()
        
        # Advanced LSTM with increased complexity
        self.lstm = nn.LSTM(
            input_size=2048, 
            hidden_size=768,  # Increased hidden size 
            num_layers=3,     # Increased number of layers
            batch_first=True, 
            dropout=dropout_rate
        )
        
        # More sophisticated classifier with additional layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        
        # Feature extraction
        x = x.view(batch_size * time_steps, C, H, W)
        features = self.resnet(x)
        features = features.view(batch_size, time_steps, -1)
        
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(features)
        
        # Use last hidden state for classification
        out = self.classifier(hidden[-1])
        return out

def train_model():
    """Comprehensive model training with enhanced strategies."""
    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Load the entire dataset
    X, y = load_dataset()

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=RANDOM_SEED, 
        stratify=y
    )

    # Data loaders
    train_dataset = VideoDataset(X_train, y_train)
    test_dataset = VideoDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model initialization
    model = EnhancedResNetLSTM().to(DEVICE)
    
    # More sophisticated optimization
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=LEARNING_RATE, 
        weight_decay=1e-4,  # Adjusted weight decay
        betas=(0.9, 0.999)  # More stable optimization
    )
    
    criterion = nn.BCELoss()
    
    # Cosine annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS, 
        eta_min=LEARNING_RATE * 0.1
    )

    best_val_loss = float('inf')
    early_stopping_counter = 0
    max_early_stopping = 8

    # Comprehensive training loop
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.float().to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        y_preds, y_trues = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.float().to(DEVICE)
                
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                
                val_losses.append(loss.item())
                y_preds.extend(outputs.cpu().numpy())
                y_trues.extend(y_batch.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Step the scheduler
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), "models/best_deepfake_model.pth")
        else:
            early_stopping_counter += 1

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if early_stopping_counter >= max_early_stopping:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Final model evaluation
    model.load_state_dict(torch.load("models/best_deepfake_model.pth"))
    model.eval()

    # Comprehensive performance metrics
    y_pred_proba = []
    y_true = []
    
    with torch.no_grad():
        for X_batch, batch_labels in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            y_pred_proba.extend(outputs.cpu().numpy())
            y_true.extend(batch_labels.numpy())

    y_pred_proba = np.array(y_pred_proba).squeeze()
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    y_true = np.array(y_true)

    # Performance metrics
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {(y_pred_binary == y_true).mean() * 100:.2f}%")
    print(f"AUC-ROC: {roc_auc_score(y_true, y_pred_proba):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=["Real", "Fake"], zero_division=1))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Real", "Fake"], 
                yticklabels=["Real", "Fake"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

def prepare_dataset():
    """Prepare dataset by extracting frames."""
    os.makedirs(FRAME_DIR, exist_ok=True)
    for category in ["real", "fake"]:
        category_path = os.path.join(FRAME_DIR, category)
        os.makedirs(category_path, exist_ok=True)

    for category in ["real", "fake"]:
        category_dir = os.path.join(DATASET_DIR, category)
        for video_file in os.listdir(category_dir):
            video_path = os.path.join(category_dir, video_file)
            
            output_folder = os.path.join(FRAME_DIR, category, os.path.splitext(video_file)[0])
            os.makedirs(output_folder, exist_ok=True)
            
            split_video_to_frames(video_path, output_folder)

    print("Dataset preparation complete.")

if __name__ == "__main__":
    # Prepare dataset
    # prepare_dataset()
    
    # Train the model
    train_model()