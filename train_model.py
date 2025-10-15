import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

"""train_model.py
A beginner-friendly script to train a simple CNN that distinguishes tit birds from non-tits.
It expects images to be organized like:
    dataset/
        tits/
            image1.jpg
            ...
        not_tits/
            image1.jpg
            ...
Running:
    python train_model.py
This will train the network for a few epochs and save the best weights to 'tit_detector_cnn.pth'.
"""

# =========================
# Hyperparameters (feel free to tweak)
# -------------------------
# BATCH_SIZE – how many images the network sees at once. Larger batches train
# faster on GPUs but need more memory. 32 is a safe default for CPUs or small GPUs.
# EPOCHS – one epoch = one full pass through the training data. More epochs can
# improve accuracy but also increase training time / risk of over-fitting.
# LEARNING_RATE – how fast the network tries to adjust its weights. Too large
# can overshoot, too small can get stuck. 1e-3 is a common starting point.
# -------------------------

# =========================
# Hyperparameters
# Batch size = images processed in parallel; increase if you have GPU memory.
BATCH_SIZE = 32
# More epochs usually means better performance (to a point). We'll train for 15.
EPOCHS = 15
# Adam works well; start LR 1e-3 then we'll use a scheduler to decay it.
LEARNING_RATE = 1e-3
DATA_DIR = "dataset"
MODEL_PATH = "tit_detector_cnn.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Data preprocessing & augmentation
# -------------------------
# 1. Resize – ensures every image is exactly 128×128 pixels (matches our CNN).
# 2. RandomHorizontalFlip – a simple data-augmentation trick: randomly flips
#    images left↔right during training so the model learns that orientation
#    doesn’t matter. (Birds face left or right – still the same tit!)
# 3. ToTensor – converts PIL image (0-255) → PyTorch tensor (0-1).
# 4. Normalize – rescales pixels using the same mean / std as ImageNet so our
#    randomly initialized network starts from familiar ranges.
# =========================
# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # simple augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet
                         std=[0.229, 0.224, 0.225])
])

# =========================
# Load dataset
# ImageFolder automatically assigns class indices based on sub-folder names:
#   dataset/
#     not_tits/  -> class 0
#     tits/      -> class 1
# =========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Split into training (80%) and validation (20%) sets so we can measure
# how well the network generalizes to UNSEEN images.
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


class SimpleCNN(nn.Module):
    """A small 3-convolutional-layer CNN for binary image classification."""

    def __init__(self):
        super().__init__()
        # Improved CNN: more filters + BatchNorm + Dropout for better generalization
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),   # 128 -> 64
            conv_block(32, 64),  # 64  -> 32
            conv_block(64, 128), # 32  -> 16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)         
        )

    def forward(self, x):
        x = self.features(x)          # CNN extracts visual patterns
        return self.classifier(x)     # Fully-connected layers decide final class


# Helper to compute accuracy (fraction of correct predictions)
def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)  # highest-probability class
    return (preds == labels).float().mean()


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    # loop over small *batches* instead of full dataset → fits in memory & speeds up
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
        epoch_acc += accuracy(outputs, labels) * imgs.size(0)
    return epoch_loss / len(loader.dataset), epoch_acc / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    # no_grad → we’re only measuring, not updating weights, so save memory & time
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * imgs.size(0)
            epoch_acc += accuracy(outputs, labels) * imgs.size(0)
    return epoch_loss / len(loader.dataset), epoch_acc / len(loader.dataset)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Scheduler to slowly decrease LR for finer learning later epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0  # store as Python float for easy printing
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc_tensor = eval_epoch(model, val_loader, criterion)
        val_acc = val_acc_tensor.item()  # convert tensor → float

        # Nice progress report every epoch
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Keep the weights that perform BEST on validation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"📝 Saved new best model with Val Acc: {best_val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Step LR scheduler
        scheduler.step()

    print("Training complete! Best validation accuracy:", best_val_acc)

    # -------------------------
    # Plot training & validation loss curves for pedagogical purposes
    # -------------------------
    import matplotlib.pyplot as plt

    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("📈 Saved training curve to loss_curve.png") 