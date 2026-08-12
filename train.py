import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import BraTSDataset
from model import UNet
from metrics import soft_dice_loss, dice_coefficient, iou_score, pixel_accuracy

DATA_CSV = 'data/sample_metadata.csv'
SLICES_DIR = 'data/slices'
MODEL_DIR = 'models'
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device('cpu')

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_CSV)
train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['target'], random_state=42)

train_loader = DataLoader(BraTSDataset(train_df, SLICES_DIR, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(BraTSDataset(val_df, SLICES_DIR, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False)

model = UNet(in_channels=4, num_classes=4, base_ch=16).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
ce_loss = nn.CrossEntropyLoss()

best_val_dice = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = ce_loss(logits, labels) + soft_dice_loss(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_dice, val_iou, val_acc = 0.0, 0.0, 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            d, _ = dice_coefficient(preds, labels)
            i, _ = iou_score(preds, labels)
            a = pixel_accuracy(preds, labels)
            val_dice += d
            val_iou += i
            val_acc += a
    val_dice /= len(val_loader)
    val_iou /= len(val_loader)
    val_acc /= len(val_loader)

    print(f'epoch {epoch}/{EPOCHS} - train_loss {train_loss:.4f} - val_dice {val_dice:.4f} - val_iou {val_iou:.4f} - val_acc {val_acc:.4f}')

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
        print(f'  saved new best model (val_dice {val_dice:.4f})')

print('training complete, best val dice:', best_val_dice)