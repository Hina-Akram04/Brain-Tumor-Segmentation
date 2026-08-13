import argparse
import cv2
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.model import UNet
from src.metrics import dice_coefficient, iou_score, pixel_accuracy

IMG_SIZE = 128
MODEL_PATH = 'models/best_model.pth'
DEVICE = torch.device('cpu')

CLASS_NAMES = ['Background', 'NCR/NET', 'Edema', 'Enhancing Tumor']
CLASS_COLORS = ['black', 'red', 'green', 'blue']


def load_model():
    model = UNet(in_channels=4, num_classes=4, base_ch=16)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def preprocess_slice(h5_path):
    with h5py.File(h5_path, 'r') as f:
        image = f['image'][:].astype(np.float32)
        mask = f['mask'][:].astype(np.uint8) if 'mask' in f else None

    for c in range(image.shape[-1]):
        ch = image[..., c]
        mean, std = ch.mean(), ch.std()
        image[..., c] = (ch - mean) / (std + 1e-8)

    image_r = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    label = None
    if mask is not None:
        mask_r = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        label = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int64)
        label[mask_r[..., 0] == 1] = 1
        label[mask_r[..., 1] == 1] = 2
        label[mask_r[..., 2] == 1] = 3

    img_tensor = torch.from_numpy(image_r).permute(2, 0, 1).float().unsqueeze(0)
    return img_tensor, label, image_r


def predict(model, img_tensor):
    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).numpy()
    return pred


def visualize(image, pred, label=None, save_path=None):
    base = image[..., 1]  # FLAIR-ish channel as the background view
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)

    n_cols = 3 if label is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    cmap = mcolors.ListedColormap(CLASS_COLORS)

    axes[0].imshow(base, cmap='gray')
    axes[0].set_title('MRI slice')
    axes[0].axis('off')

    axes[1].imshow(base, cmap='gray')
    axes[1].imshow(pred, cmap=cmap, alpha=0.4, vmin=0, vmax=3)
    axes[1].set_title('Predicted mask')
    axes[1].axis('off')

    if label is not None:
        axes[2].imshow(base, cmap='gray')
        axes[2].imshow(label, cmap=cmap, alpha=0.4, vmin=0, vmax=3)
        axes[2].set_title('Ground truth')
        axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print('saved:', save_path)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='path to a .h5 slice')
    parser.add_argument('--save', type=str, default=None, help='path to save the visualization image')
    args = parser.parse_args()

    model = load_model()
    img_tensor, label, image_r = preprocess_slice(args.file)
    pred = predict(model, img_tensor)

    if label is not None:
        d, _ = dice_coefficient(torch.from_numpy(pred), torch.from_numpy(label))
        i, _ = iou_score(torch.from_numpy(pred), torch.from_numpy(label))
        a = pixel_accuracy(torch.from_numpy(pred), torch.from_numpy(label))
        print(f'dice: {d:.4f} | iou: {i:.4f} | accuracy: {a:.4f}')

    visualize(image_r, pred, label, save_path=args.save)