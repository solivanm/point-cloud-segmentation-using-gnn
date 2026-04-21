import os
import time
import math
import random
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


# ============================================================
# Configuration
# ============================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_FILE = r"your training file address"
TEST_FILE = r"your teting file address"
OUT_DIR = "outputs_pointcloud_seg"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASSES = 11
CLASS_NAMES = [
    "low veg", "impervious", "vehicle", "urban furniture", "roof",
    "facade", "shrub", "tree", "soil", "vertical surface", "chimney"
]

NUM_POINTS = 2048
BLOCK_SIZE = 20.0
STRIDE = 10.0
MIN_POINTS_IN_BLOCK = 512
MIN_UNIQUE_CLASSES_IN_BLOCK = 2
VAL_RATIO = 0.20

RARE_CLASSES = [2, 3, 8, 9, 10]   # vehicle, urban furniture, soil, vertical surface, chimney

BATCH_SIZE = 8
EPOCHS = 80
LEARNING_RATE = 5e-4              # lowered to improve stability
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0                   # Windows-safe
USE_AMP = False                   # disabled to debug and avoid NaN

K_NEIGHBORS = 16
DROPOUT = 0.30

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
CE_WEIGHT = 0.65
FOCAL_WEIGHT = 0.35
RARE_BLOCK_POWER = 2.25

SAVE_EVERY = 5
MAX_TRAIN_BLOCKS = None
MAX_TEST_BLOCKS = None


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Color map
# ============================================================
def label_to_color(label):
    segmented_img = np.empty((label.shape[0], 3), dtype=np.float32)
    segmented_img[(label == 0)] = [124 / 255, 252 / 255, 0]
    segmented_img[(label == 1)] = [120 / 255, 120 / 255, 120 / 255]
    segmented_img[(label == 2)] = [32 / 255, 178 / 255, 170 / 255]
    segmented_img[(label == 3)] = [199 / 255, 21 / 255, 133 / 255]
    segmented_img[(label == 4)] = [255 / 255, 99 / 255, 71 / 255]
    segmented_img[(label == 5)] = [255 / 255, 222 / 255, 173 / 255]
    segmented_img[(label == 6)] = [107 / 255, 142 / 255, 35 / 255]
    segmented_img[(label == 7)] = [0 / 255, 255 / 255, 0]
    segmented_img[(label == 8)] = [205 / 255, 133 / 255, 63 / 255]
    segmented_img[(label == 9)] = [254 / 255, 254 / 255, 51 / 255]
    segmented_img[(label == 10)] = [178 / 255, 34 / 255, 34 / 255]
    return segmented_img


# ============================================================
# Read TXT
# Expected columns: x y z r g b ... label
# Uses first 6 columns as xyzrgb and last column as label
# ============================================================
def read_txt_point_cloud(path):
    print(f"Loading: {path}")
    data = np.loadtxt(path)
    xyz = data[:, :3].astype(np.float32)
    rgb = data[:, 3:6].astype(np.float32)
    labels = data[:, -1].astype(np.int64)

    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    return xyz, rgb, labels


# ============================================================
# Block generation
# ============================================================
def build_blocks(
    xyz,
    rgb,
    labels,
    block_size=20.0,
    stride=10.0,
    num_points=2048,
    min_points_in_block=512,
    max_blocks=None,
    min_unique_classes=2,
):
    coord_min = xyz.min(axis=0)
    xy = xyz[:, :2]

    cell_xy = np.floor((xy - coord_min[:2]) / stride).astype(np.int32)
    cell_dict = defaultdict(list)

    for i, c in enumerate(cell_xy):
        cell_dict[(int(c[0]), int(c[1]))].append(i)

    for key in list(cell_dict.keys()):
        cell_dict[key] = np.asarray(cell_dict[key], dtype=np.int32)

    cells_per_side = max(1, int(np.ceil(block_size / stride)))

    blocks = []
    total_windows = 0
    occupied_cells = sorted(cell_dict.keys())

    for cx, cy in occupied_cells:
        idx_parts = []

        for dx in range(cells_per_side):
            for dy in range(cells_per_side):
                key = (cx + dx, cy + dy)
                if key in cell_dict:
                    idx_parts.append(cell_dict[key])

        total_windows += 1

        if not idx_parts:
            continue

        idx = np.concatenate(idx_parts)

        if idx.shape[0] < min_points_in_block:
            continue

        block_labels_raw = labels[idx]
        if np.unique(block_labels_raw).shape[0] < min_unique_classes:
            continue

        if idx.shape[0] >= num_points:
            chosen = np.random.choice(idx, num_points, replace=False)
        else:
            extra = np.random.choice(idx, num_points - idx.shape[0], replace=True)
            chosen = np.concatenate([idx, extra])

        block_xyz = xyz[chosen]
        block_rgb = rgb[chosen]
        block_label = labels[chosen]
        blocks.append((block_xyz, block_rgb, block_label))

        if max_blocks is not None and len(blocks) >= max_blocks:
            print(f"Reached block cap: {max_blocks}")
            print(f"Occupied-cell anchors checked: {total_windows}")
            print(f"Valid blocks created         : {len(blocks)}")
            return blocks

    print(f"Occupied-cell anchors checked: {total_windows}")
    print(f"Valid blocks created         : {len(blocks)}")
    return blocks


# ============================================================
# Stats and class weights
# ============================================================
def compute_global_stats(blocks):
    all_xyz = np.concatenate([b[0] for b in blocks], axis=0)
    all_rgb = np.concatenate([b[1] for b in blocks], axis=0)

    xyz_mean = all_xyz.mean(axis=0).astype(np.float32)
    xyz_std = all_xyz.std(axis=0).astype(np.float32) + 1e-6
    rgb_mean = all_rgb.mean(axis=0).astype(np.float32)
    rgb_std = all_rgb.std(axis=0).astype(np.float32) + 1e-6

    return {
        "xyz_mean": xyz_mean,
        "xyz_std": xyz_std,
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
    }


def compute_class_weights(blocks, num_classes=11):
    counts = np.zeros(num_classes, dtype=np.int64)

    for _, _, y in blocks:
        counts += np.bincount(y, minlength=num_classes)

    freq = counts / max(counts.sum(), 1)
    weights = 1.0 / np.log(1.2 + freq + 1e-12)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    weights = weights / max(weights.mean(), 1e-6)

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# Dataset
# Features:
# xyz_norm (3)
# rgb_norm (3)
# xyz_rel  (3)
# z_norm_local (1)
# z_rel (1)
# => 11 channels
# ============================================================
class PointBlockDataset(Dataset):
    def __init__(self, blocks, stats, augment=False):
        self.blocks = blocks
        self.stats = stats
        self.augment = augment

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        xyz, rgb, label = self.blocks[idx]
        xyz = xyz.copy()
        rgb = rgb.copy()
        label = label.copy()

        if self.augment:
            xyz = self.apply_augmentation(xyz)

        xyz_norm = (xyz - self.stats["xyz_mean"]) / self.stats["xyz_std"]
        rgb_norm = (rgb - self.stats["rgb_mean"]) / self.stats["rgb_std"]

        block_center = xyz.mean(axis=0, keepdims=True)
        xyz_rel = xyz - block_center

        z = xyz[:, 2:3]
        z_min = z.min(axis=0, keepdims=True)
        z_max = z.max(axis=0, keepdims=True)
        z_norm_local = (z - z_min) / (z_max - z_min + 1e-6)
        z_rel = z - z.mean(axis=0, keepdims=True)

        features = np.concatenate(
            [xyz_norm, rgb_norm, xyz_rel, z_norm_local, z_rel],
            axis=1
        ).astype(np.float32)

        return (
            torch.from_numpy(features),
            torch.from_numpy(label.astype(np.int64)),
            torch.from_numpy(xyz.astype(np.float32)),
        )

    @staticmethod
    def apply_augmentation(xyz):
        if np.random.rand() < 0.5:
            theta = np.random.uniform(-np.pi / 12, np.pi / 12)
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array(
                [[c, -s, 0],
                 [s,  c, 0],
                 [0,  0, 1]],
                dtype=np.float32,
            )
            xyz = xyz @ rot.T

        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.95, 1.05)
            xyz = xyz * scale

        if np.random.rand() < 0.5:
            jitter = np.random.normal(0.0, 0.01, size=xyz.shape).astype(np.float32)
            xyz = xyz + jitter

        return xyz


# ============================================================
# kNN utils
# ============================================================
def knn_indices(xyz, k):
    with torch.no_grad():
        dist = torch.cdist(xyz, xyz)
        idx = dist.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]
    return idx


def index_points(points, idx):
    B = points.shape[0]
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand_as(idx)
    return points[batch_indices, idx]


# ============================================================
# Model blocks
# ============================================================
class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.attn(x)
        return x * w


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, xyz):
        idx = knn_indices(xyz, self.k)

        x_t = x.transpose(1, 2).contiguous()
        neighbors = index_points(x_t, idx)
        center = x_t.unsqueeze(2).expand(-1, -1, self.k, -1)
        edge = torch.cat([center, neighbors - center], dim=-1)
        edge = edge.permute(0, 3, 1, 2).contiguous()

        feat = self.mlp(edge)
        feat = torch.max(feat, dim=-1)[0]
        return feat


class PointGraphSegNet(nn.Module):
    def __init__(self, num_classes=11, k=16, dropout=0.30):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv1d(11, 96, 1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )

        self.low_level_embed = nn.Sequential(
            nn.Conv1d(11, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 96, 1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )

        self.edge1 = EdgeConvBlock(96, 128, k=k)
        self.edge2 = EdgeConvBlock(128, 224, k=k)
        self.edge3 = EdgeConvBlock(224, 384, k=k)

        self.fuse = nn.Sequential(
            nn.Conv1d(11 + 96 + 128 + 224 + 384, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.fuse_attn = ChannelAttention1D(512, reduction=8)

        self.global_proj = nn.Sequential(
            nn.Conv1d(512, 768, 1, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv1d(512 + 768, 384, 1, bias=False),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(384, 192, 1, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(192, num_classes, 1),
        )

    def forward(self, feats):
        xyz = feats[:, :, :3]
        x = feats.transpose(1, 2).contiguous()

        x_low = self.low_level_embed(x)
        x0 = self.input_proj(x)

        x1 = self.edge1(x0, xyz)
        x2 = self.edge2(x1, xyz)
        x3 = self.edge3(x2, xyz)

        fused = torch.cat([x, x_low, x1, x2, x3], dim=1)
        fused = self.fuse(fused)
        fused = self.fuse_attn(fused)

        global_feat = self.global_proj(fused)
        global_feat = torch.max(global_feat, dim=2, keepdim=True)[0].repeat(1, 1, fused.shape[2])

        logits = self.head(torch.cat([fused, global_feat], dim=1))
        return logits


# ============================================================
# Losses
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        logits = logits.float()
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets_unsq = targets.unsqueeze(1)

        pt = torch.gather(probs, 1, targets_unsq).squeeze(1)
        log_pt = torch.gather(log_probs, 1, targets_unsq).squeeze(1)

        pt = torch.clamp(pt, min=self.eps, max=1.0)
        log_pt = torch.clamp(log_pt, min=math.log(self.eps), max=0.0)

        loss = -self.alpha * ((1.0 - pt) ** self.gamma) * log_pt

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise RuntimeError("NaN/Inf detected inside FocalLoss")

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedSegLoss(nn.Module):
    def __init__(self, class_weights, ce_weight=0.65, focal_weight=0.35, gamma=2.0, alpha=0.25):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.focal = FocalLoss(gamma=gamma, alpha=alpha, reduction="mean")
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        logits = logits.float()

        ce_loss = self.ce(logits, targets)

        if self.focal_weight > 0:
            focal_loss = self.focal(logits, targets)
        else:
            focal_loss = torch.tensor(0.0, device=logits.device)

        total = self.ce_weight * ce_loss + self.focal_weight * focal_loss

        if torch.isnan(total) or torch.isinf(total):
            print("ce_loss:", float(ce_loss.detach().cpu()))
            print("focal_loss:", float(focal_loss.detach().cpu()))
            raise RuntimeError("NaN/Inf detected in CombinedSegLoss")

        return total


# ============================================================
# Helpers
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_block_sampling_weights(blocks, num_classes=11, power=1.5, rare_classes=None):
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, _, y in blocks:
        counts += np.bincount(y, minlength=num_classes)

    freq = counts / max(counts.sum(), 1.0)
    inv = 1.0 / np.maximum(freq, 1e-8)
    inv = inv / inv.mean()

    rare_classes = [] if rare_classes is None else list(rare_classes)
    rare_set = set(rare_classes)

    block_scores = []
    for _, _, y in blocks:
        hist = np.bincount(y, minlength=num_classes).astype(np.float64)
        hist = hist / max(hist.sum(), 1.0)

        score = float((hist * inv).sum())

        present = set(np.where(hist > 0)[0].tolist())
        rare_hits = len(present.intersection(rare_set))
        if rare_hits > 0:
            score *= (1.0 + 0.75 * rare_hits)

        diversity = np.count_nonzero(hist)
        score *= (1.0 + 0.05 * diversity)

        block_scores.append(score)

    block_scores = np.asarray(block_scores, dtype=np.float64)
    block_scores = np.power(block_scores, power)
    block_scores = block_scores / max(block_scores.sum(), 1e-12)

    return torch.as_tensor(block_scores, dtype=torch.double)


# ============================================================
# Train / eval
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, num_classes):
    model.train()

    total_loss = 0.0
    total_points = 0
    all_true = []
    all_pred = []

    for step, (feats, labels, _) in enumerate(loader):
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.startswith("cuda"))):
            logits = model(feats)
            loss = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"NaN/Inf loss at training step {step}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.numel()
        total_points += labels.numel()

        preds = torch.argmax(logits, dim=1)
        all_true.append(labels.detach().cpu().numpy().reshape(-1))
        all_pred.append(preds.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    avg_loss = total_loss / max(total_points, 1)
    oa = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    return avg_loss, oa, mf1, cm


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()

    total_loss = 0.0
    total_points = 0
    all_true = []
    all_pred = []

    for step, (feats, labels, _) in enumerate(loader):
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.startswith("cuda"))):
            logits = model(feats)
            loss = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"NaN/Inf loss at validation/test step {step}")

        total_loss += loss.item() * labels.numel()
        total_points += labels.numel()

        preds = torch.argmax(logits, dim=1)
        all_true.append(labels.detach().cpu().numpy().reshape(-1))
        all_pred.append(preds.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    avg_loss = total_loss / max(total_points, 1)
    oa = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    return avg_loss, oa, mf1, cm, y_true, y_pred


# ============================================================
# Plot utils
# ============================================================
def plot_training_curves(history, out_dir):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_oa"], label="Train OA")
    plt.plot(epochs, history["val_oa"], label="Val OA")
    plt.plot(epochs, history["train_mf1"], label="Train mF1")
    plt.plot(epochs, history["val_mf1"], label="Val mF1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Accuracy and Mean F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metric_curves.png"), dpi=200)
    plt.close()


def plot_confusion_matrix(cm, out_path, class_names):
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=np.float64), where=cm_sum != 0)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# Export helpers
# ============================================================
def save_colored_txt(path, xyz, labels):
    colors = label_to_color(labels)
    colors_255 = np.clip(colors * 255.0, 0, 255).astype(np.int32)
    out = np.concatenate([xyz, colors_255, labels.reshape(-1, 1)], axis=1)
    np.savetxt(path, out, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d", "%d"])


def make_features_for_inference(xyz, rgb, stats):
    xyz_norm = (xyz - stats["xyz_mean"]) / stats["xyz_std"]
    rgb_norm = (rgb - stats["rgb_mean"]) / stats["rgb_std"]

    xyz_rel = xyz - xyz.mean(axis=0, keepdims=True)

    z = xyz[:, 2:3]
    z_min = z.min(axis=0, keepdims=True)
    z_max = z.max(axis=0, keepdims=True)
    z_norm_local = (z - z_min) / (z_max - z_min + 1e-6)
    z_rel = z - z.mean(axis=0, keepdims=True)

    feats = np.concatenate(
        [xyz_norm, rgb_norm, xyz_rel, z_norm_local, z_rel],
        axis=1
    ).astype(np.float32)

    return feats


def save_pred_gt_visualizations(model, blocks, stats, device, out_dir, num_save_blocks=8):
    model.eval()
    save_dir = os.path.join(out_dir, "test_visualizations")
    os.makedirs(save_dir, exist_ok=True)

    chosen_indices = np.linspace(0, len(blocks) - 1, min(num_save_blocks, len(blocks)), dtype=int)

    for bi, idx in enumerate(chosen_indices):
        xyz, rgb, label = blocks[idx]
        feats = make_features_for_inference(xyz, rgb, stats)

        feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
        logits = model(feats_t)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        save_colored_txt(os.path.join(save_dir, f"block_{bi:02d}_gt.txt"), xyz, label)
        save_colored_txt(os.path.join(save_dir, f"block_{bi:02d}_pred.txt"), xyz, pred)


@torch.no_grad()
def infer_all_test_blocks(model, blocks, stats, device, out_dir):
    model.eval()
    pred_dir = os.path.join(out_dir, "test_full_exports")
    os.makedirs(pred_dir, exist_ok=True)

    global_true = []
    global_pred = []

    for i, (xyz, rgb, label) in enumerate(blocks):
        feats = make_features_for_inference(xyz, rgb, stats)

        feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
        logits = model(feats_t)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        save_colored_txt(os.path.join(pred_dir, f"test_block_{i:05d}_gt.txt"), xyz, label)
        save_colored_txt(os.path.join(pred_dir, f"test_block_{i:05d}_pred.txt"), xyz, pred)

        global_true.append(label.reshape(-1))
        global_pred.append(pred.reshape(-1))

    y_true = np.concatenate(global_true)
    y_pred = np.concatenate(global_pred)

    oa = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))

    return oa, mf1, cm, y_true, y_pred


# ============================================================
# Main
# ============================================================

print("=" * 60)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU   :", torch.cuda.get_device_name(0))
print("=" * 60)

t0 = time.time()

print("Reading training file...")
train_xyz, train_rgb, train_labels = read_txt_point_cloud(TRAIN_FILE)

print("Reading test file...")
test_xyz, test_rgb, test_labels = read_txt_point_cloud(TEST_FILE)

print(f"Train raw points: {train_xyz.shape[0]:,}")
print(f"Test raw points : {test_xyz.shape[0]:,}")
print("Train unique labels:", np.unique(train_labels))
print("Test unique labels :", np.unique(test_labels))

if train_labels.min() < 0 or train_labels.max() >= NUM_CLASSES:
    raise ValueError(f"Train labels must be in [0, {NUM_CLASSES - 1}]")
if test_labels.min() < 0 or test_labels.max() >= NUM_CLASSES:
    raise ValueError(f"Test labels must be in [0, {NUM_CLASSES - 1}]")

print("Building train blocks...")
train_blocks_all = build_blocks(
    train_xyz,
    train_rgb,
    train_labels,
    block_size=BLOCK_SIZE,
    stride=STRIDE,
    num_points=NUM_POINTS,
    min_points_in_block=MIN_POINTS_IN_BLOCK,
    max_blocks=MAX_TRAIN_BLOCKS,
    min_unique_classes=MIN_UNIQUE_CLASSES_IN_BLOCK,
)

print("Building test blocks...")
test_blocks = build_blocks(
    test_xyz,
    test_rgb,
    test_labels,
    block_size=BLOCK_SIZE,
    stride=STRIDE,
    num_points=NUM_POINTS,
    min_points_in_block=MIN_POINTS_IN_BLOCK,
    max_blocks=MAX_TEST_BLOCKS,
    min_unique_classes=MIN_UNIQUE_CLASSES_IN_BLOCK,
)

if len(train_blocks_all) == 0 or len(test_blocks) == 0:
    raise RuntimeError("No valid blocks were generated. Adjust block parameters.")

train_idx, val_idx = train_test_split(
    np.arange(len(train_blocks_all)),
    test_size=VAL_RATIO,
    random_state=SEED,
    shuffle=True,
)

train_blocks = [train_blocks_all[i] for i in train_idx]
val_blocks = [train_blocks_all[i] for i in val_idx]

print(f"Train blocks: {len(train_blocks)}")
print(f"Val blocks  : {len(val_blocks)}")
print(f"Test blocks : {len(test_blocks)}")

stats = compute_global_stats(train_blocks)
class_weights = compute_class_weights(train_blocks, num_classes=NUM_CLASSES).to(DEVICE)

print("Class weights:", class_weights)
print("Any NaN in class weights?", torch.isnan(class_weights).any().item())
print("Any Inf in class weights?", torch.isinf(class_weights).any().item())

train_dataset = PointBlockDataset(train_blocks, stats, augment=True)
val_dataset = PointBlockDataset(val_blocks, stats, augment=False)
test_dataset = PointBlockDataset(test_blocks, stats, augment=False)

sampling_weights = build_block_sampling_weights(
    train_blocks,
    num_classes=NUM_CLASSES,
    power=RARE_BLOCK_POWER,
    rare_classes=RARE_CLASSES,
)

train_sampler = WeightedRandomSampler(
    weights=sampling_weights,
    num_samples=len(train_blocks),
    replacement=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(NUM_WORKERS > 0),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(NUM_WORKERS > 0),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(NUM_WORKERS > 0),
)

model = PointGraphSegNet(num_classes=NUM_CLASSES, k=K_NEIGHBORS, dropout=DROPOUT).to(DEVICE)
num_params = count_parameters(model)
print(f"Trainable parameters: {num_params:,}")

criterion = CombinedSegLoss(
    class_weights=class_weights,
    ce_weight=CE_WEIGHT,
    focal_weight=FOCAL_WEIGHT,
    gamma=FOCAL_GAMMA,
    alpha=FOCAL_ALPHA,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.startswith("cuda")))

history = defaultdict(list)
best_val_mf1 = -1.0
best_path = os.path.join(OUT_DIR, "best_model.pth")

print("Starting training...")
train_start = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    train_loss, train_oa, train_mf1, _ = train_one_epoch(
        model, train_loader, optimizer, criterion, scaler, DEVICE, NUM_CLASSES
    )

    val_loss, val_oa, val_mf1, _, _, _ = evaluate(
        model, val_loader, criterion, DEVICE, NUM_CLASSES
    )

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_oa"].append(train_oa)
    history["val_oa"].append(val_oa)
    history["train_mf1"].append(train_mf1)
    history["val_mf1"].append(val_mf1)

    elapsed = time.time() - epoch_start
    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train OA: {train_oa:.4f} | Train mF1: {train_mf1:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val OA: {val_oa:.4f} | Val mF1: {val_mf1:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    if val_mf1 > best_val_mf1:
        best_val_mf1 = val_mf1
        torch.save(model.state_dict(), best_path)
        print(f"  Saved best model to {best_path}")

    if epoch % SAVE_EVERY == 0:
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"checkpoint_epoch_{epoch:03d}.pth"))

total_train_time = time.time() - train_start
print(f"Training finished in {total_train_time / 60:.2f} minutes")

plot_training_curves(history, OUT_DIR)

print("Loading best model for final evaluation...")
model.load_state_dict(torch.load(best_path, map_location=DEVICE))

val_loss, val_oa, val_mf1, val_cm, _, _ = evaluate(model, val_loader, criterion, DEVICE, NUM_CLASSES)
print(f"Best model validation -> Loss: {val_loss:.4f}, OA: {val_oa:.4f}, mF1: {val_mf1:.4f}")

test_loss, test_oa, test_mf1, test_cm, y_true, y_pred = evaluate(
    model, test_loader, criterion, DEVICE, NUM_CLASSES
)
print(f"Test -> Loss: {test_loss:.4f}, OA: {test_oa:.4f}, mF1: {test_mf1:.4f}")

plot_confusion_matrix(test_cm, os.path.join(OUT_DIR, "test_confusion_matrix.png"), CLASS_NAMES)

per_class_f1 = f1_score(y_true, y_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
per_class_acc = np.divide(
    np.diag(test_cm),
    test_cm.sum(axis=1),
    out=np.zeros(NUM_CLASSES, dtype=np.float64),
    where=test_cm.sum(axis=1) != 0,
)

print("Per-class performance:")
for i in range(NUM_CLASSES):
    print(f"Class {i:02d} ({CLASS_NAMES[i]:>16s}) -> Acc: {per_class_acc[i]:.4f}, F1: {per_class_f1[i]:.4f}")

np.savetxt(
    os.path.join(OUT_DIR, "per_class_metrics.txt"),
    np.column_stack([np.arange(NUM_CLASSES), per_class_acc, per_class_f1]),
    fmt=["%d", "%.6f", "%.6f"],
    header="class_id class_accuracy class_f1",
)

print("Saving GT and prediction colored TXT files for sample test blocks...")
save_pred_gt_visualizations(model, test_blocks, stats, DEVICE, OUT_DIR, num_save_blocks=8)

print("Saving all test block predictions...")
test_oa_full, test_mf1_full, test_cm_full, _, _ = infer_all_test_blocks(model, test_blocks, stats, DEVICE, OUT_DIR)
print(f"Full exported test blocks -> OA: {test_oa_full:.4f}, mF1: {test_mf1_full:.4f}")

plot_confusion_matrix(
    test_cm_full,
    os.path.join(OUT_DIR, "test_confusion_matrix_full_exports.png"),
    CLASS_NAMES,
)

total_time = time.time() - t0
print(f"All done in {total_time / 60:.2f} minutes")
print(f"Outputs saved to: {OUT_DIR}")

