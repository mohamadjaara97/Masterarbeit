#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mean-Teacher 2D training for LiTS (liver & tumor) with volume-aware splits.

Core features:
- Student/Teacher (EMA) with consistency loss (MSE on logits) + ramp-up
- Supervised loss: Dice + BCE (multi-class optional)
- Strong/weak augmentation (student strong, teacher weak)
- Volume-aware train/val split reading (produced by create_lits_splits.py)
- Reproducible seeding
- Periodic validation (Dice, IoU, HD95 optional) and checkpointing
- Mixed precision optional

Notes:
- Input expects preprocessed 2D slices as .npy (H,W) or (C,H,W) for image,
  and mask as integer labels (H,W): 0=background, 1=liver, 2=tumor (if 2 classes)
- Adjust Dataset class if your storage differs (e.g., .nii.gz)

References (overview):
- Mean Teacher & consistency regularization are standard in SSL for med-seg. See survey discussions. 
"""

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tensorboardX import SummaryWriter

# Import existing dataloader structure
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default='../Masterarbeit/SSL4MIS/code/data/LiTS', help='Path to data directory')
    parser.add_argument("--exp", type=str, default='LiTS/Mean_Teacher', help='experiment_name')
    parser.add_argument("--model", type=str, default='unet', help='model_name')
    parser.add_argument("--max_iterations", type=int, default=5000, help='maximum epoch number to train')
    parser.add_argument("--batch_size", type=int, default=24, help='batch_size per gpu')
    parser.add_argument("--deterministic", type=int, default=1, help='whether use deterministic training')
    parser.add_argument("--base_lr", type=float, default=0.01, help='segmentation network learning rate')
    parser.add_argument("--patch_size", type=list, default=[256, 256], help='patch size of network input')
    parser.add_argument("--seed", type=int, default=1337, help='random seed')
    parser.add_argument("--num_classes", type=int, default=4, help='output channel of network')

    # label and unlabel
    parser.add_argument("--labeled_bs", type=int, default=12, help='labeled_batch_size per gpu')
    parser.add_argument("--labeled_num", type=int, default=150, help='labeled data')
    parser.add_argument("--data_percentage", type=float, default=0.2, help='percentage of total data to use (0.2 = 20%)')
    
    # costs
    parser.add_argument("--ema_decay", type=float, default=0.99, help='ema_decay')
    parser.add_argument("--consistency_type", type=str, default="mse", help='consistency_type')
    parser.add_argument("--consistency", type=float, default=0.1, help='consistency')
    parser.add_argument("--consistency_rampup", type=float, default=40, help='consistency_rampup')
    
    # Additional arguments
    parser.add_argument("--save_dir", type=str, default="./runs/mean_teacher_lits2d")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging / snapshot
    save_root = Path(args.save_dir)
    ensure_dir(save_root)
    run_dir = save_root / f"labeled{args.labeled_num}_seed{args.seed}"
    ensure_dir(run_dir)
    (run_dir / "code").mkdir(exist_ok=True)
    # save this script
    try:
        this_file = Path(__file__).resolve()
        shutil.copy(this_file, run_dir / "code" / this_file.name)
    except Exception:
        pass
    writer = SummaryWriter(str(run_dir))

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(run_dir / "train.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Args: {args}")

    # Data loading using existing structure
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # Load full datasets
    db_train_full = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val_full = BaseDataSets(base_dir=args.root_path, split="val")
    
    # Calculate percentage of total data
    total_train_slices = len(db_train_full)
    total_val_slices = len(db_val_full)
    
    # Use only specified percentage of training data
    train_subset_size = int(total_train_slices * args.data_percentage)
    val_subset_size = int(total_val_slices * args.data_percentage)
    
    # Create subsets using torch.utils.data.Subset
    train_indices = list(range(train_subset_size))
    val_indices = list(range(val_subset_size))
    
    db_train = Subset(db_train_full, train_indices)
    db_val = Subset(db_val_full, val_indices)
    
    print(f"Using {args.data_percentage*100:.0f}% of data: {train_subset_size}/{total_train_slices} training slices, {val_subset_size}/{total_val_slices} validation slices")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total slices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # Model using existing factory
    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model().to(device)
    ema_model = create_model(ema=True).to(device)
    ema_model.load_state_dict(model.state_dict())

    # Loss
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_classes)
    args.sup_loss_fn = lambda outputs, labels: 0.5 * (dice_loss(torch.softmax(outputs, dim=1), labels.unsqueeze(1)) + ce_loss(outputs, labels))

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    scaler = GradScaler(enabled=args.amp)

    # Train
    best_dice = 0.0
    global_step = 0
    max_epoch = args.max_iterations // len(trainloader) + 1
    
    for epoch in range(1, max_epoch + 1):
        logging.info(f"Epoch {epoch}/{max_epoch}")
        global_step = train_one_epoch(args, epoch, global_step, model, ema_model,
                                      trainloader, optimizer, scaler, writer, device)
        
        if epoch % args.val_every == 0:
            # Evaluate student and teacher
            d_student = validate(args, model, valloader, device, global_step, writer, "val_student")
            d_teacher = validate(args, ema_model, valloader, device, global_step, writer, "val_teacher")
            logging.info(f"Val Dice - student: {d_student:.4f} | teacher: {d_teacher:.4f}")
            
            # Save best (teacher)
            if d_teacher > best_dice:
                best_dice = d_teacher
                torch.save({
                    "epoch": epoch,
                    "state_dict": ema_model.state_dict(),
                    "dice": best_dice,
                    "args": vars(args)
                }, run_dir / "best_teacher.pth")
                logging.info(f"Saved best checkpoint @ epoch {epoch} (Dice {best_dice:.4f})")

        # periodic snapshot
        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "state_dict": ema_model.state_dict(),
                "args": vars(args)
            }, run_dir / f"teacher_ep{epoch}.pth")

    writer.close()
    logging.info("Training complete.")


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps=1e-6, ignore_index: Optional[int] = None):
    """
    pred: logits or probabilities with shape (N, C, H, W)
    target: long tensor (N, H, W) with values in [0..C-1]
    """
    if pred.shape[1] == 1:
        # binary dice on foreground (sigmoid)
        prob = torch.sigmoid(pred)
        pred_fg = prob
        target_fg = (target > 0).float().unsqueeze(1)
        inter = (pred_fg * target_fg).sum(dim=(0, 2, 3))
        denom = pred_fg.sum(dim=(0, 2, 3)) + target_fg.sum(dim=(0, 2, 3)) + eps
        dice = (2 * inter / denom).mean()
        return dice

    # multi-class softmax dice excluding background by default
    probs = F.softmax(pred, dim=1)
    N, C, H, W = probs.shape
    dices = []
    for c in range(C):
        if ignore_index is not None and c == ignore_index:
            continue
        pred_c = probs[:, c]
        target_c = (target == c).float()
        inter = (pred_c * target_c).sum(dim=(1, 2))
        denom = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2)) + eps
        dices.append((2 * inter / denom).mean())
    if len(dices) == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(dices).mean()


class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.5, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.smooth = smooth

    def forward(self, logits, target):
        # binary only (foreground vs background)
        target_fg = (target > 0).float().unsqueeze(1)
        bce = self.bce(logits, target_fg)
        prob = torch.sigmoid(logits)
        inter = (prob * target_fg).sum(dim=(0, 2, 3))
        denom = prob.sum(dim=(0, 2, 3)) + target_fg.sum(dim=(0, 2, 3)) + self.smooth
        dice = 1.0 - (2 * inter / denom).mean()
        return self.weight_bce * bce + (1 - self.weight_bce) * dice


# ----------------------------
# Dataset & Augmentations
# ----------------------------

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "LiTS" in dataset or "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312,"150": 4500}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


# ----------------------------
# Mean Teacher helpers
# ----------------------------

def update_ema_variables(model, ema_model, alpha, global_step):
    # classic EMA update
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data * (1 - alpha))


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


# ----------------------------
# Training
# ----------------------------

def train_one_epoch(args, epoch, global_step, model, ema_model, trainloader,
                    optimizer, scaler, writer, device):
    model.train()
    ema_model.train()  # BN stats still update via teacher's own pass (or keep in eval if desired)

    loss_meter, dice_meter = 0.0, 0.0

    for i_batch, sampled_batch in enumerate(trainloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        unlabeled_volume_batch = volume_batch[args.labeled_bs:]

        # Add noise for teacher input (weak augmentation)
        noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
        ema_inputs = unlabeled_volume_batch + noise

        # Teacher forward pass (weak augmentation)
        with torch.no_grad():
            ema_output = ema_model(ema_inputs)
            ema_output_soft = torch.softmax(ema_output, dim=1)

        # Student forward pass
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            # Supervised loss on labeled data
            sup_loss = args.sup_loss_fn(outputs[:args.labeled_bs], label_batch[:args.labeled_bs].long())

            # Consistency loss on unlabeled data (mean squared error)
            if global_step < 1000:
                cons_loss = 0.0
            else:
                cons_loss = torch.mean((outputs_soft[args.labeled_bs:] - ema_output_soft) ** 2)

            # ramp-up for consistency
            cons_w = get_current_consistency_weight(global_step // 150)

            loss = sup_loss + cons_w * cons_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update EMA teacher
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # metrics
        with torch.no_grad():
            d = dice_coefficient(outputs[:args.labeled_bs], label_batch[:args.labeled_bs])
        dice_meter += d.item()
        loss_meter += loss.item()
        global_step += 1

        if (i_batch + 1) % args.log_every == 0:
            writer.add_scalar("train/loss", loss_meter / args.log_every, global_step)
            writer.add_scalar("train/sup_loss", sup_loss.item(), global_step)
            writer.add_scalar("train/cons_loss", cons_loss, global_step)
            writer.add_scalar("train/consistency_w", cons_w, global_step)
            writer.add_scalar("train/dice", dice_meter / args.log_every, global_step)
            loss_meter = 0.0
            dice_meter = 0.0

    return global_step


@torch.no_grad()
def validate(args, model, val_loader, device, global_step, writer, split_name="val"):
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(val_loader):
        metric_i = test_single_volume(
            sampled_batch["image"], sampled_batch["label"], model, classes=args.num_classes)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(val_loader.dataset)
    
    # Log metrics for each class
    for class_i in range(args.num_classes-1):
        writer.add_scalar(f'{split_name}/class_{class_i+1}_dice', metric_list[class_i, 0], global_step)
        writer.add_scalar(f'{split_name}/class_{class_i+1}_hd95', metric_list[class_i, 1], global_step)
    
    # Calculate mean dice
    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    writer.add_scalar(f'{split_name}/mean_dice', mean_dice, global_step)
    writer.add_scalar(f'{split_name}/mean_hd95', mean_hd95, global_step)
    
    return mean_dice


if __name__ == "__main__":
    main()
