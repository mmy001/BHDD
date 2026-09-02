from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from math import cos
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import BHDDDataset
from model import Net


def build_adjacency_matrix() -> torch.Tensor:
    """Build the normalized facial landmark adjacency matrix.

    The graph contains 68 original facial landmarks and 9 additional global
    region nodes, resulting in a 77 x 77 adjacency matrix.
    """
    num_original_nodes = 68
    num_regions = 9
    total_nodes = num_original_nodes + num_regions
    adj = np.zeros((total_nodes, total_nodes), dtype=np.float32)

    regions = [
        (0, 16),
        (17, 21),
        (22, 26),
        (27, 30),
        (31, 35),
        (36, 41),
        (42, 47),
        (48, 59),
        (60, 67),
    ]

    for i in range(0, 16):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    for i in range(17, 21):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    for i in range(22, 26):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    for i in range(27, 30):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    for i in range(31, 35):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1

    for i in range(36, 41):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    adj[41, 36] = 1

    for i in range(42, 47):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    adj[47, 42] = 1

    for i in range(48, 59):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    adj[59, 48] = 1

    for i in range(60, 67):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    adj[67, 60] = 1

    for region_idx, (start, end) in enumerate(regions):
        global_node = num_original_nodes + region_idx
        for node in range(start, end + 1):
            adj[node, global_node] = 1
            adj[global_node, node] = 1

    for i in range(num_original_nodes, total_nodes):
        for j in range(num_original_nodes, total_nodes):
            if i != j:
                adj[i, j] = 1
                adj[j, i] = 1

    adj += np.eye(total_nodes)

    degree = np.sum(adj, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    d_inv_sqrt = np.diag(degree_inv_sqrt)
    adj_normalized = d_inv_sqrt @ adj @ d_inv_sqrt
    return torch.tensor(adj_normalized, dtype=torch.float32)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels_name,
    savename,
    title: str | None = None,
    thresh: float = 0.6,
    axis_labels=None,
) -> None:
    """Plot and save a normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels_name)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 6))
    plt.imshow(cm_normalized, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.colorbar()

    if title is not None:
        plt.title(title)

    num_local = np.arange(len(labels_name))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels)
    plt.yticks(num_local, labels_name, rotation=90, va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            if cm[i][j] > 0:
                plt.text(
                    j,
                    i,
                    f"{cm_normalized[i][j] * 100:.2f}%",
                    ha="center",
                    va="center",
                    color="white" if cm_normalized[i][j] > thresh else "black",
                )

    plt.tight_layout()
    plt.savefig(savename, format="png")
    plt.close()


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
):
    """Linear warmup followed by linear decay."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
):
    """Linear warmup followed by cosine decay."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.5 * (
            cos(
                min(
                    (current_step - num_warmup_steps)
                    / (num_training_steps - num_warmup_steps),
                    1.0,
                )
                * math.pi
            )
            + 1
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def reconstruction_loss(visual: torch.Tensor, visual_reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute the reconstruction loss used by the original training code."""
    mse_loss = nn.MSELoss(reduction="mean")
    loss = mse_loss(visual, visual_reconstructed)
    loss = torch.clamp(loss, min=1e-10, max=1e10) / 1e10
    return loss


def check_for_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Print warnings if a tensor contains NaN or Inf values."""
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN values.")
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains Inf values.")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_file: Path) -> None:
    """Configure logging to both file and stdout."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%m-%d %H:%M",
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def sort_feature_filenames(feature_dir: Path) -> list[str]:
    """Sort feature file names numerically when possible."""
    files = [p.name for p in feature_dir.iterdir() if p.suffix == ".npy"]

    def sort_key(name: str):
        stem = Path(name).stem
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    return sorted(files, key=sort_key)


def collect_labels(file_list: list[str], label_dir: Path) -> np.ndarray:
    """Collect labels for all samples using the same parsing logic as the dataset."""
    labels = []
    for file in file_list:
        sample_name = Path(file).stem
        label = BHDDDataset._parse_label_value(label_dir / f"{sample_name}_Depression.csv")
        labels.append(label)
    return np.array(labels)


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    epoch_size: int,
    warmup_epoch: int,
    test_every: int,
    save_path: Path,
    fold_num: int,
    patience: int = 10,
):
    """Train one fold and return predictions from the best checkpoint."""
    best_acc = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_preds = []
    best_labels = []
    alpha = 0.7
    epochs_no_improve = 0
    early_stop = False

    logging.info("Fold %s training started.", fold_num)
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epoch_size + 1):
        if early_stop:
            logging.info("Early stopping triggered for fold %s.", fold_num)
            break

        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (lm, au_pose_gaze, audio, label) in loop:
            lm = lm.to(device)
            au_pose_gaze = au_pose_gaze.to(device)
            audio = audio.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output, _, _, visual, visual_reconstructed = model(lm, au_pose_gaze, audio)

            check_for_nan_inf(visual, "visual")
            check_for_nan_inf(visual_reconstructed, "visual_reconstructed")

            ce_loss = criterion(output, label.long())
            rec_loss = reconstruction_loss(visual, visual_reconstructed)
            train_loss = ce_loss if epoch < warmup_epoch else ce_loss + alpha * rec_loss

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += train_loss.item()
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            loop.set_description(f"Train Epoch [{epoch}/{epoch_size}]")
            loop.set_postfix(loss=running_loss / (batch_idx + 1), acc=100.0 * correct / total)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        logging.info(
            "Epoch [%s/%s] Train Loss: %.4f Acc: %.2f%%",
            epoch,
            epoch_size,
            epoch_loss,
            epoch_acc,
        )

        if epoch >= warmup_epoch and epoch % test_every == 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_preds = []

            with torch.no_grad():
                loop = tqdm(enumerate(dev_loader), total=len(dev_loader))
                for batch_idx, (lm, au_pose_gaze, audio, label) in loop:
                    lm = lm.to(device)
                    au_pose_gaze = au_pose_gaze.to(device)
                    audio = audio.to(device)
                    label = label.to(device)

                    dev_output, _, _, visual, visual_reconstructed = model(lm, au_pose_gaze, audio)
                    ce_loss = criterion(dev_output, label.long())
                    rec_loss = reconstruction_loss(visual, visual_reconstructed)
                    loss = ce_loss + alpha * rec_loss
                    val_loss += loss.item()

                    _, predicted = torch.max(dev_output.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

                    all_labels.extend(label.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                    loop.set_postfix(loss=val_loss / (batch_idx + 1), acc=100.0 * correct / total)

            val_loss = val_loss / len(dev_loader)
            val_acc = 100.0 * correct / total
            precision = precision_score(all_labels, all_preds, average="weighted")
            recall = recall_score(all_labels, all_preds, average="weighted")
            f1score = f1_score(all_labels, all_preds, average="weighted")

            logging.info(
                "Epoch [%s/%s] Validation Loss: %.4f Acc: %.2f%% Precision: %.4f Recall: %.4f F1: %.4f",
                epoch,
                epoch_size,
                val_loss,
                val_acc,
                precision,
                recall,
                f1score,
            )

            if val_acc > best_acc:
                best_acc = val_acc
                best_precision = precision
                best_recall = recall
                best_f1 = f1score
                best_preds = all_preds
                best_labels = all_labels
                epochs_no_improve = 0

                checkpoint = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "scheduler": scheduler.state_dict(),
                    "metrics": {
                        "accuracy": val_acc,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1score,
                    },
                }
                checkpoint_name = (
                    f"fold_{fold_num}_epoch_{epoch}_acc_{val_acc:.2f}_"
                    f"p_{precision:.4f}_r_{recall:.4f}_f1_{f1score:.4f}.pth"
                )
                torch.save(checkpoint, save_path / checkpoint_name)
                logging.info("Validation accuracy improved to %.2f%%. Model saved.", val_acc)
            else:
                epochs_no_improve += 1
                logging.info("No improvement for %s epoch(s).", epochs_no_improve)

            if epochs_no_improve >= patience:
                logging.info("Early stopping after %s epochs without improvement.", patience)
                early_stop = True

    logging.info(
        "Fold %s training completed. Best Acc: %.2f%% Precision: %.4f Recall: %.4f F1: %.4f",
        fold_num,
        best_acc,
        best_precision,
        best_recall,
        best_f1,
    )
    return best_labels, best_preds


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the BHDD model on LMVD.")
    parser.add_argument("--lm-dir", type=Path, required=True, help="Directory containing LM .npy files.")
    parser.add_argument(
        "--au-pose-gaze-dir",
        type=Path,
        required=True,
        help="Directory containing AU+pose+gaze .npy files.",
    )
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing audio .npy files.")
    parser.add_argument("--label-dir", type=Path, required=True, help="Directory containing label CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for logs and checkpoints.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name. Defaults to a timestamp.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--test-every", type=int, default=1)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--topk", type=int, default=186, help="Number of selected tokens.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2222)
    return parser.parse_args()


def main() -> None:
    """Run k-fold training for BHDD."""
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or time.strftime("%m_%d__%H_%M", time.localtime())
    run_dir = args.output_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "training.log")

    logging.info("Using device: %s", device)
    logging.info("Run directory: %s", run_dir)

    adj_matrix = build_adjacency_matrix().to(device)
    file_list = sort_feature_filenames(args.lm_dir)
    labels = collect_labels(file_list, args.label_dir)

    config_to_save = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)

    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    results = {"acc": [], "precision": [], "recall": [], "f1": []}
    all_fold_labels = []
    all_fold_preds = []

    for fold, (train_index, test_index) in enumerate(skf.split(file_list, labels), start=1):
        x_train = np.array(file_list)[train_index]
        x_test = np.array(file_list)[test_index]

        logging.info("Fold %s training started.", fold)
        logging.info("Fold %s training set size: %s", fold, len(x_train))
        logging.info("Fold %s validation set size: %s", fold, len(x_test))

        train_dataset = BHDDDataset(
            lm_path=args.lm_dir,
            au_pose_gaze_path=args.au_pose_gaze_dir,
            audio_path=args.audio_dir,
            label_path=args.label_dir,
            file_list=x_train,
            mode="train",
        )
        dev_dataset = BHDDDataset(
            lm_path=args.lm_dir,
            au_pose_gaze_path=args.au_pose_gaze_dir,
            audio_path=args.audio_dir,
            label_path=args.label_dir,
            file_list=x_test,
            mode="dev",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        model = Net(adj_matrix=adj_matrix, K=args.topk).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
            amsgrad=False,
        )

        train_steps = len(train_loader) * args.num_epochs
        warmup_steps = len(train_loader) * args.warmup_epochs
        if args.schedule == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=train_steps,
            )
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=train_steps,
            )

        fold_ckpt_dir = ckpt_dir / f"fold_{fold}"
        top_label, top_pred = train_one_fold(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch_size=args.num_epochs,
            warmup_epoch=args.warmup_epochs,
            test_every=args.test_every,
            save_path=fold_ckpt_dir,
            fold_num=fold,
            patience=args.patience,
        )

        all_fold_labels.extend(top_label)
        all_fold_preds.extend(top_pred)

        acc = 100.0 * (np.sum(np.array(top_pred) == np.array(top_label))) / len(top_label)
        precision = precision_score(top_label, top_pred, average="weighted")
        recall = recall_score(top_label, top_pred, average="weighted")
        f1score = f1_score(top_label, top_pred, average="weighted")

        results["acc"].append(acc)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1score)

        logging.info(
            "Fold %s completed. Best Acc: %.2f%% Precision: %.4f Recall: %.4f F1: %.4f",
            fold,
            acc,
            precision,
            recall,
            f1score,
        )

    all_fold_labels = np.array(all_fold_labels)
    all_fold_preds = np.array(all_fold_preds)
    np.save(run_dir / "total_pre.npy", all_fold_preds)
    np.save(run_dir / "total_label.npy", all_fold_labels)

    plot_confusion_matrix(
        all_fold_labels,
        all_fold_preds,
        labels_name=[0, 1],
        savename=run_dir / "confusion_matrix.png",
        title="Confusion Matrix",
    )

    avg_acc = float(np.mean(results["acc"]))
    avg_precision = float(np.mean(results["precision"]))
    avg_recall = float(np.mean(results["recall"]))
    avg_f1 = float(np.mean(results["f1"]))

    summary = {
        "average_accuracy": avg_acc,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1": avg_f1,
    }
    with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("Average Accuracy: %.2f%%", avg_acc)
    logging.info("Average Precision: %.4f", avg_precision)
    logging.info("Average Recall: %.4f", avg_recall)
    logging.info("Average F1 Score: %.4f", avg_f1)

    print("Training completed.")
    print(f"Average Accuracy: {avg_acc:.2f}%")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")


if __name__ == "__main__":
    main()
