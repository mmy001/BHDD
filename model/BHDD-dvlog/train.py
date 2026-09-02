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
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import BHDDDvlogDataset
from model import Net


def build_adjacency_matrix() -> torch.Tensor:
    """Build the normalized facial landmark adjacency matrix."""
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    epoch_size: int,
    warmup_epoch: int,
    test_every: int,
    save_path: Path,
):
    """Train BHDD on D-Vlog and return the best checkpoint information."""
    best_val_acc = 0.0
    best_model_path = None
    alpha = 0.8
    best_labels = []
    best_preds = []
    best_samples = []
    best_epoch = 0
    best_val_loss = float("inf")
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0

    logging.info("Training started.")
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epoch_size + 1):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()

        for batch_idx, (video, audio, label, sample_names) in loop:
            video = video.to(device)
            audio = audio.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output, visual, visual_reconstructed = model(video, audio)

            check_for_nan_inf(visual, "visual")
            check_for_nan_inf(visual_reconstructed, "visual_reconstructed")

            ce_loss = criterion(output, label.long())
            rec_loss = reconstruction_loss(visual, visual_reconstructed)
            train_loss = ce_loss if epoch < warmup_epoch else ce_loss + alpha * rec_loss

            train_loss.backward()
            max_norm = 0.5 if epoch < warmup_epoch else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
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
        logging.info("Epoch [%s/%s] Train Loss: %.4f Acc: %.2f%%", epoch, epoch_size, epoch_loss, epoch_acc)

        if epoch >= warmup_epoch and epoch % test_every == 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_preds = []
            all_samples = []

            with torch.no_grad():
                loop = tqdm(enumerate(eval_loader), total=len(eval_loader))
                for batch_idx, (video, audio, label, sample_names) in loop:
                    video = video.to(device)
                    audio = audio.to(device)
                    label = label.to(device)

                    dev_output, visual, visual_reconstructed = model(video, audio)
                    ce_loss = criterion(dev_output, label.long())
                    rec_loss = reconstruction_loss(visual, visual_reconstructed)
                    loss = ce_loss + alpha * rec_loss
                    val_loss += loss.item()

                    _, predicted = torch.max(dev_output.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

                    all_labels.extend(label.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                    all_samples.extend(sample_names)
                    loop.set_postfix(loss=val_loss / (batch_idx + 1), acc=100.0 * correct / total)

            val_loss = val_loss / len(eval_loader)
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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_val_loss = val_loss
                best_precision = precision
                best_recall = recall
                best_f1 = f1score
                best_labels = all_labels.copy()
                best_preds = all_preds.copy()
                best_samples = all_samples.copy()

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
                best_model_path = save_path / (
                    f"best_epoch_{epoch}_acc_{val_acc:.2f}_"
                    f"p_{precision:.4f}_r_{recall:.4f}_f1_{f1score:.4f}.pth"
                )
                torch.save(checkpoint, best_model_path)
                logging.info("Validation accuracy improved to %.2f%%. Model saved.", val_acc)

    logging.info(
        "Training completed. Best Val Acc: %.2f%% (Epoch: %s), Precision: %.4f, Recall: %.4f, F1: %.4f, Best Validation Loss: %.4f",
        best_val_acc,
        best_epoch,
        best_precision,
        best_recall,
        best_f1,
        best_val_loss,
    )
    return best_model_path, best_labels, best_preds, best_samples


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the BHDD model on the D-Vlog dataset.")
    parser.add_argument("--video-root", type=Path, required=True, help="Root directory containing D-Vlog video features.")
    parser.add_argument("--audio-root", type=Path, required=True, help="Root directory containing D-Vlog audio features.")
    parser.add_argument("--label-root", type=Path, required=True, help="Root directory containing D-Vlog label CSV files.")
    parser.add_argument("--train-split", type=str, default="train", help="Training split name under the dataset root.")
    parser.add_argument("--eval-split", type=str, default="valid", help="Evaluation split name under the dataset root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for logs and checkpoints.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name. Defaults to a timestamp.")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--test-every", type=int, default=1)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--topk", type=int, default=100, help="Number of selected tokens.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2222)
    return parser.parse_args()


def main() -> None:
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

    train_video_dir = args.video_root / args.train_split
    eval_video_dir = args.video_root / args.eval_split
    train_audio_dir = args.audio_root / args.train_split
    eval_audio_dir = args.audio_root / args.eval_split
    train_label_dir = args.label_root / args.train_split
    eval_label_dir = args.label_root / args.eval_split

    adj_matrix = build_adjacency_matrix().to(device)
    train_file_list = sort_feature_filenames(train_video_dir)
    eval_file_list = sort_feature_filenames(eval_video_dir)

    config_to_save = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)

    train_dataset = BHDDDvlogDataset(
        video_path=train_video_dir,
        audio_path=train_audio_dir,
        label_path=train_label_dir,
        file_list=train_file_list,
        mode="train",
    )
    eval_dataset = BHDDDvlogDataset(
        video_path=eval_video_dir,
        audio_path=eval_audio_dir,
        label_path=eval_label_dir,
        file_list=eval_file_list,
        mode=args.eval_split,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logging.info("Training set size: %s batches", len(train_loader))
    logging.info("Evaluation set size: %s batches", len(eval_loader))

    model = Net(adj_matrix=adj_matrix, k=args.topk).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.85, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=True,
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

    best_model_path, best_labels, best_preds, best_samples = train_model(
        model,
        train_loader,
        eval_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        args.num_epochs,
        args.warmup_epochs,
        args.test_every,
        ckpt_dir,
    )

    if best_labels and best_preds:
        plot_confusion_matrix(
            best_labels,
            best_preds,
            labels_name=[0, 1],
            savename=run_dir / "confusion_matrix.png",
            title=f"Confusion Matrix ({args.eval_split})",
            axis_labels=["Non-depressed", "Depressed"],
        )

    prediction_log_path = run_dir / "sample_predictions.csv"
    with open(prediction_log_path, "w", encoding="utf-8") as f:
        f.write("sample,prediction,actual\n")
        for sample, pred, label in zip(best_samples, best_preds, best_labels):
            f.write(f"{sample},{pred},{label}\n")
    logging.info("Saved individual sample predictions to %s", prediction_log_path)

    metrics_summary = {
        "best_checkpoint": str(best_model_path) if best_model_path is not None else None,
        "num_samples": len(best_labels),
        "accuracy": float(np.mean(np.array(best_preds) == np.array(best_labels))) if best_labels else None,
        "precision_weighted": float(precision_score(best_labels, best_preds, average="weighted")) if best_labels else None,
        "recall_weighted": float(recall_score(best_labels, best_preds, average="weighted")) if best_labels else None,
        "f1_weighted": float(f1_score(best_labels, best_preds, average="weighted")) if best_labels else None,
    }
    with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    if best_model_path is not None:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["net"])
        model.to(device)


if __name__ == "__main__":
    main()
