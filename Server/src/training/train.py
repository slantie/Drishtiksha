# src/training/train.py

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import json
import numpy as np

# Use relative imports for modules within the training package
from .plotting import plot_confusion_matrix, plot_training_history
from .data_loader_lstm import get_lstm_data_loaders
from .model_lstm import create_lstm_model
from .utils import load_config


def run_training_loop(model, train_loader, val_loader, optimizer, device, config):
    """The main training and validation loop with checkpointing, plotting, and early stopping."""
    train_config = config["training"]
    model_name = train_config["target_model"]
    model_config = config["models"][model_name]

    model.to(device)
    # Your model has 1 output logit, so BCEWithLogitsLoss is correct.
    # It's more numerically stable than using Sigmoid + BCELoss.
    criterion = nn.BCEWithLogitsLoss()

    output_dir = train_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(train_config["checkpoint_path"]), exist_ok=True)

    checkpoint_path = train_config["checkpoint_path"]
    history_path = os.path.join(output_dir, "training_history.json")

    patience = train_config.get("patience", 5)
    patience_counter = 0
    best_val_accuracy = 0.0

    if os.path.exists(checkpoint_path):
        print(f"=> Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        history = (
            json.load(open(history_path, "r"))
            if os.path.exists(history_path)
            else {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        )
        best_val_accuracy = max(history.get("val_acc", [0]))
        print(f"=> Loaded checkpoint. Resuming training from epoch {start_epoch + 1}")
    else:
        print("=> No checkpoint found, starting from scratch.")
        start_epoch = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(start_epoch, train_config["num_epochs"]):
        current_epoch = epoch + 1
        print(f"\n--- Epoch {current_epoch}/{train_config['num_epochs']} ---")

        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        train_progress = tqdm(train_loader, desc=f"Epoch {current_epoch} [Training]")
        for i, (pixel_values, labels) in enumerate(train_progress):
            if pixel_values is None:
                continue

            pixel_values, labels = pixel_values.to(device), labels.to(device)

            # The model expects the number of frames to be passed
            outputs = model(
                pixel_values, num_frames_per_video=model_config["num_frames"]
            ).squeeze()
            loss = criterion(outputs, labels)

            # Gradient Accumulation
            loss = loss / train_config["accumulation_steps"]
            loss.backward()

            if (i + 1) % train_config["accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += (
                loss.item() * train_config["accumulation_steps"]
            )  # Unscale for logging
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            train_total += labels.size(0)
            train_correct += (predicted == labels.long()).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        val_progress = tqdm(val_loader, desc=f"Epoch {current_epoch} [Validation]")
        with torch.no_grad():
            for pixel_values, labels in val_progress:
                if pixel_values is None:
                    continue
                pixel_values, labels = pixel_values.to(device), labels.to(device)
                outputs = model(
                    pixel_values, num_frames_per_video=model_config["num_frames"]
                ).squeeze()
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).long()
                val_total += labels.size(0)
                val_correct += (predicted == labels.long()).sum().item()
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        print(
            f"Epoch {current_epoch} Summary | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_accuracy)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        plot_confusion_matrix(all_val_labels, all_val_preds, current_epoch, output_dir)
        plot_training_history(history, output_dir)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            print(
                f"Validation accuracy improved to {val_accuracy:.2f}%. Saving best model."
            )
            torch.save(model.state_dict(), model_config["model_path"])
        else:
            patience_counter += 1
            print(
                f"Validation accuracy did not improve. Patience: {patience_counter}/{patience}"
            )

        torch.save(
            {
                "epoch": current_epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_path,
        )

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs with no improvement."
            )
            break

    print("Training finished.")


if __name__ == "__main__":
    # This allows the script to be run directly.
    print("Starting training process...")
    config = load_config("configs/config.yaml")

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model from config
    model_config = config["models"][config["training"]["target_model"]]
    model = create_lstm_model(model_config["model_definition"])

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"]
    )

    # Get data loaders
    print("Initializing data loaders...")
    train_loader, val_loader = get_lstm_data_loaders(config)

    # Start training
    run_training_loop(model, train_loader, val_loader, optimizer, device, config)
