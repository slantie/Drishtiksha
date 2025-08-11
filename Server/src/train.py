# /home/dell-pc-03/Deepfake/deepfake-detection/Raj/src/train.py

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import json
import numpy as np

from src.plotting import plot_confusion_matrix, plot_training_history

def train_model(model, train_loader, val_loader, optimizer, device, config):
    """The main training and validation loop with checkpointing, plotting, and early stopping."""
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_path = config['checkpoint_path']
    history_path = os.path.join(output_dir, 'training_history.json')
    
    # --- EARLY STOPPING PARAMETERS ---
    patience = config.get('patience', 5) # Get from config, or default to 5
    patience_counter = 0
    best_val_accuracy = 0.0

    # --- LOAD OR INITIALIZE STATE ---
    if os.path.exists(checkpoint_path):
        print(f"=> Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            best_val_accuracy = max(history['val_acc'])
        else:
             history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print(f"=> Loaded checkpoint. Resuming training from epoch {start_epoch + 1}")
    else:
        print("=> No checkpoint found, starting from scratch.")
        start_epoch = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # --- MAIN TRAINING LOOP ---
    for epoch in range(start_epoch, config['num_epochs']):
        current_epoch = epoch + 1
        print(f"\n--- Epoch {current_epoch}/{config['num_epochs']} ---")
        
        # Training and Validation phases (condensed for brevity, no change here)
        model.train()
        total_train_loss = 0.0; train_correct = 0; train_total = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {current_epoch} [Training]")
        for pixel_values, labels in train_progress:
            if pixel_values is None: continue
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1); train_total += labels.size(0); train_correct += (predicted == labels).sum().item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        model.eval()
        total_val_loss = 0.0; val_correct = 0; val_total = 0; all_val_preds = []; all_val_labels = []
        val_progress = tqdm(val_loader, desc=f"Epoch {current_epoch} [Validation]")
        with torch.no_grad():
            for pixel_values, labels in val_progress:
                if pixel_values is None: continue
                pixel_values, labels = pixel_values.to(device), labels.to(device)
                outputs = model(pixel_values=pixel_values).logits
                loss = criterion(outputs, labels); total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1); val_total += labels.size(0); val_correct += (predicted == labels).sum().item()
                all_val_preds.extend(predicted.cpu().numpy()); all_val_labels.extend(labels.cpu().numpy())
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        print(f"Epoch {current_epoch} Summary | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Update History & Save Plots
        history['train_loss'].append(avg_train_loss); history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss); history['val_acc'].append(val_accuracy)
        with open(history_path, 'w') as f: json.dump(history, f, indent=4)
        plot_confusion_matrix(all_val_labels, all_val_preds, current_epoch, output_dir)
        plot_training_history(history, output_dir)
        
        # --- CHECK FOR IMPROVEMENT AND EARLY STOPPING ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0  # Reset patience
            print(f"Validation accuracy improved to {val_accuracy:.2f}%. Saving best model.")
            torch.save(model.state_dict(), config['best_model_save_path'])
        else:
            patience_counter += 1
            print(f"Validation accuracy did not improve. Patience: {patience_counter}/{patience}")

        # Save checkpoint
        torch.save({'epoch': current_epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint_path)
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break # Exit the training loop

    print("Training finished.")