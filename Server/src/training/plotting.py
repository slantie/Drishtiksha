# src/training/plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(y_true, y_pred, epoch, output_dir):
    """Generates and saves a confusion matrix plot."""
    class_names = ["Real", "Fake"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    save_path = os.path.join(output_dir, f'confusion_matrix_epoch_{epoch}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def plot_training_history(history, output_dir):
    """Generates and saves plots for training/validation accuracy and loss."""
    epochs = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training history plot to {save_path}")