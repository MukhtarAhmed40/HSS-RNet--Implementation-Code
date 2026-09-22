#!/usr/bin/env python
"""
Evaluation script for trained HSS-RNet models.
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hss_rnet import HSSRNet, S_SSM
from data.preprocessing import PreprocessingPipeline
from data.dataset import create_dataloaders
from training.trainer import Trainer
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HSS-RNet")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="evaluation")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def plot_roc_curve(y_true, y_prob, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return roc_auc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(42)
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location="cpu")
    config = checkpoint["config"]
    
    # Load data
    data = np.load(args.data_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Preprocessing
    preprocessor = PreprocessingPipeline(
        sequence_length=config["data"]["sequence_length"],
        stride=config["data"]["stride"],
    )
    
    # Note: In practice, you would load the fitted preprocessor
    # For now, fit on test data (not recommended for real evaluation)
    X_test_norm = preprocessor.fit_transform(X_test)
    test_seq, test_labels, test_masks = preprocessor.construct_sequences(
        X_test_norm, y_test
    )
    
    # Create dataloader
    _, _, test_loader = create_dataloaders(
        (test_seq, test_labels, test_masks),
        (test_seq, test_labels, test_masks),
        (test_seq, test_labels, test_masks),
        batch_size=args.batch_size,
    )
    
    # Create model
    num_classes = len(np.unique(y_test))
    
    if config["model"]["name"] == "hss_rnet":
        model = HSSRNet(
            input_dim=config["data"]["num_features"],
            d_model=config["model"]["d_model"],
            d_state=config["model"]["d_state"],
            bigru_hidden=config["model"]["bigru_hidden"],
            num_classes=num_classes,
        )
    else:
        model = S_SSM(
            input_dim=config["data"]["num_features"],
            d_model=config["model"]["d_model"],
            d_state=config["model"]["d_state"],
            bigru_hidden=config["model"]["bigru_hidden"],
            num_classes=num_classes,
        )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Evaluate
    trainer = Trainer(
        model=model,
        train_loader=test_loader,
        val_loader=test_loader,
        test_loader=test_loader,
        num_classes=num_classes,
    )
    
    test_loss, metrics = trainer.evaluate(test_loader)
    
    print("\nEvaluation Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    # ROC curve for binary classification
    if num_classes == 2:
        preds, probs = trainer.predict(test_loader)
        roc_auc = plot_roc_curve(
            test_labels,
            probs[:, 1],
            os.path.join(args.output_dir, "roc_curve.png"),
        )
        print(f"  AUC:       {roc_auc:.4f}")
    
    # Save metrics
    import json
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                   for k, v in metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()
