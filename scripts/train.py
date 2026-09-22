#!/usr/bin/env python
"""
Main training script for HSS-RNet and S-SSM.
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hss_rnet import HSSRNet, S_SSM
from data.preprocessing import PreprocessingPipeline
from data.dataset import create_dataloaders, compute_class_weights
from training.trainer import Trainer
from utils.config import load_config
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train HSS-RNet")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_type", type=str, default="hss_rnet",
                        choices=["hss_rnet", "s_ssm"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=1)
    return parser.parse_args()


def load_data(data_path: str):
    """Load preprocessed data from numpy files."""
    data = np.load(data_path, allow_pickle=True)
    return data


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Override model type
    config["model"]["name"] = args.model_type
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_run_metrics = []
    
    for run in range(args.num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{args.num_runs}")
        print(f"{'='*60}")
        
        # Set seed for reproducibility
        set_seed(args.seed + run)
        
        # Load data
        data = load_data(args.data_path)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        
        # Preprocessing
        preprocessor = PreprocessingPipeline(
            sequence_length=config["data"]["sequence_length"],
            stride=config["data"]["stride"],
        )
        
        # Fit on training data only
        X_train_norm = preprocessor.fit_transform(X_train)
        X_val_norm = preprocessor.transform(X_val)
        X_test_norm = preprocessor.transform(X_test)
        
        # Construct sequences
        train_seq, train_labels, train_masks = preprocessor.construct_sequences(
            X_train_norm, y_train
        )
        val_seq, val_labels, val_masks = preprocessor.construct_sequences(
            X_val_norm, y_val
        )
        test_seq, test_labels, test_masks = preprocessor.construct_sequences(
            X_test_norm, y_test
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            (train_seq, train_labels, train_masks),
            (val_seq, val_labels, val_masks),
            (test_seq, test_labels, test_masks),
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
        )
        
        # Class weights
        num_classes = len(np.unique(y_train))
        class_weights = compute_class_weights(y_train, num_classes)
        
        # Create model
        if config["model"]["name"] == "hss_rnet":
            model = HSSRNet(
                input_dim=config["data"]["num_features"],
                d_model=config["model"]["d_model"],
                d_state=config["model"]["d_state"],
                bigru_hidden=config["model"]["bigru_hidden"],
                num_classes=num_classes,
                dropout=config["model"]["dropout"],
            )
        else:
            model = S_SSM(
                input_dim=config["data"]["num_features"],
                d_model=config["model"]["d_model"],
                d_state=config["model"]["d_state"],
                bigru_hidden=config["model"]["bigru_hidden"],
                num_classes=num_classes,
                dropout=config["model"]["dropout"],
            )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {config['model']['name']}")
        print(f"Trainable parameters: {num_params:,}")
        
        # Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_weights=class_weights,
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            epochs=config["training"]["epochs"],
            early_stopping_patience=config["training"]["early_stopping_patience"],
            grad_clip=config["training"]["grad_clip"],
            num_classes=num_classes,
        )
        
        # Train
        results = trainer.train()
        
        # Save results
        run_metrics = results["test_metrics"]
        all_run_metrics.append(run_metrics)
        
        print(f"\nTest Results (Run {run + 1}):")
        print(f"  Accuracy:  {run_metrics['accuracy']:.4f}")
        print(f"  Precision: {run_metrics['precision']:.4f}")
        print(f"  Recall:    {run_metrics['recall']:.4f}")
        print(f"  F1-Score:  {run_metrics['f1']:.4f}")
        
        # Save model
        model_path = os.path.join(
            args.output_dir,
            f"{config['model']['name']}_run{run+1}.pt"
        )
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "metrics": run_metrics,
        }, model_path)
    
    # Aggregate results across runs
    if args.num_runs > 1:
        print(f"\n{'='*60}")
        print(f"Aggregated Results ({args.num_runs} runs)")
        print(f"{'='*60}")
        
        metrics_calc = Trainer.__new__(Trainer)
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = [m[metric] for m in all_run_metrics]
            print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")


if __name__ == "__main__":
    main()
