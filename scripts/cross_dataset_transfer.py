#!/usr/bin/env python
"""
Cross-dataset transfer evaluation (zero-shot generalization).
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
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-dataset transfer")
    parser.add_argument("--source_data", type=str, required=True)
    parser.add_argument("--target_data", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="hss_rnet")
    parser.add_argument("--output_dir", type=str, default="transfer_results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    # Load source and target data
    source_data = np.load(args.source_data, allow_pickle=True)
    target_data = np.load(args.target_data, allow_pickle=True)
    
    X_source_train = source_data["X_train"]
    y_source_train = source_data["y_train"]
    X_source_val = source_data["X_val"]
    y_source_val = source_data["y_val"]
    
    X_target_test = target_data["X_test"]
    y_target_test = target_data["y_test"]
    
    # Preprocessing: fit on source only
    preprocessor = PreprocessingPipeline(sequence_length=30, stride=30)
    
    X_source_train_norm = preprocessor.fit_transform(X_source_train)
    X_source_val_norm = preprocessor.transform(X_source_val)
    X_target_norm = preprocessor.transform(X_target_test)  # No target statistics!
    
    # Construct sequences
    train_seq, train_labels, train_masks = preprocessor.construct_sequences(
        X_source_train_norm, y_source_train
    )
    val_seq, val_labels, val_masks = preprocessor.construct_sequences(
        X_source_val_norm, y_source_val
    )
    test_seq, test_labels, test_masks = preprocessor.construct_sequences(
        X_target_norm, y_target_test
    )
    
    # Binary mapping: benign=0, malicious=1
    train_labels = (train_labels > 0).astype(np.int64)
    val_labels = (val_labels > 0).astype(np.int64)
    test_labels = (test_labels > 0).astype(np.int64)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        (train_seq, train_labels, train_masks),
        (val_seq, val_labels, val_masks),
        (test_seq, test_labels, test_masks),
        batch_size=64,
    )
    
    # Class weights
    class_weights = compute_class_weights(train_labels, 2)
    
    # Create model
    if args.model_type == "hss_rnet":
        model = HSSRNet(
            input_dim=11,
            d_model=128,
            d_state=256,
            bigru_hidden=128,
            num_classes=2,
        )
    else:
        model = S_SSM(
            input_dim=11,
            d_model=128,
            d_state=256,
            bigru_hidden=128,
            num_classes=2,
        )
    
    # Train on source
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        learning_rate=1e-3,
        epochs=50,
        num_classes=2,
    )
    
    results = trainer.train()
    
    # Evaluate on target (zero-shot)
    target_loss, target_metrics = trainer.evaluate(test_loader)
    
    print(f"\nZero-Shot Transfer Results:")
    print(f"  Source: {args.source_data}")
    print(f"  Target: {args.target_data}")
    print(f"  Accuracy:  {target_metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {target_metrics['f1']:.4f}")
    
    if "auc" in target_metrics:
        print(f"  AUC:       {target_metrics['auc']:.4f}")
    
    # Save results
    import json
    results_dict = {
        "source": args.source_data,
        "target": args.target_data,
        "model_type": args.model_type,
        "metrics": {
            k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in target_metrics.items()
        },
    }
    
    with open(os.path.join(args.output_dir, "transfer_results.json"), "w") as f:
        json.dump(results_dict, f, indent=2)


if __name__ == "__main__":
    main()
