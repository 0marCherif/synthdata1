#!/usr/bin/env python3
"""
Evaluation script for synthetic data classification.
Converts the evaluation notebook into a standalone Python script.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLPClassifier(nn.Module):
    """Multi-layer perceptron classifier for text classification."""
    
    def __init__(self, input_dim=4096, hidden_dim=128, num_classes=6):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def equal_distribution(data, labels, amount):
    """
    Create a balanced dataset with equal distribution of labels.
    
    Args:
        data: List of text data
        labels: List of corresponding labels
        amount: Number of samples per label
        
    Returns:
        Dictionary with balanced data per label
    """
    new_dataset = {}
    label_counter = {}
    
    for label in labels:
        label_counter[label] = 0
        new_dataset[label] = []
    
    for label, text in zip(labels, data):
        if label_counter[label] < amount:
            new_dataset[label].append(text)
            label_counter[label] += 1
    
    return new_dataset


def add_header_to_csv(input_file, output_file, header="label,text"):
    """
    Add header to CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file with header
        header: Header string to add
    """
    logger.info(f"Adding header to {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write(header + '\n')  # Write header first
        for line in f_in:            # Copy rest of the file line by line
            f_out.write(line)


def load_synthetic_data(csv_path):
    """
    Load synthetic data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Loading synthetic data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    logger.info(f"Loaded {len(texts)} samples with {len(set(labels))} unique labels")
    return texts, labels


def load_acl_arc_dataset():
    """
    Load ACL-ARC dataset for validation and testing.
    
    Returns:
        Tuple of (train, validation, test) datasets
    """
    logger.info("Loading ACL-ARC dataset")
    
    ds = load_dataset("hrithikpiyush/acl-arc")
    ds.set_format("pandas")
    
    train = ds["train"][:]
    validation = ds["validation"][:]
    test = ds["test"][:]
    
    logger.info(f"Loaded ACL-ARC: train={len(train)} samples, "
                f"validation={len(validation)} samples, test={len(test)} samples")
    
    return train, validation, test


def initialize_embedding_model():
    """
    Initialize the sentence transformer embedding model.
    
    Returns:
        SentenceTransformer model
    """
    logger.info("Initializing Qwen3-Embedding-8B model")
    
    # Define arguments for 4-bit quantization
    model_kwargs = {
        "device_map": "auto",         # Automatically map layers to devices (GPU/CPU)
        "torch_dtype": torch.bfloat16,  # Recommended dtype for Qwen models
        "load_in_4bit": True,           # Enable 4-bit quantization
    }
    
    embedding_model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        model_kwargs=model_kwargs,
        trust_remote_code=True,  # Often required for custom model architectures
    )
    
    logger.info("Embedding model initialized successfully")
    return embedding_model


def train_classifier(model, train_loader, num_epochs=10, learning_rate=0.001):
    """
    Train the MLP classifier.
    
    Args:
        model: MLPClassifier model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained model
    """
    logger.info(f"Training classifier for {num_epochs} epochs")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()           # Reset gradients from previous step
            outputs = model(X_batch)        # Forward pass
            loss = criterion(outputs, y_batch)
            loss.backward()                 # Compute gradients
            optimizer.step()                # Update weights
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    logger.info("Training completed")
    return model


def evaluate_model(model, validation_data, validation_labels, test_data, test_labels):
    """
    Evaluate the trained model on validation and test sets.
    
    Args:
        model: Trained MLPClassifier model
        validation_data: Validation embeddings
        validation_labels: Validation labels
        test_data: Test embeddings
        test_labels: Test labels
        
    Returns:
        Tuple of (validation_report, test_report)
    """
    logger.info("Evaluating model on validation and test sets")
    
    with torch.no_grad():
        validation_outputs = model(validation_data)
        test_outputs = model(test_data)
    
    validation_predictions = torch.argmax(validation_outputs, axis=1)
    test_predictions = torch.argmax(test_outputs, axis=1)
    
    validation_report = classification_report(
        validation_labels, validation_predictions, output_dict=True
    )
    test_report = classification_report(
        test_labels, test_predictions, output_dict=True
    )
    
    logger.info("Validation Results:")
    logger.info(f"Accuracy: {validation_report['accuracy']:.4f}")
    
    logger.info("Test Results:")
    logger.info(f"Accuracy: {test_report['accuracy']:.4f}")
    
    return validation_report, test_report


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Evaluate synthetic data classification')
    parser.add_argument('--input-csv', default='DATASET6/final.csv',
                       help='Path to input CSV file (default: DATASET6/final.csv)')
    parser.add_argument('--output-csv', default='DATASET6/final_with_header.csv',
                       help='Path to output CSV file with header (default: DATASET6/final_with_header.csv)')
    parser.add_argument('--samples-per-class', type=int, default=10,
                       help='Number of samples per class for balanced dataset (default: 10)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for optimizer (default: 0.001)')
    
    args = parser.parse_args()
    
    try:
        # Step 1: Add header to CSV file
        add_header_to_csv(args.input_csv, args.output_csv)
        
        # Step 2: Load synthetic data
        texts, labels = load_synthetic_data(args.output_csv)
        
        # Step 3: Create balanced dataset
        logger.info(f"Creating balanced dataset with {args.samples_per_class} samples per class")
        balanced_dataset = equal_distribution(texts, labels, args.samples_per_class)
        
        # Flatten balanced dataset
        balanced_texts = []
        balanced_labels = []
        for label, text_list in balanced_dataset.items():
            balanced_texts.extend(text_list)
            balanced_labels.extend([label] * len(text_list))
        
        logger.info(f"Balanced dataset: {len(balanced_texts)} samples")
        
        # Step 4: Load ACL-ARC dataset
        train, validation, test = load_acl_arc_dataset()
        
        # Step 5: Initialize embedding model
        embedding_model = initialize_embedding_model()
        
        # Step 6: Generate embeddings
        logger.info("Generating embeddings for synthetic data")
        synth_embeddings = torch.from_numpy(embedding_model.encode(balanced_texts))
        
        logger.info("Generating embeddings for validation data")
        validation_embeddings = torch.from_numpy(
            embedding_model.encode(validation['cleaned_cite_text'].tolist())
        )
        
        logger.info("Generating embeddings for test data")
        test_embeddings = torch.from_numpy(
            embedding_model.encode(test['cleaned_cite_text'].tolist())
        )
        
        # Step 7: Initialize and train classifier
        logger.info("Initializing MLP classifier")
        model = MLPClassifier(
            input_dim=synth_embeddings.shape[1],
            hidden_dim=128,
            num_classes=len(set(balanced_labels))
        )
        
        # Create data loader
        X_train = synth_embeddings
        y_train = torch.tensor(balanced_labels)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True
        )
        
        # Train model
        trained_model = train_classifier(
            model, train_loader, args.epochs, args.learning_rate
        )
        
        # Step 8: Evaluate model
        validation_report, test_report = evaluate_model(
            trained_model,
            validation_embeddings,
            validation['intent'].tolist(),
            test_embeddings,
            test['intent'].tolist()
        )
        
        # Print detailed results
        print("\n" + "="*50)
        print("VALIDATION RESULTS")
        print("="*50)
        print(classification_report(
            validation['intent'].tolist(),
            torch.argmax(trained_model(validation_embeddings), axis=1)
        ))
        
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        print(classification_report(
            test['intent'].tolist(),
            torch.argmax(trained_model(test_embeddings), axis=1)
        ))
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
