
"""
Anomaly Detection using Normalizing Flows for Network Traffic Analysis

This script implements anomaly detection framework using normalizing flows
to identify malicious network traffic patterns. 

Key Features:
- Normalizing flow-based density estimation
- Automated threshold selection using validation data
- Evaluation metrics (AUROC, precision, recall, F1-score)
- Support for multiple network flow feature sets (TOR2016, IDS2017)
"""

import argparse
import logging
import warnings
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import optax
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, 
    confusion_matrix, precision_score, recall_score, f1_score
)
from pzflow import Flow, FlowEnsemble
import shap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Feature sets for different datasets
TOR2016_FEATURES = [
    'Protocol', 'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min', 
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

IDS2017_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'SYN Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'Down/Up Ratio',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
    'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std',
    'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]


def normalize_and_filter(
    input_df: pd.DataFrame, 
    scaler: StandardScaler, 
    feature_filter: List[str], 
    is_fit: bool
) -> pd.DataFrame:
    """
    Normalize and filter features from input dataframe.
    
    Args:
        input_df: Input dataframe containing network flow features
        scaler: StandardScaler instance for normalization
        feature_filter: List of feature names to select
        is_fit: Whether to fit the scaler (True) or just transform (False)
    
    Returns:
        Normalized and filtered dataframe with selected features
        
    Raises:
        ValueError: If required features are missing from input dataframe
    """
    # Validate that all required features exist
    missing_features = set(feature_filter) - set(input_df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Filter and normalize features
    dataframe_filtered = input_df.filter(feature_filter)
    
    if is_fit:
        dataframe_scaled = scaler.fit_transform(dataframe_filtered[feature_filter])
    else:
        dataframe_scaled = scaler.transform(dataframe_filtered[feature_filter])
    
    return pd.DataFrame(dataframe_scaled, columns=feature_filter)



def get_feature_list(feature_set: str, feature_indexes: List[int]) -> List[str]:
    """
    Get the list of features based on feature set and indexes.
    
    Args:
        feature_set: Name of the feature set ('IDS2017' or 'TOR2016')
        feature_indexes: List of indexes to select from the feature set
    
    Returns:
        List of selected feature names
        
    Raises:
        ValueError: If feature set is invalid or indexes are out of range
    """
    if feature_set == "IDS2017":
        available_features = IDS2017_FEATURES
    elif feature_set == "TOR2016":
        available_features = TOR2016_FEATURES
    else:
        raise ValueError(f"Invalid feature set: {feature_set}")
    
    # Validate indexes
    max_index = len(available_features) - 1
    invalid_indexes = [idx for idx in feature_indexes if idx < 0 or idx > max_index]
    if invalid_indexes:
        raise ValueError(f"Invalid feature indexes {invalid_indexes}. Valid range: 0-{max_index}")
    
    # Select features by index
    selected_features = [available_features[idx] for idx in feature_indexes]
    
    logger.info(f"Selected {len(selected_features)} features from {feature_set} feature set:")
    for i, feature in enumerate(selected_features, 1):
        logger.info(f"  {i:2d}. {feature}")
    
    return selected_features


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the anomaly detection script.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Anomaly Detection using Normalizing Flows for Network Traffic Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data format and paths
    parser.add_argument(
        "--data-format", 
        choices=["csv", "pkl"], 
        default="pkl",
        help="Data format: 'csv' for single CSV file with Label column, 'pkl' for separate pickle files (default: pkl)"
    )
    
    parser.add_argument(
        "--benign-path",
        type=str,
        default="dataset/sample/BENIGN.pkl",
        help="Path to benign data file (CSV or PKL) (default: dataset/sample/BENIGN.pkl)"
    )
    
    parser.add_argument(
        "--attack-path",
        type=str,
        default="dataset/sample/ATTACK.pkl",
        help="Path to attack data file (required only for PKL format) (default: dataset/sample/ATTACK.pkl)"
    )
    
    # Data splitting parameters
    parser.add_argument(
        "--train-size",
        type=int,
        default=5000,
        help="Number of samples for training"
    )
    
    parser.add_argument(
        "--val-size",
        type=int,
        default=1000,
        help="Number of samples for validation"
    )
    
    parser.add_argument(
        "--test-size",
        type=int,
        default=5000,
        help="Number of samples for testing"
    )
    
    # Model parameters
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose model training output"
    )
    
    parser.add_argument(
        "--evaluate-threshold",
        action="store_true",
        default=False,
        help="Evaluate threshold-based metrics (Precision, Recall, F1)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for model training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="sgd",
        help="Optimizer type for training"
    )
    
    # Feature selection parameters
    parser.add_argument(
        "--feature-set",
        choices=["IDS2017", "TOR2016"],
        default="IDS2017",
        help="Feature set to use: 'IDS2017' or 'TOR2016' (default: IDS2017)"
    )
    
    parser.add_argument(
        "--feature-indexes",
        type=int,
        nargs="+",
        default=[31, 13, 35, 36, 18, 27],  # Default indexes for current features
        help="List of feature indexes to select from the chosen feature set"
    )
    
    return parser.parse_args()


def load_data_from_csv(csv_path: str, train_size: int, val_size: int, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from CSV file and split based on Label column.
    
    Args:
        csv_path: Path to CSV file
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
    
    Returns:
        Tuple containing (train_benign, val_benign, test_benign, val_attack, test_attack)
    """
    logger.info(f"Loading data from CSV file: {csv_path}")
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} total samples from CSV")
    
    # Check for Label column
    if 'Label' not in df.columns:
        raise ValueError("CSV file must contain a 'Label' column")
    
    # Split data based on Label column
    df_benign = df[df['Label'] == 0].drop('Label', axis=1)
    df_attack = df[df['Label'] == 1].drop('Label', axis=1)
    
    logger.info(f"Found {len(df_benign):,} benign samples and {len(df_attack):,} attack samples")
    
    # Check if we have enough data
    total_benign_needed = train_size + val_size + test_size
    total_attack_needed = val_size + test_size
    
    if len(df_benign) < total_benign_needed:
        logger.warning(f"Not enough benign samples. Need {total_benign_needed}, have {len(df_benign)}")
        # Use all available benign data
        train_size = min(train_size, len(df_benign) - val_size - test_size)
        if train_size < 0:
            train_size = max(0, len(df_benign) - val_size - test_size)
    
    if len(df_attack) < total_attack_needed:
        logger.warning(f"Not enough attack samples. Need {total_attack_needed}, have {len(df_attack)}")
        # Use all available attack data
        val_size = min(val_size, len(df_attack) - test_size)
        if val_size < 0:
            val_size = max(0, len(df_attack) - test_size)
    
    # Shuffle and split benign data
    df_benign_shuffled = df_benign.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train_benign = df_benign_shuffled.iloc[:train_size]
    df_val_benign = df_benign_shuffled.iloc[train_size:train_size + val_size]
    df_test_benign = df_benign_shuffled.iloc[train_size + val_size:train_size + val_size + test_size]
    
    # Shuffle and split attack data
    df_attack_shuffled = df_attack.sample(frac=1, random_state=42).reset_index(drop=True)
    df_val_attack = df_attack_shuffled.iloc[:val_size]
    df_test_attack = df_attack_shuffled.iloc[val_size:val_size + test_size]
    
    logger.info("CSV data preparation completed successfully")
    return df_train_benign, df_val_benign, df_test_benign, df_val_attack, df_test_attack


def load_data_from_pkl(benign_path: str, attack_path: str, train_size: int, val_size: int, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from separate PKL files.
    
    Args:
        benign_path: Path to benign data PKL file
        attack_path: Path to attack data PKL file
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
    
    Returns:
        Tuple containing (train_benign, val_benign, test_benign, val_attack, test_attack)
    """
    logger.info("Loading data from PKL files...")
    
    # Load benign data
    logger.info(f"Loading benign data from {benign_path}")
    df_benign = pd.read_pickle(benign_path)
    logger.info(f"Loaded {len(df_benign):,} benign samples")
    
    # Load attack data
    logger.info(f"Loading attack data from {attack_path}")
    df_attack = pd.read_pickle(attack_path)
    logger.info(f"Loaded {len(df_attack):,} attack samples")
    
    # Check if we have enough data
    total_benign_needed = train_size + val_size + test_size
    total_attack_needed = val_size + test_size
    
    if len(df_benign) < total_benign_needed:
        logger.warning(f"Not enough benign samples. Need {total_benign_needed}, have {len(df_benign)}")
        # Use all available benign data
        train_size = min(train_size, len(df_benign) - val_size - test_size)
        if train_size < 0:
            train_size = max(0, len(df_benign) - val_size - test_size)
    
    if len(df_attack) < total_attack_needed:
        logger.warning(f"Not enough attack samples. Need {total_attack_needed}, have {len(df_attack)}")
        # Use all available attack data
        val_size = min(val_size, len(df_attack) - test_size)
        if val_size < 0:
            val_size = max(0, len(df_attack) - test_size)
    
    # Shuffle and split benign data
    df_benign_shuffled = df_benign.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train_benign = df_benign_shuffled.iloc[:train_size]
    df_val_benign = df_benign_shuffled.iloc[train_size:train_size + val_size]
    df_test_benign = df_benign_shuffled.iloc[train_size + val_size:train_size + val_size + test_size]
    
    # Shuffle and split attack data
    df_attack_shuffled = df_attack.sample(frac=1, random_state=42).reset_index(drop=True)
    df_val_attack = df_attack_shuffled.iloc[:val_size]
    df_test_attack = df_attack_shuffled.iloc[val_size:val_size + test_size]
    
    logger.info("PKL data preparation completed successfully")
    return df_train_benign, df_val_benign, df_test_benign, df_val_attack, df_test_attack


def load_and_prepare_data(
    data_format: str,
    benign_path: str,
    attack_path: Optional[str] = None,
    train_size: int = 5000,
    val_size: int = 1000,
    test_size: int = 5000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare datasets for training, validation, and testing.
    
    Args:
        data_format: Data format ('csv' or 'pkl')
        benign_path: Path to benign data file
        attack_path: Path to attack data file (required for PKL format)
        train_size: Number of samples for training
        val_size: Number of samples for validation
        test_size: Number of samples for testing
    
    Returns:
        Tuple containing (train_benign, val_benign, test_benign, val_attack, test_attack)
    """
    if data_format == "csv":
        if attack_path is not None:
            logger.warning("Attack path provided but data format is CSV. Attack path will be ignored.")
        return load_data_from_csv(benign_path, train_size, val_size, test_size)
    elif data_format == "pkl":
        if attack_path is None:
            raise ValueError("Attack path is required when using PKL format")
        return load_data_from_pkl(benign_path, attack_path, train_size, val_size, test_size)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")


def train_normalizing_flow(
    train_data: pd.DataFrame,
    learning_rate: float = 0.001,
    epochs: int = 100,
    optimizer_type: str = "sgd",
    verbose: bool = True
) -> Flow:
    """
    Train a normalizing flow model on the provided training data.
    
    Args:
        train_data: Training data dataframe
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
        optimizer_type: Type of optimizer ('sgd' or 'adam')
        verbose: Whether to print training progress
    
    Returns:
        Trained normalizing flow model
    """
    logger.info(f"Training normalizing flow model for {epochs} epochs with {optimizer_type} optimizer...")
    
    # Select optimizer
    if optimizer_type == "sgd":
        optimizer = optax.sgd(learning_rate=learning_rate)
    elif optimizer_type == "adam":
        optimizer = optax.adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    flow_model = Flow(train_data.keys())
    losses = flow_model.train(
        train_data, 
        optimizer=optimizer,
        epochs=epochs,
        verbose=verbose
    )
    
    logger.info("Normalizing flow training completed")
    return flow_model


def find_optimal_threshold(
    flow_model: Flow,
    val_benign: pd.DataFrame,
    val_attack: pd.DataFrame,
    scaler: StandardScaler,
    feature_filter: List[str]
) -> Tuple[float, float, float, float]:
    """
    Find optimal threshold using validation data.
    
    Args:
        flow_model: Trained normalizing flow model
        val_benign: Validation benign data
        val_attack: Validation attack data
        scaler: Fitted scaler for normalization
        feature_filter: List of features to use
    
    Returns:
        Tuple containing (optimal_threshold, val_auc, val_tpr, val_fpr)
    """
    logger.info("Finding optimal threshold using validation data...")
    
    # Prepare validation data
    val_benign_filtered = normalize_and_filter(val_benign, scaler, feature_filter, False)
    val_attack_filtered = normalize_and_filter(val_attack, scaler, feature_filter, False)
    
    # Calculate log probabilities
    val_benign_prob = flow_model.log_prob(val_benign_filtered)
    val_attack_prob = flow_model.log_prob(val_attack_filtered)
    
    # Create validation data for threshold selection
    val_data = np.concatenate((val_attack_prob * -1, val_benign_prob * -1))
    val_labels = np.concatenate((np.ones(len(val_attack_prob)), np.zeros(len(val_benign_prob))))
    
    # Find optimal threshold using Youden's J statistic
    val_fpr, val_tpr, val_thresholds = roc_curve(val_labels, val_data)
    optimal_threshold_idx = np.argmax(val_tpr - val_fpr)
    optimal_threshold = val_thresholds[optimal_threshold_idx]
    
    # Calculate validation AUROC
    val_auc = roc_auc_score(val_labels, val_data)
    
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}, Validation AUROC: {val_auc:.4f}")
    
    return optimal_threshold, val_auc, val_tpr[optimal_threshold_idx], val_fpr[optimal_threshold_idx]


def evaluate_model(
    flow_model: Flow,
    test_benign: pd.DataFrame,
    test_attack: pd.DataFrame,
    scaler: StandardScaler,
    feature_filter: List[str],
    optimal_threshold: Optional[float] = None,
    evaluate_threshold: bool = True
) -> Dict[str, Any]:
    """
    Evaluate the trained model on test data.
    
    Args:
        flow_model: Trained normalizing flow model
        test_benign: Test benign data
        test_attack: Test attack data
        scaler: Fitted scaler for normalization
        feature_filter: List of features to use
        optimal_threshold: Optimal threshold for classification (required if evaluate_threshold=True)
        evaluate_threshold: Whether to evaluate threshold-based metrics
    
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model on test data...")
    
    # Prepare test data
    test_benign_filtered = normalize_and_filter(test_benign, scaler, feature_filter, False)
    test_attack_filtered = normalize_and_filter(test_attack, scaler, feature_filter, False)
    
    # Calculate log probabilities
    test_benign_prob = flow_model.log_prob(test_benign_filtered)
    test_attack_prob = flow_model.log_prob(test_attack_filtered)
    
    # Create test data for evaluation
    test_data = np.concatenate((test_attack_prob * -1, test_benign_prob * -1))
    test_labels = np.concatenate((np.ones(len(test_attack_prob)), np.zeros(len(test_benign_prob))))
    
    # Calculate basic metrics (always available)
    test_auc = roc_auc_score(test_labels, test_data)
    
    metrics = {
        'test_auc': test_auc,
        'test_data': test_data,
        'test_labels': test_labels
    }
    
    # Calculate threshold-based metrics if requested
    if evaluate_threshold:
        if optimal_threshold is None:
            raise ValueError("optimal_threshold is required when evaluate_threshold=True")
        
        # Calculate predictions
        predictions = (test_data >= optimal_threshold).astype(int)
        
        # Calculate threshold-based metrics
        tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tpr': recall,
            'fpr': fp / (fp + tn),
            'optimal_threshold': optimal_threshold
        })
    
    return metrics


def print_results(
    train_size: int,
    val_benign_size: int,
    val_attack_size: int,
    test_benign_size: int,
    test_attack_size: int,
    feature_filter: List[str],
    val_auc: Optional[float] = None,
    optimal_threshold: Optional[float] = None,
    val_tpr: Optional[float] = None,
    val_fpr: Optional[float] = None,
    metrics: Dict[str, Any] = None,
    evaluate_threshold: bool = True
) -> None:
    """
    Print comprehensive results in a formatted manner.
    
    Args:
        train_size: Number of training samples
        val_benign_size: Number of validation benign samples
        val_attack_size: Number of validation attack samples
        test_benign_size: Number of test benign samples
        test_attack_size: Number of test attack samples
        feature_filter: List of features used
        val_auc: Validation AUROC (optional)
        optimal_threshold: Optimal threshold (optional)
        val_tpr: Validation TPR (optional)
        val_fpr: Validation FPR (optional)
        metrics: Dictionary containing test metrics
        evaluate_threshold: Whether threshold-based metrics were evaluated
    """
    print("=" * 80)
    print("NORMALIZING FLOW ANOMALY DETECTION RESULTS")
    print("=" * 80)
    
    print(f"\n DATASET INFO:")
    print(f"   • Training samples (benign): {train_size:,}")
    print(f"   • Validation samples (benign): {val_benign_size:,}")
    print(f"   • Validation samples (attack): {val_attack_size:,}")
    print(f"   • Test samples (benign): {test_benign_size:,}")
    print(f"   • Test samples (attack): {test_attack_size:,}")
    print(f"   • Total test samples: {test_benign_size + test_attack_size:,}")
    
    print(f"\n FEATURES USED ({len(feature_filter)} features):")
    for i, feature in enumerate(feature_filter, 1):
        print(f"   {i:2d}. {feature}")
    
    if evaluate_threshold and val_auc is not None:
        print(f"\n VALIDATION SET METRICS (Threshold Selection):")
        print(f"   • Validation AUROC: {val_auc:.4f}")
        print(f"   • Optimal Threshold: {optimal_threshold:.4f}")
        print(f"   • Validation TPR: {val_tpr:.4f}")
        print(f"   • Validation FPR: {val_fpr:.4f}")
    
    print(f"\n TEST SET METRICS (Final Evaluation):")
    print(f"   • Test AUROC: {metrics['test_auc']:.4f}")
    
    if evaluate_threshold and 'tpr' in metrics:
        print(f"   • Test TPR: {metrics['tpr']:.4f}")
        print(f"   • Test FPR: {metrics['fpr']:.4f}")
        
        print(f"\n CONFUSION MATRIX:")
        print(f"   • True Positives (TP):  {metrics['tp']:,}")
        print(f"   • True Negatives (TN):  {metrics['tn']:,}")
        print(f"   • False Positives (FP): {metrics['fp']:,}")
        print(f"   • False Negatives (FN): {metrics['fn']:,}")
        
        print(f"\n CLASSIFICATION METRICS:")
        print(f"   • Precision: {metrics['precision']:.4f}")
        print(f"   • Recall:    {metrics['recall']:.4f}")
        print(f"   • F1-Score:  {metrics['f1']:.4f}")
    else:
        print("   • Threshold-based metrics not evaluated (--evaluate-threshold not set)")
    
    print("=" * 80)


def main():
    """
    Main execution function for the anomaly detection pipeline.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        logger.info("Starting anomaly detection pipeline...")
        logger.info(f"Configuration: {args.data_format} format, train={args.train_size}, val={args.val_size}, test={args.test_size}")
        logger.info(f"Model: {args.optimizer} optimizer, lr={args.learning_rate}, epochs={args.epochs}")
        logger.info(f"Threshold evaluation: {'enabled' if args.evaluate_threshold else 'disabled'}")
        
        # Get feature list based on arguments
        feature_filter = get_feature_list(args.feature_set, args.feature_indexes)
        
        # Load and prepare data
        df_train, df_val_benign, df_test_benign, df_val_attack, df_test_attack = load_and_prepare_data(
            args.data_format, args.benign_path, args.attack_path, 
            args.train_size, args.val_size, args.test_size
        )
        
        # Initialize scaler and normalize training data
        scaler = StandardScaler()
        df_train_filtered = normalize_and_filter(df_train, scaler, feature_filter, True)
        
        # Train normalizing flow model
        flow_model = train_normalizing_flow(
            df_train_filtered, 
            learning_rate=args.learning_rate, 
            epochs=args.epochs,
            optimizer_type=args.optimizer,
            verbose=args.verbose
        )
        
        # Initialize variables for threshold evaluation
        optimal_threshold = None
        val_auc = None
        val_tpr = None
        val_fpr = None
        
        # Find optimal threshold using validation data (if requested)
        if args.evaluate_threshold:
            optimal_threshold, val_auc, val_tpr, val_fpr = find_optimal_threshold(
                flow_model, df_val_benign, df_val_attack, scaler, feature_filter
            )
        
        # Evaluate model on test data
        metrics = evaluate_model(
            flow_model, df_test_benign, df_test_attack, scaler, feature_filter, 
            optimal_threshold, args.evaluate_threshold
        )
        
        # Print results
        print_results(
            len(df_train), len(df_val_benign), len(df_val_attack),
            len(df_test_benign), len(df_test_attack), feature_filter,
            val_auc, optimal_threshold, val_tpr, val_fpr, metrics, args.evaluate_threshold
        )
        
        logger.info("Anomaly detection pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error occurred during execution: {e}")
        raise


if __name__ == "__main__":
    main()       