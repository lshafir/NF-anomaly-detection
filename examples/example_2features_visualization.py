import optax
from pzflow import Flow
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# IDS2017 feature list from main script
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

# Feature pairs to analyze
FEATURE_PAIRS = [
    ('Flow Duration', 'Bwd Packet Length Max'),  # Discriminative features
    ('Total Fwd Packets', 'Fwd Packet Length Min')  # Non-discriminative features
]

def train_flow_and_evaluate(train_data, test_benign, test_attack, feature_pair):
    """Train normalizing flow and evaluate AUROC"""
    
    # Train flow on benign data
    flow = Flow(train_data.keys())
    losses = flow.train(train_data, optimizer=optax.sgd(learning_rate=0.001), verbose=True)
    
    # Evaluate on test data
    benign_probs = flow.log_prob(test_benign)
    attack_probs = flow.log_prob(test_attack)
    
    # Create labels and scores
    scores = np.concatenate([-attack_probs, -benign_probs])
    labels = np.concatenate([np.ones(len(attack_probs)), np.zeros(len(benign_probs))])
    
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    
    return auroc, losses, fpr, tpr

if __name__ == '__main__':
    # Load data
    df_benign = pd.read_pickle("dataset/sample/BENIGN.pkl")
    df_attack = pd.read_pickle("dataset/sample/ATTACK.pkl")
    
    # Sample 5000 benign samples for training
    df_train = df_benign.sample(n=5000, random_state=42)
    
    # Sample test data
    df_test_benign = df_benign.sample(n=1000, random_state=42)
    df_test_attack = df_attack.sample(n=1000, random_state=42)
    
    for i, (feature_x, feature_y) in enumerate(FEATURE_PAIRS):
        print(f"\nAnalyzing pair {i+1}: {feature_x} vs {feature_y}")
        
        # Prepare data
        X_train = df_train[[feature_x, feature_y]].copy()
        X_test_benign = df_test_benign[[feature_x, feature_y]].copy()
        X_test_attack = df_test_attack[[feature_x, feature_y]].copy()
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_benign_scaled = scaler.transform(X_test_benign)
        X_test_attack_scaled = scaler.transform(X_test_attack)
        
        # Convert to DataFrame for Flow
        train_df = pd.DataFrame(X_train_scaled, columns=[feature_x, feature_y])
        test_benign_df = pd.DataFrame(X_test_benign_scaled, columns=[feature_x, feature_y])
        test_attack_df = pd.DataFrame(X_test_attack_scaled, columns=[feature_x, feature_y])
        
        # Train flow and evaluate
        auroc, losses, fpr, tpr = train_flow_and_evaluate(train_df, test_benign_df, test_attack_df, (feature_x, feature_y))
        print(f"AUROC: {auroc:.4f}")
        
        # Create visualization
        fig = plt.figure(figsize=(15, 6))
        
        # Left: Scatter plot
        ax1 = plt.subplot(1, 2, 1)
        ax1.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], s=5, c='blue', alpha=0.6, label='Benign')
        ax1.scatter(X_test_attack_scaled[:, 0], X_test_attack_scaled[:, 1], s=5, c='red', alpha=0.6, label='Attack')
        ax1.set_xlabel(f'{feature_x} (normalized)')
        ax1.set_ylabel(f'{feature_y} (normalized)')
        ax1.set_title(f'2D Distribution: {feature_x} vs {feature_y}\nAUROC: {auroc:.4f}')
        ax1.legend()
        ax1.grid(True)
        
        # Right: Split into two subplots vertically
        # Top: Training loss
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(losses)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True)
        
        # Bottom: ROC curve
        ax3 = plt.subplot(2, 2, 4)
        ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auroc:.4f})')
        ax3.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve')
        ax3.legend()
        ax3.grid(True)
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'images/{feature_x.replace(" ", "_")}_vs_{feature_y.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()