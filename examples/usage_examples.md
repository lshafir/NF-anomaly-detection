# Usage Examples for NF-anomaly-detection Script

This document provides comprehensive examples of how to use the `nf_anomaly_detection.py` script with different configurations and data formats.

## Basic Usage

### Using Default Settings (PKL Files)

```bash
python examples/nf_anomaly_detection.py
```

This uses the default configuration:
- Data format: PKL
- Benign path: `dataset/sample/BENIGN.pkl`
- Attack path: `dataset/sample/ATTACK.pkl`
- Feature set: IDS2017
- Feature indexes: [31, 13, 35, 36, 18, 27] (Packet Length Mean, Flow IAT Min, PSH Flag Count, ACK Flag Count, Fwd IAT Max, Fwd Packets/s)
- Verbose output: Enabled
- Threshold evaluation: Disabled (AUROC only)

### Using CSV File with Label Column

```bash
python examples/nf_anomaly_detection.py \
    --data-format csv \
    --benign-path dataset/sample/combined_data.csv
```

### Using Custom PKL Files

```bash
python examples/nf_anomaly_detection.py \
    --benign-path path/to/your/benign.pkl \
    --attack-path path/to/your/attack.pkl
```

## Advanced Configuration

### Custom Data Splitting

```bash
python examples/nf_anomaly_detection.py \
    --train-size 10000 \
    --val-size 2000 \
    --test-size 5000
```

### Feature Selection

#### Using TOR2016 Feature Set

```bash
python examples/nf_anomaly_detection.py \
    --feature-set TOR2016 \
    --feature-indexes 0 1 2 3 4 5
```

#### Using Custom IDS2017 Features

```bash
python examples/nf_anomaly_detection.py \
    --feature-set IDS2017 \
    --feature-indexes 0 1 2 3 4 5 6 7 8 9
```

#### Using Specific Features (e.g., Flow Duration, Total Fwd Packets, etc.)

```bash
python examples/nf_anomaly_detection.py \
    --feature-indexes 0 1 2 32 13 33
```

### Model Configuration

```bash
python examples/nf_anomaly_detection.py \
    --data-format pkl \
    --benign-path dataset/sample/BENIGN.pkl \
    --attack-path dataset/sample/ATTACK.pkl \
    --learning-rate 0.01 \
    --epochs 200 \
    --optimizer adam \
    --verbose
```

### Enable Threshold Evaluation (Full Metrics)

```bash
python examples/nf_anomaly_detection.py \
    --evaluate-threshold \
    --learning-rate 0.001 \
    --epochs 50
```

### AUROC Only (Default)

```bash
python examples/nf_anomaly_detection.py \
    --learning-rate 0.001 \
    --epochs 50
```

## Complete Example with All Parameters

```bash
python examples/nf_anomaly_detection.py \
    --train-size 8000 \
    --val-size 1500 \
    --test-size 6000 \
    --learning-rate 0.005 \
    --epochs 150 \
    --optimizer adam \
    --verbose \
    --evaluate-threshold
```

## Parameter Reference

### Required Parameters

Noneץ All parameters have defaults.

### Optional Parameters

#### Data Configuration
- `--data-format`: Data format (`csv` or `pkl`, default: `pkl`)
- `--benign-path`: Path to benign data file (default: `dataset/sample/BENIGN.pkl`)
- `--attack-path`: Path to attack data file (default: `dataset/sample/ATTACK.pkl`)

#### Data Splitting
- `--train-size`: Number of training samples (default: 5000)
- `--val-size`: Number of validation samples (default: 1000)
- `--test-size`: Number of test samples (default: 5000)

#### Feature Selection
- `--feature-set`: Feature set to use (`IDS2017` or `TOR2016`, default: `IDS2017`)
- `--feature-indexes`: List of feature indexes to select (default: `32 13 33 27 29 34`)

#### Model Configuration
- `--verbose`: Enable verbose model training output (default: True)
- `--evaluate-threshold`: Evaluate threshold-based metrics (default: False)
- `--learning-rate`: Learning rate for training (default: 0.001)
- `--epochs`: Number of training epochs (default: 100)
- `--optimizer`: Optimizer type (`sgd` or `adam`, default: `sgd`)

## Data Format Requirements

### CSV Format
- Must contain a `Label` column with values 0 (benign) and 1 (attack)
- All other columns are treated as features
- Example:
```csv
Feature1,Feature2,Feature3,Label
1.2,3.4,5.6,0
2.1,4.3,6.5,1
...
```

### PKL Format
- Two separate pickle files containing pandas DataFrames
- Benign file: DataFrame with only benign samples
- Attack file: DataFrame with only attack samples

