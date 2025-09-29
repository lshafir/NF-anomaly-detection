# Feature Reference for NF-anomaly-detection


## IDS2017 Features (Default)

| Index | Feature Name | Description |
|-------|-------------|-------------|
| 0 | Flow Duration | Duration of the flow |
| 1 | Total Fwd Packets | Total number of forward packets |
| 2 | Total Backward Packets | Total number of backward packets |
| 3 | Total Length of Fwd Packets | Total length of forward packets |
| 4 | Total Length of Bwd Packets | Total length of backward packets |
| 5 | Fwd Packet Length Min | Minimum forward packet length |
| 6 | Fwd Packet Length Mean | Mean forward packet length |
| 7 | Fwd Packet Length Std | Standard deviation of forward packet length |
| 8 | Bwd Packet Length Max | Maximum backward packet length |
| 9 | Bwd Packet Length Min | Minimum backward packet length |
| 10 | Bwd Packet Length Mean | Mean backward packet length |
| 11 | Flow IAT Mean | Mean inter-arrival time |
| 12 | Flow IAT Std | Standard deviation of inter-arrival time |
| 13 | Flow IAT Min | Minimum inter-arrival time |
| 14 | Flow IAT Max | Maximum inter-arrival time |
| 15 | Fwd IAT Total | Total forward inter-arrival time |
| 16 | Fwd IAT Mean | Mean forward inter-arrival time |
| 17 | Fwd IAT Std | Standard deviation of forward inter-arrival time |
| 18 | Fwd IAT Max | Maximum forward inter-arrival time |
| 19 | Fwd IAT Min | Minimum forward inter-arrival time |
| 20 | Bwd IAT Total | Total backward inter-arrival time |
| 21 | Bwd IAT Mean | Mean backward inter-arrival time |
| 22 | Bwd IAT Std | Standard deviation of backward inter-arrival time |
| 23 | Bwd IAT Max | Maximum backward inter-arrival time |
| 24 | Bwd IAT Min | Minimum backward inter-arrival time |
| 25 | Fwd Header Length | Forward header length |
| 26 | Bwd Header Length | Backward header length |
| 27 | Fwd Packets/s | Forward packets per second |
| 28 | Bwd Packets/s | Backward packets per second |
| 29 | Min Packet Length | Minimum packet length |
| 30 | Max Packet Length | Maximum packet length |
| 31 | Packet Length Mean | Mean packet length |
| 32 | Packet Length Std | Standard deviation of packet length |
| 33 | Packet Length Variance | Variance of packet length |
| 34 | SYN Flag Count | SYN flag count |
| 35 | PSH Flag Count | PSH flag count |
| 36 | ACK Flag Count | ACK flag count |
| 37 | URG Flag Count | URG flag count |
| 38 | Down/Up Ratio | Down/Up ratio |
| 39 | Average Packet Size | Average packet size |
| 40 | Avg Fwd Segment Size | Average forward segment size |
| 41 | Avg Bwd Segment Size | Average backward segment size |
| 42 | Subflow Fwd Packets | Subflow forward packets |
| 43 | Subflow Fwd Bytes | Subflow forward bytes |
| 44 | Subflow Bwd Packets | Subflow backward packets |
| 45 | Subflow Bwd Bytes | Subflow backward bytes |
| 46 | Init_Win_bytes_forward | Initial window bytes forward |
| 47 | Init_Win_bytes_backward | Initial window bytes backward |
| 48 | act_data_pkt_fwd | Active data packets forward |
| 49 | min_seg_size_forward | Minimum segment size forward |
| 50 | Active Mean | Mean active time |
| 51 | Active Std | Standard deviation of active time |
| 52 | Active Max | Maximum active time |
| 53 | Active Min | Minimum active time |
| 54 | Idle Mean | Mean idle time |
| 55 | Idle Std | Standard deviation of idle time |
| 56 | Idle Max | Maximum idle time |
| 57 | Idle Min | Minimum idle time |


## TOR2016 Features

| Index | Feature Name | Description |
|-------|-------------|-------------|
| 0 | Protocol | Protocol type |
| 1 | Flow Duration | Duration of the flow |
| 2 | Flow Bytes/s | Flow bytes per second |
| 3 | Flow Packets/s | Flow packets per second |
| 4 | Flow IAT Mean | Mean inter-arrival time |
| 5 | Flow IAT Std | Standard deviation of inter-arrival time |
| 6 | Flow IAT Max | Maximum inter-arrival time |
| 7 | Flow IAT Min | Minimum inter-arrival time |
| 8 | Fwd IAT Mean | Mean forward inter-arrival time |
| 9 | Fwd IAT Std | Standard deviation of forward inter-arrival time |
| 10 | Fwd IAT Max | Maximum forward inter-arrival time |
| 11 | Fwd IAT Min | Minimum forward inter-arrival time |
| 12 | Bwd IAT Mean | Mean backward inter-arrival time |
| 13 | Bwd IAT Std | Standard deviation of backward inter-arrival time |
| 14 | Bwd IAT Max | Maximum backward inter-arrival time |
| 15 | Bwd IAT Min | Minimum backward inter-arrival time |
| 16 | Active Mean | Mean active time |
| 17 | Active Std | Standard deviation of active time |
| 18 | Active Max | Maximum active time |
| 19 | Active Min | Minimum active time |
| 20 | Idle Mean | Mean idle time |
| 21 | Idle Std | Standard deviation of idle time |
| 22 | Idle Max | Maximum idle time |
| 23 | Idle Min | Minimum idle time |

## Usage Examples

### Select specific features by index:

```bash
# Use first 6 features from IDS2017
python examples/nf_anomaly_detection.py --feature-indexes 0 1 2 3 4 5

# Use TOR2016 features
python examples/nf_anomaly_detection.py --feature-set TOR2016 --feature-indexes 0 1 2 3 4 5
```

