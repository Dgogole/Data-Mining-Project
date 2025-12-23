## Part1 bank-marketing rrl
### 1. 结构

| Structure (Depth x Width) | Log(#Edges) | Accuracy | Macro F1 | Weighted F1 | Class 1 Recall | Class 1 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 5 x 64 (Baseline) | - | 0.89 | 0.63 | 0.87 | 0.21 | 0.32 |
| 10 x 32 | 3.22 | 0.90 | 0.68 | 0.88 | 0.30 | 0.41 |
| 10 x 32 x 16 | 4.82 | 0.90 | 0.71 | 0.89 | 0.38 | 0.47 |
| **16 x 128 (Best)** | **3.53** | **0.90** | **0.72** | **0.89** | **0.39** | **0.49** |

### 2. 学习率

| Learning Rate | Log(#Edges) | Accuracy | Macro F1 | Weighted F1 | Class 1 Recall | Class 1 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.002 | 3.50 | 0.90 | 0.68 | 0.88 | 0.32 | 0.42 |
| **0.005 (Optimal)** | **3.53** | **0.90** | **0.72** | **0.89** | **0.39** | **0.49** |
| 0.010 | 3.53 | 0.90 | 0.68 | 0.88 | 0.29 | 0.41 |

### 3. NLAF Tuning

| Exp ID | Alpha | Beta | Gamma | Log(#Edges) | Accuracy | Macro F1 | Class 1 Recall | Class 1 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3.0 (Best)** | **0.999** | **8** | **1** | **3.53** | **0.90** | **0.72** | **0.39** | **0.49** |
| 3.1 | 0.999 | 8 | *3* | 3.61 | 0.89 | 0.63 | 0.21 | 0.32 |
| 3.2 | *0.900* | *3* | 1 | 3.04 | 0.90 | 0.69 | 0.31 | 0.42 |
| 3.3 | *0.990* | *5* | 1 | 3.18 | 0.90 | 0.67 | 0.29 | 0.40 |


### 4. 其他实验

| Experiment | Log(#Edges) | Accuracy | Macro F1 | Class 1 Recall | Class 1 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| True Mode | 3.53 | 0.90 | 0.72 | **0.39** | **0.49** |
| Not True Mode | 3.53 | 0.90 | 0.67 | 0.29 | 0.40 |

### 5. 基线

| Baseline | Accuracy | Macro F1 | Weighted F1 | Class 1 Recall | Class 1 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DCN | 0.9034 | 0.7654 | 0.9032 | 0.5836 | 0.5855 |
| GradientBoostingClassifier (n_estimators=200) | 0.9073 | 0.7371 | 0.8992 | 0.4390 | 0.5256 |

## Part2 Boston-Housing

### 5. 基线

| Baseline | MSE | MAE | R2 |
| :--- | :---: | :---: | :---: |
| DCN | 12.31 | 2.46 | 0.85 |
| GradientBoostingClassifier (n_estimators=200) | 11.26 | 2.21 | 0.87 |