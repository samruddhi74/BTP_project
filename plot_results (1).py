import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Experiment Setup 

strategies = {
    #'3012': 'no attackers 10epochs',
    #'3013': 'no attackers 10epochs method1' ,
    # '3004': 'no attackers',
    # '3007': '3 attackers',
    # '3011': '3 attackers, trimmed_mean aggregator'
    '2001': 'no attackers 50',
    #'2002': '5 attackers no defense 50',
    '2003': '15 attackers no defense 50',
    '2004': '15 attackers strategy: trimmed mean',
    #'2005': '15 attackers strategy: krum', 
    '2007': '15 attackers strategy: median 50' 
}

all_accuracy_data = {}
all_loss_data = {}
all_precision_data = {}
all_recall_data = {}

num_classes = 10  

# Data Extraction

for exp_id in strategies.keys():
    try:
        file_path = f'{exp_id}_results.csv'
        with open(file_path, 'r') as f:
            lines = f.readlines()

        accuracy_list = []
        loss_list = []
        precision_list = []  # store per-epoch avg precision
        recall_list = []     # store per-epoch avg recall

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 2 + 2 * num_classes:
                acc = float(parts[0])
                loss = float(parts[1])
                precisions = [float(p) for p in parts[2:2 + num_classes]]
                recalls = [float(r) for r in parts[2 + num_classes:2 + 2 * num_classes]]

                accuracy_list.append(acc)
                loss_list.append(loss)
                precision_list.append(np.mean(precisions))
                recall_list.append(np.mean(recalls))

        all_accuracy_data[exp_id] = accuracy_list
        all_loss_data[exp_id] = loss_list
        all_precision_data[exp_id] = precision_list
        all_recall_data[exp_id] = recall_list

    except FileNotFoundError:
        print(f"File {file_path} not found.")

# Accuracy & Loss Plots

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(14, 6))

# Accuracy
plt.subplot(1, 2, 1)
for exp_id, accuracy_list in all_accuracy_data.items():
    plt.plot(range(len(accuracy_list)), accuracy_list, label=strategies[exp_id])
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
for exp_id, loss_list in all_loss_data.items():
    plt.plot(range(len(loss_list)), loss_list, label=strategies[exp_id])
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Precision & Recall Plot (averaged)

plt.figure(figsize=(12, 5))

for exp_id in strategies.keys():
    plt.plot(all_precision_data[exp_id], label=f"{strategies[exp_id]} - Precision", linestyle='--')
    plt.plot(all_recall_data[exp_id], label=f"{strategies[exp_id]} - Recall")

plt.title("Average Precision and Recall per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# F1-score per epoch

plt.figure(figsize=(12, 5))

for exp_id in strategies.keys():
    p = np.array(all_precision_data[exp_id])
    r = np.array(all_recall_data[exp_id])
    f1 = np.where((p + r) > 0, 2 * (p * r) / (p + r), 0)
    plt.plot(f1, label=f"{strategies[exp_id]} - F1-score")

plt.title("Average F1-score per Epoch")
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
