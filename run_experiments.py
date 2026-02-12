import torch
from itertools import combinations
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from train import train
from config import settings as setting
from g2g.gcn import GCN
from evaluate import evaluate
from utils.utils import compute_class_weights
from loss.imbalance_loss import ImbalanceLoss
import numpy as np

# -----------------------------
# Experiment configuration
# -----------------------------
n_runs = 5  # تعداد تکرار برای هر experiment
dataset_name = "MUTAG"
batch_size = 32
num_epochs = 50

# -----------------------------
# Load dataset
# -----------------------------
dataset = TUDataset(root=f"data/TUDataset/{dataset_name}", name=dataset_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Define experiment sets
# -----------------------------
baseline_experiments = [
    "baseline",
    "baseline+imbalance",
    "baseline+imbalance+focal"
]

aug_ops = ["NodeDrop", "EdgePerturb", "FeatureMask"]
aug_combos = []
for r in range(1, len(aug_ops)+1):
    aug_combos.extend(combinations(aug_ops, r))

g2g_combos = ["baseline+G2G"] + ["G2G+" + '+'.join(combo) for combo in aug_combos]

# All experiments
all_experiments = baseline_experiments + \
                  ["Aug-" + '+'.join(combo) for combo in aug_combos] + \
                  g2g_combos

# -----------------------------
# Results dictionary
# -----------------------------
results = {exp: {"Acc": [], "F1": []} for exp in all_experiments}

# -----------------------------
# Run experiments
# -----------------------------
for exp_name in all_experiments:
    print(f"\n===== Running experiment: {exp_name} =====")

    for run in range(1, n_runs+1):
        # Reset settings
        setting.AUG_NODE_DROP = False
        setting.AUG_EDGE_PERTURB = False
        setting.AUG_FEATURE_MASK = False
        setting.RUN_G2G = False

        # Enable augmentation if experiment name contains it
        if "Aug-" in exp_name:
            setting.RUN_AUGMENTATION = True
            setting.AUG_NODE_DROP = "NodeDrop" in exp_name
            setting.AUG_EDGE_PERTURB = "EdgePerturb" in exp_name
            setting.AUG_FEATURE_MASK = "FeatureMask" in exp_name
        elif "G2G" in exp_name:
            setting.RUN_G2G = True
            setting.RUN_AUGMENTATION = any(op in exp_name for op in ["NodeDrop","EdgePerturb","FeatureMask"])
            setting.AUG_NODE_DROP = "NodeDrop" in exp_name
            setting.AUG_EDGE_PERTURB = "EdgePerturb" in exp_name
            setting.AUG_FEATURE_MASK = "FeatureMask" in exp_name
        else:
            setting.RUN_AUGMENTATION = False

        # DataLoader
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(dataset, batch_size=batch_size)
        test_loader  = DataLoader(dataset, batch_size=batch_size)

        # Model
        model = GCN(
            in_channels=dataset.num_node_features,
            hidden_channels=64,
            out_channels=dataset.num_classes
        ).to(device)

        # Loss
        if "imbalance" in exp_name:
            class_weights = compute_class_weights(dataset).to(device)
            criterion = ImbalanceLoss(
                weight=class_weights,
                use_focal="focal" in exp_name,
                gamma=setting.FOCAL_GAMMA
            )
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # Train
        best_state, best_f1 = train(
            model, train_loader, optimizer=torch.optim.Adam(model.parameters(), lr=0.005),
            criterion=criterion, device=device, val_loader=val_loader, num_epochs=num_epochs
        )
        model.load_state_dict(best_state)

        # Evaluate
        acc, f1 = evaluate(model, test_loader, device)
        results[exp_name]["Acc"].append(acc)
        results[exp_name]["F1"].append(f1)
        print(f"Run {run}/{n_runs} | Acc: {acc:.4f}, F1: {f1:.4f}")

# -----------------------------
# Compute mean and std
# -----------------------------
print("\n===== FINAL TABLE (mean ± std) =====")
for exp, metrics in results.items():
    acc_mean = np.mean(metrics["Acc"])
    acc_std = np.std(metrics["Acc"])
    f1_mean = np.mean(metrics["F1"])
    f1_std = np.std(metrics["F1"])
    print(f"{exp:40s} → Acc: {acc_mean:.4f} ± {acc_std:.4f}, F1: {f1_mean:.4f} ± {f1_std:.4f}")
