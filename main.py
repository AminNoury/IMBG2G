import torch
from itertools import combinations
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from config import settings as setting
from loss.imbalance_loss import ImbalanceLoss
from utils.utils import compute_class_weights

from g2g.gcn import GCN
from train import train
from evaluate import evaluate


def run_experiment(name, dataset, device):
    print(f"\n===== Running Experiment: {name} =====")

    dataset = dataset.shuffle()
    print(f"Total dataset size: {len(dataset)} | Num features: {dataset.num_node_features} | Num classes: {dataset.num_classes}")

    train_dataset = dataset
    val_dataset   = dataset
    test_dataset  = dataset

    if setting.TEST_MODE:
        train_dataset = train_dataset[:40]
        val_dataset   = val_dataset[:10]
        test_dataset  = test_dataset[:10]
        print("TEST MODE ACTIVE: Using small dataset subset")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32)
    test_loader  = DataLoader(test_dataset, batch_size=32)

    # ------------------------
    # Model
    # ------------------------
    model = GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        out_channels=dataset.num_classes
    ).to(device)
    print(model)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters())}")

    # ------------------------
    # G2G Log
    # ------------------------
    if setting.RUN_G2G:
        print("G2G Module ENABLED")
        print(f"G2G USE KNN Graph: {setting.G2G_USE_KNN_GRAPH}")
        print(f"G2G USE Message Passing: {setting.G2G_USE_MESSAGE_PASSING}")
        print(f"G2G K: {setting.G2G_K}, G2G Hidden Dim: {setting.G2G_HIDDEN_DIM}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    print(f"Optimizer: Adam | LR: {optimizer.param_groups[0]['lr']} | Weight decay: 5e-4")

    # ------------------------
    # Loss Function
    # ------------------------
    if "imbalance" in name:
        class_weights = compute_class_weights(train_dataset).to(device)
        criterion = ImbalanceLoss(
            weight=class_weights,
            use_focal="focal" in name,
            gamma=setting.FOCAL_GAMMA
        )
        if "focal" in name:
            print(f"Using Imbalance Loss WITH Focal | Gamma: {setting.FOCAL_GAMMA}")
        else:
            print("Using Imbalance Loss WITHOUT Focal")
        print(f"Class weights: {class_weights}")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("Using Standard CrossEntropy (Baseline)")

    # ------------------------
    # Train
    # ------------------------
    best_model_state, best_f1 = train(
        model, train_loader, optimizer, criterion, device,
        val_loader=val_loader, num_epochs=50
    )

    model.load_state_dict(best_model_state)

    # ------------------------
    # Evaluate
    # ------------------------
    acc, f1 = evaluate(model, test_loader, device)
    print(f"Test Result → Acc: {acc:.4f}, F1: {f1:.4f}")
    return acc, f1


def main():
    print("=== IMBG2G Ablation Study on MUTAG ===")
    dataset = TUDataset(root="IMBG2G/data/MUTAG/MUTAG", name="MUTAG")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------
    # G2G Settings
    # ------------------------
    setting.RUN_G2G = False                 # False: off, True: on
    setting.G2G_USE_KNN_GRAPH = True
    setting.G2G_USE_MESSAGE_PASSING = True
    setting.G2G_K = 5
    setting.G2G_HIDDEN_DIM = 64

    results = {}

    # ------------------------
    # Baseline Experiments
    # ------------------------
    baseline_experiments = [
        "baseline",
        "baseline+imbalance",
        "baseline+imbalance+focal"
    ]

    for exp_name in baseline_experiments:
        setting.AUG_NODE_DROP = False
        setting.AUG_EDGE_PERTURB = False
        setting.AUG_FEATURE_MASK = False
        setting.RUN_G2G = False
        results[exp_name] = run_experiment(exp_name, dataset, device)

    # ------------------------
    # Augmentation Experiments
    # ------------------------
    aug_ops = ["NodeDrop", "EdgePerturb", "FeatureMask"]
    aug_combos = []

    # Single, double, triple combinations
    for r in range(1, len(aug_ops)+1):
        aug_combos.extend(combinations(aug_ops, r))

    for combo in aug_combos:
        setting.AUG_NODE_DROP = "NodeDrop" in combo
        setting.AUG_EDGE_PERTURB = "EdgePerturb" in combo
        setting.AUG_FEATURE_MASK = "FeatureMask" in combo
        setting.RUN_G2G = False

        exp_name = f"Aug-{'+'.join(combo)}"
        results[exp_name] = run_experiment(exp_name, dataset, device)

    # ------------------------
    # G2G Experiments
    # ------------------------
    g2g_experiments = [
        ("baseline+G2G", False),        # baseline + G2G
    ]

    # Add G2G + Augmentation combos
    for combo in aug_combos:
        g2g_experiments.append(("+".join(["G2G"] + list(combo)), True))

    for exp_name, run_g2g in g2g_experiments:
        setting.RUN_G2G = run_g2g
        setting.AUG_NODE_DROP = "NodeDrop" in exp_name
        setting.AUG_EDGE_PERTURB = "EdgePerturb" in exp_name
        setting.AUG_FEATURE_MASK = "FeatureMask" in exp_name

        results[exp_name] = run_experiment(exp_name, dataset, device)

    # ------------------------
    # Print final comparison
    # ------------------------
    print("\n===== FINAL COMPARISON =====")
    for k, v in results.items():
        print(f"{k:40s} → Acc: {v[0]:.4f}, F1: {v[1]:.4f}")


if __name__ == "__main__":
    main()
