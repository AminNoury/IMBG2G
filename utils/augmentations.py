import torch
from torch_geometric.utils import subgraph, remove_self_loops

# ------------------ NODE DROP (SAFE + LABELS) ------------------
from torch_geometric.utils import subgraph

def node_drop(data, drop_ratio):
    num_nodes = data.num_nodes
    keep_num = max(1, int(num_nodes * (1 - drop_ratio)))
    perm = torch.randperm(num_nodes)[:keep_num]
    perm = perm.sort()[0]

    # Node features
    data.x = data.x[perm]

    # Edge index
    edge_index, _ = subgraph(perm, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
    data.edge_index = edge_index

    # Update batch if it exists
    if hasattr(data, 'batch') and data.batch is not None:
        data.batch = data.batch[perm]

    # data.y stays untouched
    return data



# ------------------ EDGE PERTURB (REMOVE + ADD) ------------------
def edge_perturb(data, perturb_ratio):
    edge_index = data.edge_index.clone()
    num_edges = edge_index.size(1)
    num_nodes = data.num_nodes

    keep_num = int(num_edges * (1 - perturb_ratio))
    perm = torch.randperm(num_edges)[:keep_num]
    edge_index = edge_index[:, perm]

    # Add random edges
    num_add = num_edges - keep_num
    rand_edges = torch.randint(0, num_nodes, (2, num_add))
    edge_index = torch.cat([edge_index, rand_edges], dim=1)

    # Remove self-loops and duplicates
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = torch.unique(edge_index, dim=1)

    data.edge_index = edge_index
    return data


# ------------------ FEATURE MASK ------------------
def feature_mask(data, mask_ratio):
    mask = torch.rand_like(data.x) < mask_ratio
    data.x = data.x.masked_fill(mask, 0)
    return data


# ------------------ APPLY BASED ON SETTINGS ------------------
def apply_augmentations(data, setting):
    applied = []
    if setting.AUG_NODE_DROP:
        data = node_drop(data, setting.NODE_DROP_RATIO)
        applied.append("NodeDrop")
    if setting.AUG_EDGE_PERTURB:
        data = edge_perturb(data, setting.EDGE_PERTURB_RATIO)
        applied.append("EdgePerturb")
    if setting.AUG_FEATURE_MASK:
        data = feature_mask(data, setting.FEATURE_MASK_RATIO)
        applied.append("FeatureMask")
    
    print(f"Applied Augmentations: {applied}")  # برای لاگ
    return data
