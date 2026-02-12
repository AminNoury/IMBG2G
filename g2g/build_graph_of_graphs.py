# g2g/build_graph_of_graphs.py

import torch
import torch.nn.functional as F

def build_g2g_edges(graph_embeddings, k=5):
    """
    graph_embeddings: [num_graphs, dim]
    connect each graph to k most similar graphs
    """
    num_graphs = graph_embeddings.size(0)
    
    sim = F.cosine_similarity(
        graph_embeddings.unsqueeze(1),
        graph_embeddings.unsqueeze(0),
        dim=-1
    )  # [N, N]

    edge_index = []

    for i in range(num_graphs):
        vals, idx = torch.topk(sim[i], k=k+1)  # +1 چون خودش هم هست
        for j in idx[1:]:  # skip self-loop
            edge_index.append([i, j.item()])

    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index
