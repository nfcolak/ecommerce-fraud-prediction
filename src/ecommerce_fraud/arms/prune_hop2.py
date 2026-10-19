"""Arm prune_hop2: final + graph_hop2 without the reach_uid columns.

The full block lost the graph-round screen (-0.020). Dropped: reach_uid and its 1d/7d/30d growth variants.
Kept (7 of 11): n_hub, reach_fp, reach_card, cnt_7d, cnt_30d, amt_mean_7d, amt_mean_30d.
"""
ARM = {
    "name": "prune_hop2",
    "description": "final + graph_hop2 without reach_uid, reach_uid_1d/7d/30d (7 of 11 kept)",
    "blocks": ["graph_hop2"],
    "block_cols": {"graph_hop2": [
        "graph_hop2__n_hub", "graph_hop2__reach_fp", "graph_hop2__reach_card",
        "graph_hop2__cnt_7d", "graph_hop2__cnt_30d",
        "graph_hop2__amt_mean_7d", "graph_hop2__amt_mean_30d",
    ]},
}
