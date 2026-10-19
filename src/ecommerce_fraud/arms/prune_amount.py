"""Arm prune_amount: final + only the uid_same_amt_prod columns of amount_patterns.

The full block lost the graph-round screen (-0.039); only the repeat-same-amount-per-uid counts are kept.
"""
ARM = {
    "name": "prune_amount",
    "description": "final + amount_patterns pruned to uid_same_amt_prod_1d/_7d",
    "blocks": ["amount_patterns"],
    "block_cols": {"amount_patterns": [
        "amount_patterns__uid_same_amt_prod_1d",
        "amount_patterns__uid_same_amt_prod_7d",
    ]},
}
