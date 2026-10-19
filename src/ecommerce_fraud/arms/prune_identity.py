"""Arm prune_identity: final + identity_velocity without all-time ('_all') and entity-age columns.

The full block lost the graph-round screen (-0.018). Dropped: the four nuniq_card1_by_*_all columns and
card1_device_age. Kept: 1d/7d windows, the two per-uid nuniq counts, the three uid_first_* first-seen flags
(flags, not ages).
"""
ARM = {
    "name": "prune_identity",
    "description": "final + identity_velocity without _all and card1_device_age columns (13 of 18 kept)",
    "blocks": ["identity_velocity"],
    "block_cols": {"identity_velocity": [
        "identity_velocity__nuniq_card1_by_DeviceInfo_1d",
        "identity_velocity__nuniq_card1_by_DeviceInfo_7d",
        "identity_velocity__nuniq_card1_by_id_19_1d",
        "identity_velocity__nuniq_card1_by_id_19_7d",
        "identity_velocity__nuniq_card1_by_id_20_1d",
        "identity_velocity__nuniq_card1_by_id_20_7d",
        "identity_velocity__nuniq_card1_by_email_addr1_1d",
        "identity_velocity__nuniq_card1_by_email_addr1_7d",
        "identity_velocity__nuniq_DeviceInfo_by_uid",
        "identity_velocity__nuniq_id_31_by_uid",
        "identity_velocity__uid_first_DeviceInfo",
        "identity_velocity__uid_first_email",
        "identity_velocity__uid_first_addr1",
    ]},
}
