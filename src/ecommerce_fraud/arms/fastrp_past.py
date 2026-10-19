"""Arm fastrp_past: final + the past-only FastRP entity-embedding block (32 dims)."""
ARM = {
    "name": "fastrp_past",
    "description": "final + fastrp_past block (2x16 past-only random-projection entity embedding, no labels)",
    "blocks": ["fastrp_past"],
}
