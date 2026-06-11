from __future__ import annotations
from itertools import product


def build_cross_slugs(
    config,
    nested_dim_slugs: list[str],
    cross_dim: str,
) -> tuple[list[str], dict[str, tuple[str, str]]]:
    """Generate (nested_slug × cross_val) slugs for cross-dim spend queries.

    Takes the actual modeled variable slugs from Deep Dive results (not all values
    from specs), so filtering done during diagnostics is preserved automatically.

    Args:
        config: DeepDiveConfig with vehicle_spec and brand.
        nested_dim_slugs: columns from results.contribs[nested_dim] — raw slugs only
            (non-slug columns like __outros__ or anchor_ are skipped automatically).
        cross_dim: name of the cross dimension, e.g. "Midia".

    Returns:
        slugs: list of cross-dim slugs to pass to load_breakdown_spend.
        slug_to_pair: mapping from cross slug → (base_nested_slug, cross_value).
    """
    vspec = config.vehicle_spec
    cross_bd = vspec["breakdowns"][cross_dim]
    cross_cat = cross_bd["category"]
    cross_values = cross_bd["values"]

    base_slugs = [s for s in nested_dim_slugs if s.startswith("$")]

    slugs: list[str] = []
    slug_to_pair: dict[str, tuple[str, str]] = {}

    for base_slug, cv in product(base_slugs, cross_values):
        cross_slug = f"{base_slug}$category:{cross_cat}:{cv}"
        slugs.append(cross_slug)
        slug_to_pair[cross_slug] = (base_slug, cv)

    return slugs, slug_to_pair
