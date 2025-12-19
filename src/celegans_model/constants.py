"""Constants for C. elegans worm space coordinate transformations."""

# Canonical positions for standard seam cells along the AP axis.
# Keys are lowercase seam cell names, values are their canonical integer positions (0-10).
# Used for "uniform" parameterization where standard cells get evenly spaced positions.
STANDARD_SEAM_CELLS: dict[str, int] = {
    "a0": 0,  # Anterior-most
    "h0": 1,
    "h1": 2,
    "h2": 3,
    "v1": 4,
    "v2": 5,
    "v3": 6,
    "v4": 7,
    "v5": 8,
    "v6": 9,
    "t": 10,  # Posterior-most (tail)
}
