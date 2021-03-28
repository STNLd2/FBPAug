import surface_distance as sd


# ####################### surface Dice Score ######################################################


def surface_dice(x, y, spacing, tolerance=1.0):
    surf_dist = sd.compute_surface_distances(x, y, spacing)
    return sd.compute_surface_dice_at_tolerance(surf_dist, tolerance)
