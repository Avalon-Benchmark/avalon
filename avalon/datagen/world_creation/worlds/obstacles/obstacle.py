from typing import Tuple

import numpy as np

from avalon.datagen.world_creation.debug_plots import plot_value_grid
from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.worlds.obstacles.harmonics import create_harmonics
from avalon.datagen.world_creation.worlds.obstacles.ring_obstacle import HeightObstacle
from avalon.datagen.world_creation.worlds.obstacles.ring_obstacle import RingObstacle


def create_obstacle_masks(
    rand: np.random.Generator,
    ring: RingObstacle,
    config: HeightObstacle,
    island_mask: MapBoolNP,
    # whether debugging is enabled
    is_debug_graph_printing_enabled: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The high level purpose of this function is to create obstacles (chasms, cliffs, and ridges).

    Using a chasm as a concrete example, we want to be able to finely control the exact distance
    that an agent would need to jump in order to cross, in addition to what fraction of the chasm
    is crossable at that width.

    Of course, we could simply draw a line or a circle for the chasm which is exactly the right size,
    but this looks extremely unnatural.

    The purpose of this function is to make it easy to create more natural looking chasms.

    At a high level, it works as follows:
    - Use spherical harmonic noise to make a wiggly circle that represents the bottom middle of the chasm
    - on either side, add purely additive spherical harmonic noise in order to vary the sizes of the chasm
        With respect to theta, this noise is varied smoothly such that within traversal_width / 2.0 of the
        traversal point, there is no noise, and at (traversal_width / 2.0) *  traversal_noise_interpolation_multiple
        the noise is the maximum value (which is defined by the safety_radius of the inner and outer rings)
    - linearly interpolate the transitions between the bottom and top of the chasm over edge_distance meters (roughly)
    """

    if config.is_inside_ring:
        assert config.traversal_theta_offset == 0.0, "Not allowed to vary theta inside the ring"

    # figure out where the traversal point is in polar coordinates
    traversal_offset = ring.config.traversal_point - ring.config.center_point
    traversal_theta = np.arctan2(traversal_offset[1], traversal_offset[0]) + config.traversal_theta_offset

    # figure out the interpolation bounds for our traversal
    # they are represented in "normalized theta delta" space, eg, 0.5 means "90 degrees away from the traversal
    # angle"
    traversal_distance = np.sqrt(np.sum((ring.config.center_point - ring.config.traversal_point) ** 2))
    if config.is_inside_ring:
        distance_delta = -config.traversal_length
    else:
        distance_delta = config.traversal_length + ring.config.chasm_bottom_size
    traversal_width_theta = np.arctan2(config.traversal_width / 2.0, traversal_distance + distance_delta)
    theta_width_interp_start = traversal_width_theta / np.pi
    theta_width_interp_end = min(config.traversal_noise_interpolation_multiple * theta_width_interp_start, 1.0)
    theta_interp_width = theta_width_interp_end - theta_width_interp_start

    # create the normalized theta space
    normalized_theta_delta = ring.theta.copy()
    normalized_theta_delta -= traversal_theta
    normalized_theta_delta[normalized_theta_delta < -np.pi] += 2 * np.pi
    normalized_theta_delta[normalized_theta_delta > np.pi] -= 2 * np.pi
    normalized_theta_delta[normalized_theta_delta > 0] *= -1.0
    normalized_theta_delta = (normalized_theta_delta / np.pi) + 1

    # create the mask (0 - 1) that can perform the actual interpolation
    theta_mask = normalized_theta_delta.copy()
    theta_mask[normalized_theta_delta >= theta_width_interp_end] = 1.0
    theta_mask[normalized_theta_delta < theta_width_interp_end] = 0.0
    theta_region = np.logical_and(
        normalized_theta_delta >= theta_width_interp_start, normalized_theta_delta < theta_width_interp_end
    )
    theta_mask[theta_region] = (normalized_theta_delta[theta_region] - theta_width_interp_start) / theta_interp_width

    # create the output obstacle mask (which we will update below)
    # 0.0 will represent the region that we start in, and 1.0 the region of the obstacle
    obstacle_mask = ring.z.copy()

    # TODO: actually tie this constant to the other
    # this is intimately related to how many grid points per meter there are
    MARGIN = 0.25

    if config.is_inside_ring:
        # figure out the maximum z that could be added without exceeding the safety radius
        safety_radius = ring.config.inner_safety_radius
        outline = np.bitwise_and(safety_radius + MARGIN > ring.r, ring.r > safety_radius - MARGIN)
        if is_debug_graph_printing_enabled:
            plot_value_grid(outline, title="Inner safety radius")
        z_at_safety_radius = ring.z[outline].max()
        max_z_to_add = (ring.mid_z - config.traversal_length) - z_at_safety_radius
        if max_z_to_add <= 0.0:
            max_z_to_add = 0.01

        # calculate the noised up inner edge and bottom cutoffs
        harmonics = (
            create_harmonics(rand, ring.theta, config.edge_config.to_harmonics(traversal_distance), is_normalized=True)
            * max_z_to_add
        )
        inner_theta_dependent_noise = harmonics * theta_mask
        inner_edge_z = (ring.mid_z - config.traversal_length) - inner_theta_dependent_noise
        inner_bottom_z = ring.mid_z - inner_theta_dependent_noise

        # everything less than the inner edge is the inner region, mask should be zero
        inner_ring_inner_region = ring.z < inner_edge_z
        obstacle_mask[inner_ring_inner_region] = 0.0

        if config.traversal_length > 0.01:
            # interpolate smoothly between inner_edge_z and inner_bottom_z
            interpolating_region = np.logical_and(ring.z >= inner_edge_z, ring.z < inner_bottom_z)
            obstacle_mask[interpolating_region] = ((ring.z - inner_edge_z) / (inner_bottom_z - inner_edge_z))[
                interpolating_region
            ]
            if is_debug_graph_printing_enabled:
                plot_value_grid(interpolating_region, title="Inner ring interpolation region")

        # if there is no outer ring, we're basically done--just clamp the rest of the values.
        # if outer_ring is None:
        obstacle_mask = np.clip(obstacle_mask, 0, 1)

        inner_region = inner_ring_inner_region
        outer_region = ring.z > inner_bottom_z
    else:
        # figure out the maximum z that could be added without exceeding the safety radius
        safety_radius = ring.config.outer_safety_radius
        outline = np.bitwise_and(safety_radius + MARGIN > ring.r, ring.r > safety_radius - MARGIN)
        if is_debug_graph_printing_enabled:
            plot_value_grid(outline, title="Outer safety radius")
        z_at_safety_radius = ring.z[outline].min()
        max_z_to_add = z_at_safety_radius - (ring.mid_z + config.traversal_length)
        # logger.debug(max_z_to_add)
        if max_z_to_add <= 0.0:
            max_z_to_add = 0.01

        # calculate the noised up outer edge and bottom cutoffs
        harmonics = (
            create_harmonics(rand, ring.theta, config.edge_config.to_harmonics(traversal_distance), is_normalized=True)
            * max_z_to_add
        )
        outer_theta_dependent_noise = harmonics * theta_mask
        outer_bottom_z = ring.mid_z + outer_theta_dependent_noise
        outer_edge_z = (ring.mid_z + config.traversal_length) + outer_theta_dependent_noise

        # everything greater than the outer edge is the outer region, mask should be zero
        outer_ring_outer_region = ring.z > outer_edge_z
        obstacle_mask[outer_ring_outer_region] = 0.0

        if config.traversal_length > 0.01:
            # interpolate smoothly between outer_bottom_z and outer_edge_z
            interpolating_region = np.logical_and(ring.z >= outer_bottom_z, ring.z < outer_edge_z)
            obstacle_mask[interpolating_region] = ((outer_edge_z - ring.z) / (outer_edge_z - outer_bottom_z))[
                interpolating_region
            ]
            if is_debug_graph_printing_enabled:
                plot_value_grid(interpolating_region, title="Outer ring interpolation region")

        # if there is no inner ring, we're basically done
        # if inner_ring is None:
        # we just set anything in the inner region to one as well (since we're going to invert below)
        obstacle_mask[ring.z < ring.mid_z] = 1.0
        # invert and clip
        obstacle_mask = 1.0 - np.clip(obstacle_mask, 0, 1)

        inner_region = ring.z < outer_bottom_z
        outer_region = outer_ring_outer_region

    # # if both the inner and outer rings are define, make sure we set all of the values in the bottom of the chasm
    # # to 1.0
    # if inner_ring and outer_ring:
    #     chasm_bottom = np.logical_and(ring.z >= inner_bottom_z, ring.z < outer_bottom_z)
    #     obstacle_mask[chasm_bottom] = 1.0
    #     if is_debug_graph_printing_enabled:
    #         plot_value_grid(chasm_bottom, title="Chasm bottom")
    #     inner_region = inner_ring_inner_region
    #     outer_region = outer_ring_outer_region
    # elif inner_ring:
    #     inner_region = inner_ring_inner_region
    #     outer_region = ring.z > inner_bottom_z
    # else:
    #     assert outer_ring
    #     inner_region = ring.z < outer_bottom_z
    #     outer_region = outer_ring_outer_region

    # print the result if debugging is enabled
    if is_debug_graph_printing_enabled:
        plot_value_grid(inner_region, title="Inner region (full)")
        plot_value_grid(outer_region, title="Outer region (full)")
        plot_value_grid(obstacle_mask, title="Final mask (full)")

    not_island = np.logical_not(island_mask)
    obstacle_mask[not_island] = 0.0
    inner_region[not_island] = 0
    outer_region[not_island] = 0

    if is_debug_graph_printing_enabled:
        plot_value_grid(inner_region, title="Inner region (minus island)")
        plot_value_grid(outer_region, title="Outer region (minus island)")
        plot_value_grid(obstacle_mask, title="Final mask (minus island)")

    return obstacle_mask, inner_region, outer_region
