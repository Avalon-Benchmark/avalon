import godot_parser
import numpy as np
from godot_parser import GDObject
from godot_parser import GDSubResourceSection
from godot_parser import Node as GDNode
from scipy.spatial.transform import Rotation

from datagen.world_creation.geometry import Plane
from datagen.world_creation.heightmap import HeightMap
from datagen.world_creation.items import ImprovedGodotScene
from datagen.world_creation.region import FloatRange
from datagen.world_creation.region import Region
from datagen.world_creation.world_config import WorldConfig


def build_outdoor_world_map(config: WorldConfig, is_debug_graph_printing_enabled: bool = False) -> HeightMap:
    """
    Fractal generation of HeightMap's

    Does the same operations (add noise, raise interior random parts of the terrain, upsample) repeatedly.

    At the end, does some smoothing and rescales to the proper height.
    """

    # create a flat heightmap of the correct size
    region = Region(
        FloatRange(-config.size_in_meters / 2.0, config.size_in_meters / 2.0),
        FloatRange(-config.size_in_meters / 2.0, config.size_in_meters / 2.0),
    )
    cells_per_meter = config.initial_point_count / config.size_in_meters
    map = HeightMap.create(region, cells_per_meter)

    rand = np.random.default_rng(config.seed)

    # set initial scale parameters
    noise = config.size_in_meters * config.initial_noise_scale
    mountain_radius = config.mountain_radius

    # fractal iteration--keep doing the same thing, but reducing the scale of the operations
    for i in range(config.fractal_iteration_count):

        # add some noise to the existing points
        map.add_noise(noise, rand)
        if is_debug_graph_printing_enabled:
            map.plot()

        # if we're adding mountains, add them
        if config.mountain_noise_count:
            for j in range(config.mountain_noise_count):
                if config.is_mountain_placement_normal_distribution:
                    x_remaining_space = (map.region.x.size / 2.0) - config.mountain_offset
                    z_remaining_space = (map.region.x.size / 2.0) - config.mountain_offset
                    mountain_center = (
                        rand.normal(map.region.x.midpoint, x_remaining_space / 2.0),
                        rand.normal(map.region.z.midpoint, z_remaining_space / 2.0),
                    )

                else:
                    offset = map.region.x.size * config.mountain_offset
                    mountain_center = (
                        rand.uniform(map.region.x.min_ge + offset, map.region.x.max_lt - offset),
                        rand.uniform(map.region.z.min_ge + offset, map.region.z.max_lt - offset),
                    )
                map.add_center_biased_noise(
                    mountain_center, noise * config.mountain_noise_scale, mountain_radius, rand
                )
            if is_debug_graph_printing_enabled:
                map.plot()

        # upsample the grid (make it finer resolution so the next pass works at a smaller scale)
        map = map.upsample()
        if is_debug_graph_printing_enabled:
            map.plot()

        # decrease the scale parameters
        mountain_radius *= config.mountain_radius_decay
        noise *= config.noise_scale_decay

    map.blur(config.blur_meters_std_dev)

    if is_debug_graph_printing_enabled:
        map.plot()

    # first do normal fade with configuration as a circle
    map.sink_edges(
        rand, config.is_fade_circular, config.fade_fraction, config.fade_noise_scale, config.fade_noise_min, 5.0
    )
    # last ditch fade out the rest
    map.sink_edges(rand, False, config.fade_fraction / 10.0, config.fade_noise_scale, 0.9)

    # scale our heights to be what was requested
    max_altitude = map.Z.max(initial=0.1)
    map.Z *= config.final_max_altitude_meters / max_altitude

    # # this is a bit of a hack. We force the edges of the world down so they're guaranteed to be under water
    # # this mostly doesn't matter because the above, more elegant ways of putting the edges of the world underwater usually do the trick
    # # but this makes absolutely certain that every edge is underwater
    # final_edge_height = -1.0
    # cells_over_which_to_fade = round((map.X.shape[0]) * config.final_world_edge_fade_fraction) + 1
    # map.lower_edges(final_edge_height, cells_over_which_to_fade)

    # force the edges of the world to be underwater, no mater what:
    min_water_border_cells = 4
    map.Z[:, 0:min_water_border_cells] = np.clip(map.Z[:, 0:min_water_border_cells], None, -1.0)
    map.Z[:, -min_water_border_cells:] = np.clip(map.Z[:, -min_water_border_cells:], None, -1.0)
    map.Z[0:min_water_border_cells, :] = np.clip(map.Z[0:min_water_border_cells, :], None, -1.0)
    map.Z[-min_water_border_cells:, :] = np.clip(map.Z[-min_water_border_cells:, :], None, -1.0)

    if is_debug_graph_printing_enabled:
        map.plot()

    return map


def add_plane_to_scene(
    plane: Plane,
    scene: ImprovedGodotScene,
    parent_node: GDNode,
    material: GDSubResourceSection,
    scaling_factor: float = 1,
) -> None:
    plane_rotation = Rotation.from_euler("xyz", [plane.pitch, plane.yaw, plane.roll], degrees=True)
    plane_rotation = plane_rotation.as_matrix().flatten()
    plane_mesh = scene.add_sub_resource(
        "PlaneMesh", size=godot_parser.Vector2(plane.width * scaling_factor, plane.length * scaling_factor)
    )
    parent_node.add_child(
        GDNode(
            f"Ramp{plane_mesh.reference.id}",
            type="MeshInstance",
            properties={
                "transform": GDObject(
                    "Transform",
                    *plane_rotation,
                    plane.x * scaling_factor,
                    plane.y * scaling_factor,
                    plane.z * scaling_factor,
                ),
                "mesh": plane_mesh.reference,
                "material/0": material.reference,
            },
        )
    )
