import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol
from typing import Set
from typing import Tuple

import attr
import numpy as np
from godot_parser import GDObject
from godot_parser import Node as GDNode
from godot_parser import Vector3 as GDVector3
from scipy.spatial.transform import Rotation

from avalon.common.utils import float_to_str
from avalon.common.utils import only
from avalon.datagen.godot_base_types import FloatRange
from avalon.datagen.world_creation.configs.flora import FloraConfig
from avalon.datagen.world_creation.constants import IS_ALL_SCENERY_IN_MAIN
from avalon.datagen.world_creation.constants import UP_VECTOR
from avalon.datagen.world_creation.constants import WATER_LINE
from avalon.datagen.world_creation.debug_plots import plot_value_grid_multi_marker
from avalon.datagen.world_creation.entities.animals import Predator
from avalon.datagen.world_creation.entities.animals import Prey
from avalon.datagen.world_creation.entities.entity import Entity
from avalon.datagen.world_creation.entities.food import Food
from avalon.datagen.world_creation.entities.scenery import Scenery
from avalon.datagen.world_creation.entities.spawn_point import PLAYER_SPAWN_POINT
from avalon.datagen.world_creation.entities.spawn_point import SpawnPoint
from avalon.datagen.world_creation.entities.tools.boulder import Boulder
from avalon.datagen.world_creation.entities.tools.colored_sphere import ColoredSphere
from avalon.datagen.world_creation.entities.tools.log import Log
from avalon.datagen.world_creation.entities.tools.stone import Stone
from avalon.datagen.world_creation.entities.tools.weapons import Weapon
from avalon.datagen.world_creation.region import Region
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.biome_map import make_fast_biome
from avalon.datagen.world_creation.worlds.height_map import clamp
from avalon.datagen.world_creation.worlds.height_map import get_flora_config_by_file
from avalon.datagen.world_creation.worlds.terrain import Terrain
from avalon.datagen.world_creation.worlds.terrain import _create_static_body
from avalon.datagen.world_creation.worlds.terrain import create_multimesh_instance
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world import get_spawn


def get_world_slug(
    task_name: str, seed: float, difficulty: float, is_practice: bool = False, index: Optional[int] = None
) -> str:
    index_part = f"_{index}" if index is not None else ""
    slug = f"{task_name}__{seed}{index_part}__{float_to_str(difficulty)}"
    if is_practice:
        return f"practice__{slug}"
    return slug


# rand is required because technically we generate the biome map and terrain first, then export
# if you'd like to non-randomly export, simply call export_world_terrain below
def export_world(
    output_folder: Path,
    rand: np.random.Generator,
    world: World,
) -> None:
    # first we create the biome map
    if world.export_config.is_biome_fast:
        biome_config = attr.evolve(world.biome_config, is_scenery_added=False)
        biome_map = make_fast_biome(world.map, biome_config)
    else:
        biome_map, world = world.make_natural_biomes(rand)

    # then we use that to create the terrain
    terrain = world.generate_terrain(rand, biome_map)

    # then we can actually do the export
    export_world_terrain(output_folder, world, terrain)


class HasPosition(Protocol):
    position: np.ndarray


def export_world_terrain(
    output_folder: Path,
    world: World,
    terrain: Terrain,
):
    output_folder.mkdir(parents=True, exist_ok=True)

    if world.export_config.debug_visualization_config is not None:
        world = _create_visualization_world(output_folder, world)

    is_exported_with_absolute_paths = world.export_config.is_exported_with_absolute_paths
    biome_map = terrain.biome_map
    map = biome_map.map
    building_name_by_id = {}
    for i, building in world.building_by_id.items():
        building_name = f"building_{i}"
        building_name_by_id[i] = building_name
    start_tile: Optional[Path] = None
    assert map.region.x.size == map.region.z.size

    # safety check
    items_outside_world = [x for x in world.items if not map.region.contains_point_2d(x.point2d)]
    if len(items_outside_world) > 0:
        raise Exception("Some items were spawned outside of the world, should never happen")

    # our tiling logic is basically--only bother exporting with tiles if the world is sufficiently large and it
    # is enabled
    if world.config.size_in_meters > 100.0 and world.export_config.is_tiled:
        # this number controls how many "tiles" will be exported for this world. Serves as a simple LOD (level of
        # detail) parameter that allows us to run much larger worlds even on low-resource devices like the oculus
        # it has been set heuristically to this value which is a fine trade-off for export time vs runtime
        tile_count = 7
    else:
        tile_count = 1

    # how many tiles away to load when running. This only really applies to the case where tile_count > 0
    # we need to know this here so that we can decide which tiles get lumped into the "distant" tile, which is
    # exported at a lower level of detail
    tile_radius = 1

    x_tile_count = tile_count
    z_tile_count = tile_count
    x_tile_size = map.region.x.size / x_tile_count
    z_tile_size = map.region.x.size / z_tile_count
    x_min = map.region.x.min_ge
    z_min = map.region.z.min_ge

    items = []
    items_by_tile_ids: Dict[Tuple[int, int], List[Entity]] = {}
    trees_by_tile_ids_and_resource: Dict[Tuple[int, int], Dict[str, List[Scenery]]] = {}
    flora_by_tile_ids_and_resource: Dict[Tuple[int, int], Dict[str, List[Scenery]]] = {}
    for item in world.items:
        zx_tile_ids = _item_to_tile_ids(item, world.map.region, x_tile_count, z_tile_count)
        if isinstance(item, Scenery):
            scenery_by_tile_ids_and_resource = (
                trees_by_tile_ids_and_resource if "trees/" in item.resource_file else flora_by_tile_ids_and_resource
            )
            if zx_tile_ids not in scenery_by_tile_ids_and_resource:
                scenery_by_tile_ids_and_resource[zx_tile_ids] = {}
            resource = item.resource_file
            if resource not in scenery_by_tile_ids_and_resource[zx_tile_ids]:
                scenery_by_tile_ids_and_resource[zx_tile_ids][resource] = []
            scenery_by_tile_ids_and_resource[zx_tile_ids][resource].append(item)
        else:
            if zx_tile_ids not in items_by_tile_ids:
                items_by_tile_ids[zx_tile_ids] = []
            items_by_tile_ids[zx_tile_ids].append(item)
            items.append(item)

    if world.export_config.is_minor_scenery_hidden:
        flora_by_tile_ids_and_resource = {}

    all_tile_ids = set()
    for x_idx in range(0, x_tile_count):
        for z_idx in range(0, z_tile_count):
            all_tile_ids.add((z_idx, x_idx))

    terrain_export_results = []
    spawn_point = None
    spawn_point_names: Set[str] = set()
    for x_idx in range(0, x_tile_count):
        for z_idx in range(0, z_tile_count):
            region = Region(
                x=FloatRange(x_min + x_idx * x_tile_size, x_min + (x_idx + 1) * x_tile_size),
                z=FloatRange(z_min + z_idx * z_tile_size, z_min + (z_idx + 1) * z_tile_size),
            )
            region_items = items_by_tile_ids.get((z_idx, x_idx), [])
            region_building_names = [
                building_name_by_id[i] for i, x in world.building_by_id.items() if region.overlaps_region(x.region)
            ]
            fine_output_file = output_folder / f"tile_{x_idx}_{z_idx}.tscn"
            coarse_output_file = output_folder / f"distant_{x_idx}_{z_idx}.tscn"
            # extend the outside regions a tiny bit so that no triangles are left behind.
            adapted_region = region.epsilon_expand(
                x_idx == 0,
                x_idx == x_tile_count - 1,
                z_idx == 0,
                z_idx == z_tile_count - 1,
            )
            neighboring_and_this_tile_ids = set()
            for i in range(-tile_radius, tile_radius + 1):
                other_tile_x = x_idx + i
                if other_tile_x < 0 or other_tile_x >= x_tile_count:
                    continue
                for j in range(-tile_radius, tile_radius + 1):
                    other_tile_z = z_idx + j
                    if other_tile_z < 0 or other_tile_z >= z_tile_count:
                        continue
                    neighboring_and_this_tile_ids.add((other_tile_z, other_tile_x))
            distant_tile_ids = set([x for x in all_tile_ids if x not in neighboring_and_this_tile_ids])
            terrain_export_results.append(
                terrain.export(
                    fine_output_file,
                    coarse_output_file,
                    adapted_region,
                    x_idx,
                    z_idx,
                    tile_radius,
                    region_building_names,
                    trees_by_tile_ids_and_resource if not IS_ALL_SCENERY_IN_MAIN else {},
                    flora_by_tile_ids_and_resource if not IS_ALL_SCENERY_IN_MAIN else {},
                    distant_tile_ids,
                    neighboring_and_this_tile_ids,
                )
            )
            spawn_items = [x for x in region_items if isinstance(x, SpawnPoint)]
            for spawn in spawn_items:
                assert spawn.name not in spawn_point_names, f"SpawnPoints name {spawn.name} is duplicated"
                spawn_point_names.add(spawn.name)
                if spawn.name == PLAYER_SPAWN_POINT:
                    assert spawn_point is None, "Should only be one player spawn point"
                    spawn_point = spawn
                    start_tile = fine_output_file
    assert start_tile is not None, "No tiles contained a spawn point--that's not good!"

    starting_terrain_tile_path = (
        str(output_folder / start_tile.name) if is_exported_with_absolute_paths else "./" + start_tile.name
    )

    # export tscn file
    scene = GodotScene()
    level_resource = scene.add_ext_resource(starting_terrain_tile_path, "PackedScene")
    terrain_manager_script = scene.add_ext_resource("res://terrain/terrain_manager.gd", "Script")

    if world.export_config.is_meta_data_exported:
        meta_output_path = output_folder / "meta.json"
        with open(meta_output_path, "w") as outfile:
            # spawn_point
            assert spawn_point is not None
            food: List[HasPosition] = [x for x in items if isinstance(x, Food)]
            prey: List[HasPosition] = [x for x in items if isinstance(x, Prey)]
            food_count = len(food) + len(prey)
            if food_count == 1:
                only_food = only(food + prey)
                total_distance = np.linalg.norm(only_food.position - spawn_point.position)
            else:
                all_food = food + prey
                total_distance = np.float64(sum([np.linalg.norm(x.position - spawn_point.position) for x in all_food]))
            meta_data = dict(
                size_in_meters=world.config.size_in_meters,
                food_count=food_count,
                total_distance=total_distance,
                is_visibility_required=spawn_point.is_visibility_required,
            )
            outfile.write(json.dumps(meta_data))

    main_output_path = output_folder / "main.tscn"

    with scene.use_tree() as tree:
        tree.root = GDNode("Avalon", type="Spatial")
        if biome_map.config.export_config is None or biome_map.config.export_config.is_sun_enabled:
            tree.root.add_child(
                GDNode(
                    "Sun",
                    type="DirectionalLight",
                    properties={
                        "transform": _get_transform_from_pitch_and_yaw(
                            *_get_pitch_and_yaw_from_sky_params(biome_map.config.godot_sky_config)
                        ),
                        **biome_map.config.godot_sun_config,
                    },
                )
            )
        camera_transform = GDObject(
            "Transform", 0.5, 0.594, -0.629, 0, 0.727, 0.686, 0.866, -0.343, 0.363, -6.525, 5.377, 4.038
        )
        tree.root.add_child(
            GDNode(
                "Camera",
                type="Camera",
                properties={"transform": camera_transform},
            )
        )
        tree.root.add_child(
            _create_godot_world_environment(
                scene, biome_map.config.godot_env_config, biome_map.config.godot_sky_config
            )
        )

        if IS_ALL_SCENERY_IN_MAIN:
            scenery_by_resource: Dict[str, List[Scenery]] = {}
            for scenery_by_tile_ids_and_resource in (
                trees_by_tile_ids_and_resource,
                flora_by_tile_ids_and_resource,
            ):
                for tile_id, inner_dict in scenery_by_tile_ids_and_resource.items():
                    for resource, scenery_list in inner_dict.items():
                        if resource not in scenery_by_resource:
                            scenery_by_resource[resource] = []
                        scenery_by_resource[resource].extend(scenery_list)
                for resource, scenery_list in scenery_by_resource.items():
                    tree.root.add_child(
                        create_multimesh_instance(
                            scene,
                            "main",
                            scenery_list,
                            resource.replace(".tscn", ".res"),
                            biome_map.config.flora_config,
                        )
                    )

        data = np.logical_not(world.is_climbable)
        if data.shape[1] % 8 != 0:
            data = np.pad(data, pad_width=[(0, 0), (0, 8 - data.shape[1] % 8)])
        bytes = np.packbits(data, bitorder="little")

        boundary = world.config.size_in_meters / 2.0

        export_path = "."
        if is_exported_with_absolute_paths:
            # TODO: fix the double quotes, update S3 checksum manifest
            export_path = f'"{str(output_folder)}"'
        if world.export_config.world_id is not None:
            export_path = f"res://worlds/{world.export_config.world_id}"

        terrain_manager = GDNode(
            "TerrainManager",
            type="Node",
            properties=dict(
                script=terrain_manager_script.reference,
                export_path=export_path,
                x_tile_count=x_tile_count,
                z_tile_count=z_tile_count,
                tile_radius=tile_radius,
                x_min=-boundary,
                z_min=-boundary,
                x_max=boundary,
                z_max=boundary,
                climb_map=GDObject("PoolByteArray", *list(bytes)),
                climb_map_x=len(world.is_climbable[0, :]),
                climb_map_y=len(world.is_climbable[:, 0]),
            ),
        )
        tiles_node = GDNode("tiles", type="Node")
        tiles_node.add_child(GDNode(str(start_tile.name).split(".")[0], instance=level_resource.reference.id))
        terrain_manager.add_child(tiles_node)

        building_group = GDNode("buildings", type="Node")
        terrain_manager.add_child(building_group)
        for i, building in world.building_by_id.items():
            building.export(
                scene,
                building_group,
                building_name_by_id[i],
                is_indoor_lighting_enabled=world.export_config.is_indoor_lighting_enabled,
            )

        terrain_manager.add_child(GDNode("walls", type="Node"))
        collision_mesh_group = GDNode("terrain_collision_meshes", type="Node")

        for i, (region_triangles, region_vertices) in enumerate(terrain_export_results):
            collision_mesh_group.add_child(_create_static_body(i, scene, region_triangles, region_vertices))
        terrain_manager.add_child(collision_mesh_group)
        tree.root.add_child(terrain_manager)

        tracker_node = GDNode("dynamic_tracker", type="Node")
        next_id = max([x.entity_id for x in items]) + 1
        for i, item in enumerate(items):
            if item.entity_id < 0:
                item = attr.evolve(item, entity_id=next_id)
                next_id += 1
            tracker_node.add_child(item.get_node(scene))
        tree.root.add_child(tracker_node)

        # figure out all of the trees
        tree_by_resource: Dict[str, List[Scenery]] = {}
        for tile_id, inner_dict in trees_by_tile_ids_and_resource.items():
            for resource, scenery_list in inner_dict.items():
                if resource not in tree_by_resource:
                    tree_by_resource[resource] = []
                tree_by_resource[resource].extend(scenery_list)

        # add all collision shapes for the trees
        tree_collision_mesh_group = GDNode("tree_collision_meshes", type="Spatial")
        for i, (resource, scenery_list) in enumerate(tree_by_resource.items()):
            for j, tree_obj in enumerate(scenery_list):
                tree_collider = _create_tree_shape(scene, tree_obj, i, j, world.biome_config.flora_config, resource)
                tree_collision_mesh_group.add_child(tree_collider)
        tree.root.add_child(tree_collision_mesh_group)

        # is big enough that you dont really notice that it isn't infinite
        tree.root.add_child(_create_ocean_node(scene, "large_ocean", ocean_size=1_000, offset=-2))

        # if we ONLY make a huge ocean, it has a rendering bug near the shore
        # so we also make a small one that is slightly higher
        # UNLESS it is a completely flat world (probably indoor-only), since we'd get overlapping planes
        world_size = round(max([map.region.x.size, map.region.z.size]) * 1.2)
        if not (map.Z == 0).all():
            tree.root.add_child(_create_ocean_node(scene, "near_ocean", ocean_size=world_size, offset=-1))

        tree.root.add_child(
            _create_ocean_collision(
                scene, "ocean", position=np.array([0, -1.5, 0]), size=np.array([world_size, 1.0, world_size])
            )
        )

    scene.write(str((main_output_path).absolute()))


def _create_visualization_world(output_folder: Path, world: World) -> World:
    spawn_item = only([x for x in world.items if isinstance(x, SpawnPoint)])
    food_items: List[HasPosition] = [x for x in world.items if isinstance(x, Food)]
    if len(food_items) == 0:
        food_items = [x for x in world.items if isinstance(x, Prey)]
    spawn_position = spawn_item.position
    food_distances = [(np.linalg.norm(spawn_position - x.position), x) for x in food_items]
    food_position = sorted(food_distances)[-1][1].position
    mid_position = (spawn_position + food_position) / 2.0
    spawn_to_goal_vec = food_position - spawn_position
    new_spawn_item = None
    for off_center in (1.0, 0.5, 0.0):
        for orientation in (1.0, -1.0):
            sideways = np.array([-spawn_to_goal_vec[2], 0.0, spawn_to_goal_vec[0]]) * orientation * off_center
            full_len = np.linalg.norm(spawn_to_goal_vec)
            new_spawn = mid_position + sideways + UP_VECTOR * full_len / 2.0
            if not world.map.region.contains_point_3d(new_spawn, epsilon=0.0):
                continue
            rand = np.random.default_rng(0)
            new_spawn_item = get_spawn(rand, 0.0, new_spawn, mid_position)
            # markers = [
            #     world.map.point_to_index(to_2d_point(x))
            #     for x in [food_position, mid_position, spawn_position, new_spawn_item.position]
            # ]
            # plot_value_grid(world.map.Z, markers=markers)
            break
        if new_spawn_item is not None:
            break
    assert new_spawn_item is not None, "Failed to find a point in the region to view from"
    world_items = [x for x in world.items if not isinstance(x, SpawnPoint)]
    world_items.append(new_spawn_item)

    # replace all items with markers
    points_by_type = defaultdict(list)
    new_items = []
    collected_types = tuple([Food, Prey, Predator, Log, Weapon, Boulder, Stone])
    for item in world_items:
        if isinstance(item, collected_types):
            for dtype in collected_types:
                if isinstance(item, dtype):
                    points_by_type[dtype.__name__].append(item.position)
        else:
            new_items.append(item)
    SPAWN_COLOR = "#FF00F8"
    color_by_type_name = {
        Food.__name__: "#FF0000",
        Prey.__name__: "#FDFF00",
        Predator.__name__: "#AE00FF",
        Log.__name__: "#FF9E00",
        Weapon.__name__: "#000000",
        Boulder.__name__: "#FF0094",
        Stone.__name__: "#FF0094",
    }
    for type_name, points in points_by_type.items():
        color = color_by_type_name[type_name]
        for point in points:
            new_items.append(ColoredSphere(position=point, color=color))
    new_items.append(ColoredSphere(position=spawn_position, color=SPAWN_COLOR))
    world = attr.evolve(world, items=tuple(new_items))

    if (
        world.export_config.debug_visualization_config is not None
        and world.export_config.debug_visualization_config.is_2d_graph_drawn
    ):
        marker_lists = []
        for type_name, points in points_by_type.items():
            color = color_by_type_name[type_name]
            markers = [world.map.point_to_index(to_2d_point(x)) for x in points]
            marker_lists.append((markers, color))
        marker_lists.append(([world.map.point_to_index(to_2d_point(spawn_position))], SPAWN_COLOR))
        height_with_water = world.map.Z.copy()
        height_with_water[height_with_water < WATER_LINE] = world.map.Z.max(initial=1.0)
        tag = ""
        if world.config.is_indoor_only:
            tag = "INDOOR"
            height_with_water[:, :] = 0.5
        plot_value_grid_multi_marker(height_with_water, tag + str(output_folder), marker_lists)

    return world


def _create_ocean_node(scene: GodotScene, name: str, ocean_size: int, offset: float):
    ocean_location = np.array([0, offset, 0])
    ocean_node = GDNode(
        name,
        type="MeshInstance",
        properties={
            "transform": GDObject("Transform", 1, 0, 0, 0, 1, 0, 0, 0, 1, *ocean_location),
            "mesh": scene.add_sub_resource(
                "CubeMesh",
                material=scene.add_ext_resource("res://materials/ocean.material", "Material").reference,
                size=GDVector3(ocean_size, 2.0, ocean_size),
            ).reference,
            "material/0": None,
        },
    )
    return ocean_node


def _item_to_tile_ids(item: Entity, world_region: Region, x_tile_count: int, z_tile_count: int) -> Tuple[int, int]:
    if x_tile_count == 1:
        assert z_tile_count == 1, "lazy implementation"
        return (0, 0)
    max_x_index = x_tile_count - 1
    one_cell_x = world_region.x.size / (max_x_index)
    x = round((item.position[0] - world_region.x.min_ge) / one_cell_x)
    max_y_index = z_tile_count - 1
    one_cell_y = world_region.z.size / (max_y_index)
    y = round((item.position[2] - world_region.z.min_ge) / one_cell_y)
    return clamp(y, 0, max_y_index), clamp(x, 0, max_x_index)


def _get_pitch_and_yaw_from_sky_params(sky_config: Dict[str, Any]) -> Tuple[float, float]:
    pitch = (180.0 - sky_config.get("sun_latitude", 170.0)) * -1.0
    yaw = sky_config.get("sun_longitude", 0.0) * -1.0
    return pitch, yaw


def _get_transform_from_pitch_and_yaw(pitch: float, yaw: float) -> GDObject:
    yaw_rotation = Rotation.from_euler("y", yaw, degrees=True)
    pitch_rotation = Rotation.from_euler("x", pitch, degrees=True)
    rotation = (yaw_rotation * pitch_rotation).as_matrix().flatten()
    return GDObject("Transform", *rotation, 0, 0, 0)


def _create_godot_world_environment(
    scene: GodotScene, env_config: Dict[str, Any], sky_config: Dict[str, Any]
) -> GDNode:
    sky_resource = scene.add_sub_resource("ProceduralSky", **sky_config)
    full_env_config = {**env_config}
    full_env_config["background_sky"] = sky_resource.reference
    env_resource = scene.add_sub_resource("Environment", **full_env_config)
    return GDNode(
        "WorldEnvironment",
        type="WorldEnvironment",
        properties={
            "environment": env_resource.reference,
        },
    )


def _create_ocean_collision(scene: GodotScene, name: str, position: np.ndarray, size: np.ndarray) -> GDNode:
    transform = GDObject("Transform", 1, 0, 0, 0, 1, 0, 0, 0, 1, *position)
    root = GDNode(
        name,
        type="StaticBody",
        properties={
            "transform": transform,
        },
    )
    root.add_child(
        GDNode(
            f"{name}_collision",
            type="CollisionShape",
            properties={
                "shape": scene.add_sub_resource(
                    "BoxShape", resource_local_to_scene=True, extents=GDVector3(*size / 2.0)
                ).reference
            },
        )
    )

    return root


def _create_tree_shape(
    scene: GodotScene, tree_obj: Scenery, i: int, j: int, flora_config: Dict[str, FloraConfig], resource: str
) -> GDNode:
    config, resource_name = get_flora_config_by_file(flora_config, resource)
    assert config is not None
    size = config.collision_extents * tree_obj.scale * config.default_scale
    offset_position = tree_obj.position.copy()
    offset_position[1] += config.collision_extents[1] * tree_obj.scale[1] * config.default_scale
    transform = GDObject("Transform", *tree_obj.rotation, *offset_position)
    tree_collider = GDNode(f"tree_{i}_collision_{j}", type="StaticBody", properties={"transform": transform})
    tree_collider.add_child(
        GDNode(
            f"tree_{i}_collision_{j}_shape",
            type="CollisionShape",
            properties={
                "shape": scene.add_sub_resource(
                    "BoxShape", resource_local_to_scene=True, extents=GDVector3(*size)
                ).reference
            },
        )
    )

    return tree_collider
