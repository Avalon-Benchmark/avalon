# TODO josh will clean this up
class IdGenerator:
    def __init__(self):
        self.next_id = 0

    def get_next_id(self) -> int:
        result = self.next_id
        self.next_id += 1
        return result


# import math
# from random import Random
# from typing import List
# from typing import Tuple
#
# import numpy as np
# from scipy import stats
#
# from common.utils import only
# from datagen.world_creation.godot_scene import GodotScene
# from datagen.world_creation.heightmap import BiomeConfig
# from datagen.world_creation.heightmap import ChasmParams
# from datagen.world_creation.heightmap import make_biome
# from datagen.world_creation.heightmap import signed_line_distance
# from datagen.world_creation.items import STICK_LENGTH
# from datagen.world_creation.items import Food
# from datagen.world_creation.items import Item
# from datagen.world_creation.items import Pillar
# from datagen.world_creation.items import Rock
# from datagen.world_creation.items import SpawnPoint
# from datagen.world_creation.items import Stick
# from datagen.world_creation.utils import IS_DEBUG_VIS
# from datagen.world_creation.utils import plot_points
# from datagen.world_creation.world import AGENT_HEIGHT
# from datagen.world_creation.world import OutdoorSmallWorld
# from datagen.world_creation.world import build_outdoor_world_map
# from datagen.world_creation.world import create_small_world
# from datagen.world_creation.world_config import OutdoorSmallWorldConfig

# def generate_move_task(rand: Random, difficulty: float, config: OutdoorSmallWorldConfig):
#     return _generate_simple_move_task(rand, difficulty, config, 0.0, is_stick_present=False, is_rock_present=False)
#
#
# def generate_jump_up_task(rand: Random, difficulty: float, config: OutdoorSmallWorldConfig):
#     pillar_height = _get_difficulty_based_value(
#         difficulty, STANDING_REACH_HEIGHT + 0.1, STANDING_REACH_HEIGHT + MAX_JUMP_HEIGHT_METERS, 0.2, rand
#     )
#     return _generate_simple_move_task(
#         rand, difficulty, config, pillar_height, is_stick_present=False, is_rock_present=False
#     )
#
#
# def generate_use_tool_on_food_task(rand: Random, difficulty: float, config: OutdoorSmallWorldConfig):
#     pillar_height = _get_difficulty_based_value(
#         difficulty,
#         STANDING_REACH_HEIGHT + MAX_JUMP_HEIGHT_METERS,
#         STANDING_REACH_HEIGHT + MAX_JUMP_HEIGHT_METERS + STICK_LENGTH,
#         0.2,
#         rand,
#     )
#     return _generate_simple_move_task(
#         rand, difficulty, config, pillar_height, is_stick_present=True, is_rock_present=False
#     )
#
#
# def generate_throw_at_food_task(rand: Random, difficulty: float, config: OutdoorSmallWorldConfig):
#     pillar_height = _get_difficulty_based_value(
#         difficulty,
#         STANDING_REACH_HEIGHT + MAX_JUMP_HEIGHT_METERS,
#         STANDING_REACH_HEIGHT + MAX_JUMP_HEIGHT_METERS + STICK_LENGTH,
#         0.2,
#         rand,
#     )
#     return _generate_simple_move_task(
#         rand, difficulty, config, pillar_height, is_stick_present=False, is_rock_present=True
#     )
#
#
# biome_config = BiomeConfig(
#     min_elevation=0.0,
#     max_elevation=0.2,
#     min_dryness=0.3,
#     max_dryness=1.0,
#     water_cutoff=0.9,
#     noise_scale=0.01,
#     noise_magnitude=1.0,
#     # TODO: yes, this is broken. Copy paste from env creation notebook
#     low_slope_threshold=1,
#     high_slope_threshold=4,
#     color_map={},
#     biome_matrix=None,
#     biomes=tuple(),
# )
#
#
# def generate_jump_forward_task(rand: Random, difficulty: float, config: OutdoorSmallWorldConfig):
#     width = _get_difficulty_based_value(difficulty, 0.1, MAX_FLAT_JUMP_METERS * 0.5, 0.2, rand)
#     depth = _get_difficulty_based_value(difficulty, 0.2, 5.0, 0.2, rand)
#     height_map = build_outdoor_world_map(config)
#     np_rand = np.random.default_rng(config.seed + 2)
#     biome_map = make_biome(height_map, biome_config, np_rand)
#     world = OutdoorSmallWorld(config, biome_map, tuple())
#     chasm_params = _create_chasm(world, rand, width, depth)
#
#     # chasm_params = attr.evolve(chasm_params, extra_unclimbable_width=5.0, width=4.0, depth=8.0)
#     # chasm_params = attr.evolve(chasm_params, width=4.0, depth=8.0)
#
#     return _generate_chasm_task(rand, difficulty, world, chasm_params)
#
#
# def generate_climb_two_handed_task(rand: Random, difficulty: float, config: OutdoorSmallWorldConfig):
#     width = _get_difficulty_based_value(difficulty, MAX_FLAT_JUMP_METERS * 0.5, MAX_FLAT_JUMP_METERS * 2, 0.2, rand)
#     depth = _get_difficulty_based_value(difficulty, MAX_JUMP_HEIGHT_METERS, 10 * MAX_JUMP_HEIGHT_METERS, 0.2, rand)
#     height_map = build_outdoor_world_map(config)
#     np_rand = np.random.default_rng(config.seed + 2)
#     biome_map = make_biome(height_map, biome_config, np_rand)
#     world = OutdoorSmallWorld(config, biome_map, tuple())
#     chasm_params = _create_chasm(world, rand, width, depth)
#     return _generate_chasm_task(rand, difficulty, world, chasm_params)
#
#
# def _generate_simple_move_task(
#     rand: Random,
#     difficulty: float,
#     config: OutdoorSmallWorldConfig,
#     pillar_height: float,
#     is_stick_present: bool,
#     is_rock_present: bool,
# ):
#     # create the level geometry
#     world = create_small_world(rand, config)
#
#     item_id_gen = ItemIdGenerator()
#
#     # decide where to place the food
#     spawn_location = world.get_spawn_location()
#     # TODO: we eventually want to weight these towards flatter locations as well, otherwise may slide out of view
#     # TODO: we also want to skew away from the edge of the world, because then the food could fall off!
#     height_to_check = FOOD_HOVER_DIST + pillar_height
#     visible_locations = world.get_locations_visible_from_point(spawn_location, height_to_check)
#     desired_distance = _get_desired_distance_for_move_task_difficulty(difficulty, config.size_in_meters)
#     target_location_distribution = stats.norm(desired_distance, 0.5)
#     location_weights = [
#         target_location_distribution.pdf(np.linalg.norm(x - spawn_location)) for x in visible_locations
#     ]
#     target_location = only(rand.choices(visible_locations, location_weights))
#
#     base_target_location = target_location - np.array([0.0, height_to_check, 0.0])
#
#     items: List[Item] = []
#     if pillar_height > 0:
#         pillar_size = np.array([1.0, pillar_height, 1.0])
#         pillar_location = base_target_location + np.array([0.0, pillar_height / 2.0, 0.0])
#         pillar = Pillar(item_id_gen.get_next_id(), position=pillar_location, size=pillar_size)
#         items.append(pillar)
#
#         # TODO: fix this
#         # also adjust the food down just a little bit so it doesnt bounce
#         target_location = target_location - np.array([0.0, 0.1, 0.0])
#
#     if is_rock_present:
#         # TODO: technically, this should be a different set of visible locations because the item isn't as tall
#         rock = Rock(
#             item_id_gen.get_next_id(),
#             _get_random_position_along_path(visible_locations, spawn_location, base_target_location, difficulty, rand),
#         )
#         items.append(rock)
#
#     if is_stick_present:
#         stick = Stick(
#             item_id_gen.get_next_id(),
#             _get_random_position_along_path(visible_locations, spawn_location, base_target_location, difficulty, rand)
#             + np.array([0.0, STICK_LENGTH / 2.0 + 0.5, 0.0]),
#         )
#         items.append(stick)
#
#     # create the godot scene
#     important_items: List[Item] = [
#         _get_spawn(item_id_gen, rand, difficulty, spawn_location, target_location),
#         Food(item_id_gen.get_next_id(), target_location),
#     ]
#     scene = GodotScene(
#         world=world,
#         items=tuple(items + important_items),
#     )
#     return scene
#
#
# def _generate_chasm_task(
#     rand: Random, difficulty: float, world: OutdoorSmallWorld, chasm_params: ChasmParams
# ) -> GodotScene:
#
#     world = world.add_chasm(chasm_params)
#
#     item_id_gen = ItemIdGenerator()
#
#     spawn_location = world.get_spawn_location()
#     visible_locations = world.get_locations_visible_from_point(spawn_location, FOOD_HOVER_DIST)
#     near_locations, chasm_locations, far_locations = _separate_chasm_locations(
#         spawn_location, visible_locations, chasm_params
#     )
#     low_weight = 0.000001
#     location_weights = np.concatenate(
#         [
#             low_weight * np.ones_like(near_locations[:, 0]),
#             low_weight * np.ones_like(chasm_locations[:, 0]),
#             np.ones_like(far_locations[:, 0]),
#         ]
#     )
#     target_location = only(
#         rand.choices(np.concatenate([near_locations, chasm_locations, far_locations]), location_weights)
#     )
#     if IS_DEBUG_VIS:
#         print(f"Placing food across the chasm, at: {target_location}")
#         plot_points(near_locations, 0, 2)
#         plot_points(far_locations, 0, 2)
#
#     # create the godot scene
#     important_items: List[Item] = [
#         _get_spawn(item_id_gen, rand, difficulty, spawn_location, target_location),
#         Food(item_id_gen.get_next_id(), target_location),
#     ]
#
#     scene = GodotScene(
#         world=world,
#         items=tuple(important_items),
#     )
#     return scene
#
#
# def _get_spawn(
#     item_id_gen: ItemIdGenerator,
#     rand: Random,
#     difficulty: float,
#     spawn_location: np.ndarray,
#     target_location: np.ndarray,
# ):
#     direction_to_target = _normalized(target_location - spawn_location)
#     # noinspection PyTypeChecker
#     target_yaw = np.angle(complex(direction_to_target[2], direction_to_target[0]), deg=True) + 180
#     spawn_view_yaw = _get_angle_value_for_difficulty(target_yaw, difficulty, rand)
#     # don't start with super weird pitches
#     # noinspection PyTypeChecker
#     target_pitch = np.angle(complex(direction_to_target[2], direction_to_target[1]), deg=True) + 180
#     spawn_view_pitch = _get_angle_value_for_difficulty(target_pitch, difficulty, rand)
#     if math.fabs(spawn_view_pitch) > 70:
#         spawn_view_pitch = 0.0
#     return SpawnPoint(
#         item_id_gen.get_next_id(),
#         position=spawn_location,
#         yaw=spawn_view_yaw,
#         pitch=spawn_view_pitch,
#     )
#
#
# def _get_angle_value_for_difficulty(target_yaw, difficulty, rand):
#     possible_yaws = range(0, 360, 10)
#     difficulty_degrees = difficulty * 180
#     difficulty_adjusted_target_yaw = round(target_yaw + rand.choice([1, -1]) * difficulty_degrees) % 360
#     yaw_distribution = stats.norm(difficulty_adjusted_target_yaw, 5)
#     angle_weights = [yaw_distribution.pdf(x) for x in possible_yaws]
#     spawn_view_yaw = only(rand.choices(possible_yaws, angle_weights))
#     return spawn_view_yaw
#
#
# def _get_random_position_along_path(
#     visible_locations: np.ndarray, start: np.ndarray, end: np.ndarray, difficulty: float, rand: Random
# ) -> np.ndarray:
#     """Difficulty scales with how far away the point is from the straight line path between start and end"""
#     path_length = np.linalg.norm(start - end)
#     desired_distance = rand.random() * difficulty * path_length * 2
#     target_location_distribution = stats.norm(desired_distance, 0.5)
#     start_point = (start[0], start[2])
#     end_point = (end[0], end[2])
#     location_weights = [
#         target_location_distribution.pdf(signed_line_distance((x[0], x[2]), start_point, end_point, path_length))
#         for x in visible_locations
#     ]
#     return only(rand.choices(visible_locations, location_weights))
#
#
# def _create_chasm(world: OutdoorSmallWorld, rand: Random, width: float, depth: float) -> ChasmParams:
#     spawn_point = world.get_spawn_location()
#     # TODO: this is a silly way of doing this--just choose random points until the spawn is not in the chasm
#     while True:
#         start = _get_random_point_in_world(world, rand)
#         end = _get_random_point_in_world(world, rand)
#         chasm_params = ChasmParams(
#             start=start,
#             end=end,
#             width=width,
#             depth=depth,
#         )
#         spawn_dist = signed_line_distance(
#             (spawn_point[0], spawn_point[2]), chasm_params.start, chasm_params.end, chasm_params.get_start_end_dist()
#         )
#         if spawn_dist > chasm_params.width:
#             return chasm_params
#
#
# def _separate_chasm_locations(
#     spawn_point: np.ndarray, locations: np.ndarray, chasm_params: ChasmParams
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     positive_side = []
#     chasm = []
#     negative_side = []
#     start_end_dist = chasm_params.get_start_end_dist()
#     for location in locations:
#         dist = signed_line_distance((location[0], location[2]), chasm_params.start, chasm_params.end, start_end_dist)
#         if math.fabs(dist) <= chasm_params.width:
#             chasm.append(location)
#         elif dist > 0.0:
#             positive_side.append(location)
#         else:
#             negative_side.append(location)
#     spawn_dist = signed_line_distance(
#         (spawn_point[0], spawn_point[2]), chasm_params.start, chasm_params.end, start_end_dist
#     )
#     if spawn_dist > 0.0:
#         near = positive_side
#         far = negative_side
#     else:
#         far = positive_side
#         near = negative_side
#     return np.array(near), np.array(chasm), np.array(far)
#
#
# def _get_random_point_in_world(world: OutdoorSmallWorld, rand: Random) -> Tuple[float, float]:
#     x_range = world.biome_map.map.region.x
#     z_range = world.biome_map.map.region.z
#     x = rand.random() * x_range.size + x_range.min_ge
#     z = rand.random() * z_range.size + z_range.min_ge
#     return (x, z)
#
#
# def _get_difficulty_based_value(
#     difficulty: float, min_val: float, max_val: float, variability: float, rand: Random
# ) -> float:
#     total_delta = max_val - min_val
#     delta = variability * total_delta
#     remainder = total_delta - delta
#     return min_val + (remainder * difficulty) + (rand.random() * delta)
#
#
# def _normalized(x: np.ndarray) -> np.ndarray:
#     return x / np.linalg.norm(x)
#
#
# def _get_desired_distance_for_move_task_difficulty(difficulty: float, size_in_meters: float) -> float:
#     max_dist = size_in_meters * 0.8  # just so we don't fall off of the edge
#     return difficulty * max_dist
