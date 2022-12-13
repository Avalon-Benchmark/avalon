import math
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

import attr
import numpy as np
from nptyping import assert_isinstance
from scipy import ndimage
from scipy import stats
from scipy.interpolate import interp1d
from skimage import morphology

from avalon.common.utils import only
from avalon.common.utils import to_immutable_array
from avalon.contrib.serialization import Serializable
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.configs.biome import BiomeConfig
from avalon.datagen.world_creation.configs.building import BuildingConfig
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.scenery import SceneryConfig
from avalon.datagen.world_creation.configs.world import WorldConfig
from avalon.datagen.world_creation.constants import DEFAULT_SAFETY_RADIUS
from avalon.datagen.world_creation.constants import FLORA_REMOVAL_METERS
from avalon.datagen.world_creation.constants import HALF_AGENT_HEIGHT_VECTOR
from avalon.datagen.world_creation.constants import METERS_OF_TREE_CLEARANCE_AROUND_LINE_OF_SIGHT
from avalon.datagen.world_creation.constants import UP_VECTOR
from avalon.datagen.world_creation.constants import WORLD_RAISE_AMOUNT
from avalon.datagen.world_creation.debug_plots import plot_terrain
from avalon.datagen.world_creation.debug_plots import plot_value_grid
from avalon.datagen.world_creation.entities.animals import Predator
from avalon.datagen.world_creation.entities.entity import Entity
from avalon.datagen.world_creation.entities.entity import EntityType
from avalon.datagen.world_creation.entities.food import CANONICAL_FOOD
from avalon.datagen.world_creation.entities.food import CANONICAL_FOOD_CLASS
from avalon.datagen.world_creation.entities.food import Food
from avalon.datagen.world_creation.entities.food import FoodTree
from avalon.datagen.world_creation.entities.item import InstancedDynamicItem
from avalon.datagen.world_creation.entities.scenery import Scenery
from avalon.datagen.world_creation.entities.spawn_point import SpawnPoint
from avalon.datagen.world_creation.entities.tools.placeholder import Placeholder
from avalon.datagen.world_creation.entities.tools.tool import Tool
from avalon.datagen.world_creation.entities.tools.weapons import Weapon
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.indoor.components import Story
from avalon.datagen.world_creation.noise import perlin
from avalon.datagen.world_creation.types import HeightMode
from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.types import MapFloatNP
from avalon.datagen.world_creation.types import Point2DNP
from avalon.datagen.world_creation.types import Point3DListNP
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.types import SceneryBorderMode
from avalon.datagen.world_creation.utils import normalized
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.biome_map import BiomeMap
from avalon.datagen.world_creation.worlds.biome_map import make_biome
from avalon.datagen.world_creation.worlds.biome_map import plot_biome_grid
from avalon.datagen.world_creation.worlds.height_map import HeightMap
from avalon.datagen.world_creation.worlds.height_map import SpecialBiomes
from avalon.datagen.world_creation.worlds.height_map import build_outdoor_world_map
from avalon.datagen.world_creation.worlds.height_map import selected_distance_weighted_points
from avalon.datagen.world_creation.worlds.obstacles.harmonics import create_harmonics
from avalon.datagen.world_creation.worlds.obstacles.height_solution import add_detail_near_items
from avalon.datagen.world_creation.worlds.obstacles.obstacle import create_obstacle_masks
from avalon.datagen.world_creation.worlds.obstacles.ring_obstacle import RingObstacle
from avalon.datagen.world_creation.worlds.obstacles.ring_obstacle import RingObstacleConfig
from avalon.datagen.world_creation.worlds.terrain import Terrain
from avalon.datagen.world_creation.worlds.utils import signed_line_distance
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class World(Serializable):  # type: ignore[no-untyped-def]  # mypy bug?
    """Enables creation of a series of Zones separated by Obstacles on a HeightMap"""

    config: WorldConfig
    export_config: ExportConfig
    biome_config: BiomeConfig

    map: HeightMap = attr.ib(converter=lambda x: x.freeze())  # type: ignore
    items: Tuple[Entity, ...] = attr.ib()
    buildings: Tuple[Building, ...] = attr.ib()
    obstacle_zones: Tuple[Tuple[Optional[MapBoolNP], Optional[MapBoolNP]], ...] = attr.ib()

    view_mask: MapBoolNP = attr.ib(converter=to_immutable_array)
    is_climbable: MapBoolNP = attr.ib(converter=to_immutable_array)
    is_detail_important: MapBoolNP = attr.ib(converter=to_immutable_array)
    full_obstacle_mask: MapBoolNP = attr.ib(converter=to_immutable_array)
    flora_mask: MapFloatNP = attr.ib(converter=to_immutable_array)
    empty_space: MapBoolNP = attr.ib(converter=to_immutable_array)
    special_biomes: Optional[SpecialBiomes] = None

    is_debug_graph_printing_enabled: bool = False
    is_safe_mode_enabled: bool = True

    @staticmethod
    def build(
        config: WorldConfig,
        export_config: ExportConfig,
        biome_config: BiomeConfig,
        is_debug_graph_printing_enabled: bool = True,
    ) -> "World":
        map = build_outdoor_world_map(config, is_debug_graph_printing_enabled)
        true_array = np.ones_like(map.Z, dtype=np.bool_)
        return World(
            config=config,
            map=map,
            export_config=export_config,
            items=tuple(),
            obstacle_zones=tuple(),
            buildings=tuple(),
            empty_space=true_array,
            view_mask=np.ones_like(true_array),
            is_climbable=true_array.copy(),
            is_detail_important=np.zeros(map.Z.shape, dtype=np.bool_),
            full_obstacle_mask=np.zeros(map.Z.shape, dtype=np.bool_),
            is_debug_graph_printing_enabled=is_debug_graph_printing_enabled,
            biome_config=biome_config,
            flora_mask=np.ones_like(map.Z),
        )

    @property
    def building_by_id(self) -> Dict[int, Building]:
        return {x.id: x for x in self.buildings}

    def make_biome_noise(
        self, rand: np.random.Generator, resource_file: str, noise_min: float, noise_scale: float = 0.05
    ):
        if self.biome_config.is_independent_noise_per_scenery:
            noise = perlin(self.map.Z.shape, noise_scale, rand, is_normalized=True, noise_min=noise_min)
        else:
            noise = np.ones_like(self.map.Z)
        is_tree = "trees/" in resource_file
        if is_tree:
            noise *= self.view_mask
        return noise

    def flatten(self, point: Point2DNP, radius: float, importance_radius: float) -> "World":
        map_mutable = self.map.to_mutable()
        dist_sq = map_mutable.radial_flatten(point, radius)
        nearby = dist_sq < importance_radius * importance_radius
        is_detail_important_mutable = self.is_detail_important.copy()
        is_detail_important_mutable[nearby] = True
        return attr.evolve(self, is_detail_important=is_detail_important_mutable, map=map_mutable)

    def get_height_at(self, point: Union[Tuple[float, float], Point2DNP]) -> float:
        if isinstance(point, tuple):
            point = np.array(point)
        return cast(float, self.map.Z[self.map.point_to_index(point)])

    def reset_height_offset(self, item: EntityType, offset: float) -> EntityType:
        pos = item.position.copy()
        pos[1] = self.get_height_at(to_2d_point(item.position)) + offset
        return attr.evolve(item, position=pos)

    def add_items(self, items: Iterable[EntityType]) -> "World":
        items_new = []
        for item in items:
            items_new.append(item)
        return attr.evolve(self, items=self.items + tuple(items_new))

    def add_item(self, item: EntityType, reset_height_offset: Optional[float] = None) -> "World":
        if isinstance(item, FoodTree):
            num_trees = len([x for x in self.items if isinstance(x, FoodTree)])
            item = attr.evolve(item, entity_id=num_trees)  # type: ignore[assignment]
        if reset_height_offset is not None:
            item = self.reset_height_offset(item, reset_height_offset)
        return attr.evolve(self, items=self.items + (item,))

    def add_spawn(
        self,
        rand: np.random.Generator,
        difficulty: float,
        spawn_location: Point3DNP,
        food_location: Point3DNP,
        is_visibility_required: bool = True,
        is_spawn_height_reset: bool = True,
    ) -> "World":
        if is_spawn_height_reset:
            spawn_location = spawn_location.copy()
            height_at_location = self.map.Z[self.map.point_to_index(to_2d_point(spawn_location))]
            spawn_location[1] = height_at_location + HALF_AGENT_HEIGHT_VECTOR[1] * 1.1
        spawn_item = get_spawn(rand, difficulty, spawn_location, food_location)
        spawn_item = attr.evolve(spawn_item, is_visibility_required=is_visibility_required)
        new_items = self.items + (spawn_item,)

        # update the flora mask to prevent things from spawning in the special areas
        flora_removal_cells = round(self.map.cells_per_meter * FLORA_REMOVAL_METERS) + 1
        flora_removal_mask = morphology.dilation(self.is_detail_important, morphology.disk(flora_removal_cells))
        new_flora_mask = self.flora_mask.copy()
        new_flora_mask[flora_removal_mask] = 0.0

        # update our view_mask to prevent trees from blocking our view of the goal
        view_mask_new = self.view_mask.copy()
        if is_visibility_required:
            visibility_line_width = METERS_OF_TREE_CLEARANCE_AROUND_LINE_OF_SIGHT
            segment_dist = self.map.get_lineseg_distances(to_2d_point(spawn_location), to_2d_point(food_location))
            close_to_sight_line = segment_dist < visibility_line_width
            view_mask_new[close_to_sight_line] = False

        return attr.evolve(self, items=new_items, flora_mask=new_flora_mask, view_mask=view_mask_new)

    def add_spawn_and_food(
        self,
        rand: np.random.Generator,
        difficulty: float,
        spawn_location: Point3DNP,
        food_location: Point3DNP,
        food_class: Optional[Type[Food]] = None,
        is_visibility_required: bool = True,
    ) -> "World":
        if food_class is None:
            food_class = CANONICAL_FOOD_CLASS
        food_location = food_location.copy()
        # TODO: not quite right, this is the wrong offset...
        food_location[1] = (
            self.map.Z[self.map.point_to_index(to_2d_point(food_location))] + CANONICAL_FOOD.get_offset()
        )
        food = food_class(position=food_location)
        new_items = self.items + (food,)
        new_world = attr.evolve(self, items=new_items)
        return new_world.add_spawn(
            rand, difficulty, spawn_location, food_location, is_visibility_required=is_visibility_required
        )

    def replace_weapon_placeholders(
        self, replacements: Iterable[Type[Weapon]], island_mask: MapBoolNP, flatten_radius: float
    ) -> "World":
        fixed_items: List[Entity] = []
        weapon_types = iter(replacements)
        flatten_areas = []
        for item in self.items:
            if isinstance(item, Placeholder):
                weapon_type = next(weapon_types)
                new_item = weapon_type(
                    position=item.position, rotation=item.rotation, solution_mask=item.solution_mask
                )

                pos = new_item.position.copy()
                pos_2d = to_2d_point(new_item.position)
                height = self.get_height_at(pos_2d)
                pos[1] = height + new_item.get_offset()
                new_item = attr.evolve(new_item, position=pos)

                flatten_areas.append((height, pos_2d))

                fixed_items.append(new_item)
            else:
                fixed_items.append(item)

        map_new = self.map.copy()
        is_detail_important_mutable = self.is_detail_important.copy()
        if flatten_radius > 0.0:
            for height, pos_2d in list(reversed(sorted(flatten_areas, key=lambda x: x[0]))):
                dist_sq = map_new.radial_flatten(pos_2d, flatten_radius, island_mask)
                nearby = dist_sq < flatten_radius * flatten_radius
                is_detail_important_mutable[nearby] = True

        return attr.evolve(
            self, items=tuple(fixed_items), map=map_new, is_detail_important=is_detail_important_mutable
        )

    def get_random_point_for_weapon_or_predator(
        self, rand: np.random.Generator, point: Point2DNP, radius: float, island_mask: MapBoolNP
    ) -> Optional[Point2DNP]:
        safe_mask = self.get_safe_mask(island_mask=island_mask)
        safe_places = np.argwhere(safe_mask)
        possible_points = self.map.get_2d_points()[safe_places[:, 0], safe_places[:, 1]]
        distances = np.linalg.norm(possible_points - point, axis=1)
        for std_dev in (1.0, 5.0):
            location_weights = stats.norm(radius, std_dev).pdf(distances)
            total_weights = location_weights.sum()
            if total_weights <= 0.0:
                continue
        if total_weights <= 0.0:
            return None
        location_probabilities = location_weights / total_weights
        return cast(Point2DNP, rand.choice(possible_points, p=location_probabilities))

    def add_random_predator_near_point(
        self,
        rand: np.random.Generator,
        predator_type: Type[Predator],
        location: Point2DNP,
        radius: float,
        island_mask: MapBoolNP,
    ) -> "World":
        predator_pos_2d = self.get_random_point_for_weapon_or_predator(rand, location, radius, island_mask)
        if predator_pos_2d is None:

            raise ImpossibleWorldError("Nowhere to stick a single predator. Weird")
        # add predator near food
        predator_position = np.array([predator_pos_2d[0], 0.0, predator_pos_2d[1]])
        predator = predator_type(position=predator_position)
        return self.add_item(predator, reset_height_offset=predator.get_offset())

    def carry_tool_randomly(self, rand: np.random.Generator, item: Tool, distance_preference: stats.norm) -> "World":

        assert (
            item.solution_mask is not None
        ), "How was this tool placed? Ideally we should be setting the solution_mask when placing tools, so that we can carry them later"

        # don't accidentally stomp on anything else
        possible_spawn_mask = np.logical_and(item.solution_mask, np.logical_not(self.is_detail_important))

        # if there is nowhere to go, we're done
        if not np.any(possible_spawn_mask):
            return attr.evolve(self)

        # figure out the new position
        center_point = to_2d_point(item.position)
        possible_points = self.map.get_2d_points()[possible_spawn_mask]
        new_point_2d = only(
            selected_distance_weighted_points(rand, possible_points, center_point, distance_preference, 1)
        )
        height = self.map.get_rough_height_at_point(new_point_2d) + item.get_offset()
        new_position = np.array([new_point_2d[0], height, new_point_2d[1]])

        # logger.debug(f"Carried object {np.linalg.norm(new_point_2d - center_point)}m")

        # replace the item with one that has an updated position
        prev_len = len(self.items)
        new_items = [x for x in self.items if x != item]
        assert len(new_items) == prev_len - 1, "Accidentally removed multiple items, that's bad"
        item = attr.evolve(item, position=new_position)
        new_items.append(item)

        important_points = add_detail_near_items(self.map, [item], item.get_offset())
        new_is_detail_important = self.is_detail_important.copy()
        new_is_detail_important[important_points] = True

        return attr.evolve(self, is_detail_important=new_is_detail_important, items=tuple(new_items))

    def get_safe_point(
        self,
        rand: np.random.Generator,
        sq_distances: Optional[MapFloatNP] = None,
        max_sq_dist: Optional[float] = None,
        island_mask: Optional[MapBoolNP] = None,
    ) -> Optional[Point3DNP]:
        mask = self.get_safe_mask(island_mask, max_sq_dist, sq_distances)
        return self._get_safe_point(rand, mask)

    def _get_safe_point(self, rand: np.random.Generator, mask: MapBoolNP) -> Optional[Point3DNP]:
        if not np.any(mask):
            return None
        selected_coords = cast(Tuple[int, int], tuple(rand.choice(np.argwhere(mask))))
        point_2d = self.map.index_to_point_2d(selected_coords)
        return np.array([point_2d[0], self.map.get_rough_height_at_point(point_2d), point_2d[1]])

    def get_safe_mask(
        self,
        island_mask: Optional[MapBoolNP] = None,
        max_sq_dist: Optional[float] = None,
        sq_distances: Optional[MapFloatNP] = None,
    ) -> MapBoolNP:
        # start with all land
        mask = self.map.get_land_mask()
        # remove all obstacles
        mask = np.logical_and(mask, np.logical_not(self.full_obstacle_mask))
        # remove places where detail is important
        mask = np.logical_and(mask, np.logical_not(self.is_detail_important))
        # remove unclimbable places
        mask = np.logical_and(mask, self.is_climbable)
        # if specified, restrict to a single continent
        if island_mask is not None:
            mask = np.logical_and(mask, island_mask)
        # if specified, restrict to points nearby
        if sq_distances is not None:
            assert max_sq_dist is not None
            mask = np.logical_and(mask, sq_distances < max_sq_dist)
        return mask

    def add_height_obstacle(
        self, rand: np.random.Generator, ring_config: RingObstacleConfig, island_mask: MapBoolNP
    ) -> "World":

        if self.is_debug_graph_printing_enabled:
            plot_terrain(self.map.Z, "Terrain (before obstacle)")

        if ring_config.inner_obstacle is None or ring_config.outer_obstacle is None:
            assert ring_config.chasm_bottom_size == 0.0, "Cannot make a chasm with a single obstacle"

        assert ring_config.chasm_bottom_size >= 0, "chasm_bottom_size cannot be negative"

        if ring_config.inner_obstacle and ring_config.outer_obstacle:
            ring = self._create_ring(
                rand,
                ring_config,
                ring_config.inner_safety_radius + ring_config.inner_obstacle.traversal_length,
                ring_config.outer_safety_radius
                - (ring_config.outer_obstacle.traversal_length + ring_config.chasm_bottom_size),
            )
        elif ring_config.inner_obstacle:
            ring = self._create_ring(
                rand,
                ring_config,
                ring_config.inner_safety_radius + ring_config.inner_obstacle.traversal_length,
                ring_config.outer_safety_radius,
            )
        elif ring_config.outer_obstacle:
            ring = self._create_ring(
                rand,
                ring_config,
                ring_config.inner_safety_radius,
                ring_config.outer_safety_radius - ring_config.outer_obstacle.traversal_length,
            )
        else:
            raise Exception("Must define one of inner or outer obstacle")

        inner_mask = None
        outer_mask = None

        if ring_config.inner_obstacle:
            inner_obstacle_weight_map, inner_mask, outside_inner_obstacle_mask = create_obstacle_masks(
                rand, ring, ring_config.inner_obstacle, island_mask
            )

        if ring_config.outer_obstacle:
            # TODO: mutating mid_z here is kinda weird, dont do that
            if ring_config.inner_obstacle:
                # ring.mid_z += ring_config.chasm_bottom_size
                orig = ring.mid_z
                ring.mid_z = ring.outer_mid_z
            outer_obstacle_weight_map, inside_outer_obstacle_mask, outer_mask = create_obstacle_masks(
                rand, ring, ring_config.outer_obstacle, island_mask
            )
            if ring_config.inner_obstacle:
                # ring.mid_z -= ring_config.chasm_bottom_size / 2.0
                ring.mid_z = (orig + ring.outer_mid_z) / 2.0

        if ring_config.inner_obstacle:
            inner_point_for_inner_solution, outer_point_for_inner_solution = ring_config.get_inner_solution_points()
            inner_traversal_midpoint = (inner_point_for_inner_solution + outer_point_for_inner_solution) / 2.0
        if ring_config.outer_obstacle:
            inner_point_for_outer_solution, outer_point_for_outer_solution = ring_config.get_outer_solution_points()
            outer_traversal_midpoint = (inner_point_for_outer_solution + outer_point_for_outer_solution) / 2.0

        # when applying for chasm or ridge, need to be really careful that the edges of the masks work out
        if ring_config.inner_obstacle and ring_config.outer_obstacle:
            if self.is_debug_graph_printing_enabled:
                plot_value_grid(
                    inner_obstacle_weight_map - outer_obstacle_weight_map, "Inner and outer obstacle delta"
                )
            map_new, is_climbable_new = self._apply_height_obstacle(
                inner_obstacle_weight_map - outer_obstacle_weight_map,
                ring,
                island_mask,
                inner_point_for_inner_solution,
                outer_point_for_outer_solution,
            )
        elif ring_config.inner_obstacle:
            map_new, is_climbable_new = self._apply_height_obstacle(
                inner_obstacle_weight_map,
                ring,
                island_mask,
                inner_point_for_inner_solution,
                outer_point_for_inner_solution,
            )
        elif ring_config.outer_obstacle:
            map_new, is_climbable_new = self._apply_height_obstacle(
                outer_obstacle_weight_map,
                ring,
                island_mask,
                inner_point_for_outer_solution,
                outer_point_for_outer_solution,
            )
        else:
            raise Exception("Pointless obstacle")

        is_detail_important_new = self.is_detail_important.copy()

        # ignore borders with the ocean, they should not be marked as important
        not_island_borders = np.logical_not(map_new.get_outline(island_mask, 3))
        for i in range(2):
            if i == 0:
                obstacle = ring_config.inner_obstacle
            else:
                obstacle = ring_config.outer_obstacle
            if obstacle:
                if i == 0:
                    mask = inner_obstacle_weight_map
                    traversal_point = inner_traversal_midpoint
                else:
                    mask = outer_obstacle_weight_map
                    traversal_point = outer_traversal_midpoint
                # update the climbability where the height changed if necessary
                if not obstacle.is_default_climbable:
                    masked_climbing_region = np.logical_and(mask < 1.0, mask > 0.0)
                    # applies a 3x3 pass to ensure that nearby elements are also considered unclimbable
                    masked_climbing_region = ndimage.binary_dilation(
                        masked_climbing_region, structure=ndimage.generate_binary_structure(2, 2)
                    )
                    # mask out anything that is faded
                    # plot_value_grid(masked_climbing_region, "masked_climbing_region")
                    is_climbable_new[masked_climbing_region] = False

                # we mark things as important if they are sufficiently close to a traversal point
                # AND there is a big delta in height. Otherwise we lose important detail on some obstacles (namely the bottoms of chasms)
                locations = np.stack([self.map.X, self.map.Y], axis=2)
                dist_sq = (locations[:, :, 0] - traversal_point[0]) ** 2 + (
                    locations[:, :, 1] - traversal_point[1]
                ) ** 2
                obstacle_radius = obstacle.detail_radius
                near_important_point = dist_sq < obstacle_radius * obstacle_radius

                # also union this with any place where a cel is 0 or 1, but neighbors are not
                for val in (0, 1):
                    extra_mask = np.zeros_like(mask)
                    extra_mask[mask == val] = 1.0
                    expanded_extra_mask = ndimage.binary_dilation(extra_mask, structure=morphology.disk(3))
                    final_extra_mask = np.logical_and(expanded_extra_mask, np.logical_not(extra_mask))
                    final_extra_mask = np.logical_and(final_extra_mask, not_island_borders)
                    is_detail_important_new[np.logical_and(final_extra_mask, near_important_point)] = True
                    # plot_value_grid(final_extra_mask, "final_extra_mask")
                    # plot_value_grid(np.logical_and(final_extra_mask, not_island_borders), "Fixed?")
                    # plot_value_grid(near_important_point, "near_important_point")
                    # plot_value_grid(is_detail_important_new, "detail")
                    if not obstacle.is_default_climbable:
                        # plot_value_grid(final_extra_mask, "final_extra_mask")
                        is_climbable_new[final_extra_mask] = False

        if self.is_debug_graph_printing_enabled:
            plot_terrain(self.map.Z, "Height before obstacle")
            plot_terrain(map_new.Z, "Height after obstacle")
            plot_value_grid(is_detail_important_new, "Is detail important after obstacle")

        items: List[InstancedDynamicItem] = []

        if ring_config.inner_solution:
            assert inner_mask is not None
            inside_inner_safety_region_mask = ring.r < ring_config.inner_safety_radius
            inner_solution_mask = np.logical_and(inner_mask, np.logical_not(inside_inner_safety_region_mask))
            inside_mid_z = ring.z < ring.mid_z
            inside_inner_obstacle = inner_mask
            only_inner_obstacle_weight_map = inner_obstacle_weight_map.copy()
            only_inner_obstacle_weight_map[inner_obstacle_weight_map == 1.0] = 0.0
            if ring_config.outer_obstacle:
                outer_solution_mask = np.logical_and(inside_mid_z, np.logical_not(inside_inner_obstacle))
            else:
                outer_solution_mask = np.logical_and(
                    ring.r < ring_config.outer_safety_radius, np.logical_not(inside_inner_obstacle)
                )
                outer_solution_mask = np.logical_and(
                    outer_solution_mask, np.logical_not(only_inner_obstacle_weight_map > 0.0)
                )
            outer_solution_mask = np.logical_and(outer_solution_mask, island_mask)

            if ring.config.height_mode == HeightMode.MIDPOINT_RELATIVE:
                map_new.interpolate_heights(
                    outer_solution_mask,
                    np.logical_and(only_inner_obstacle_weight_map > 0.0, island_mask),
                )

            new_items = ring_config.inner_solution.apply(
                rand,
                map_new,
                is_climbable_new,
                is_detail_important_new,
                only_inner_obstacle_weight_map,
                inner_solution_mask,
                outer_solution_mask,
                inner_point_for_inner_solution,
                outer_point_for_inner_solution,
            )
            items.extend(new_items)

        if ring_config.outer_solution:
            assert outer_mask is not None
            outside_mid_z = ring.z > ring.mid_z
            inside_outer_obstacle = inside_outer_obstacle_mask
            if ring_config.inner_obstacle:
                inner_solution_mask = np.logical_and(outside_mid_z, inside_outer_obstacle)
            else:
                inner_solution_mask = np.logical_and(ring.r > ring_config.inner_safety_radius, inside_outer_obstacle)
                assert False, "This really needs to be better tested!  Test before using."
            inner_solution_mask = np.logical_and(inner_solution_mask, island_mask)

            outside_outer_safety_region_mask = ring.r > ring_config.outer_safety_radius
            outer_solution_mask = np.logical_and(outer_mask, np.logical_not(outside_outer_safety_region_mask))

            only_outer_obstacle_weight_map = outer_obstacle_weight_map.copy()
            only_outer_obstacle_weight_map[outer_obstacle_weight_map == 1.0] = 0.0

            if ring.config.height_mode == HeightMode.MIDPOINT_RELATIVE:
                assert (
                    False
                ), "Not implemented. See above, but I think this needs to change to specify on which side heights should be blended"

            new_items = ring_config.outer_solution.apply(
                rand,
                map_new,
                is_climbable_new,
                is_detail_important_new,
                only_outer_obstacle_weight_map,
                inner_solution_mask,
                outer_solution_mask,
                inner_point_for_outer_solution,
                outer_point_for_outer_solution,
            )
            items.extend(new_items)

        if ring_config.dual_solution:
            assert inner_mask is not None and outer_mask is not None
            # markers = [
            #     map_new.point_to_index(x)
            #     for x in [
            #         inner_point_for_inner_solution,
            #         outer_point_for_inner_solution,
            #         inner_point_for_outer_solution,
            #         outer_point_for_outer_solution,
            #     ]
            # ]

            inside_inner_safety_region_mask = ring.r < ring_config.inner_safety_radius
            inner_solution_mask = np.logical_and(inner_mask, np.logical_not(inside_inner_safety_region_mask))
            outside_outer_safety_region_mask = ring.r > ring_config.outer_safety_radius
            outer_solution_mask = np.logical_and(outer_mask, np.logical_not(outside_outer_safety_region_mask))

            # outer_obstacle_weight_map, inside_outer_obstacle_mask,

            # plot_value_grid(inner_obstacle_weight_map - outer_obstacle_weight_map, markers=markers)
            # logger.debulogger.debug(ring.mid_z)
            # logger.debug(np.linalg.norm(outer_point_for_outer_solution - inner_point_for_inner_solution))
            # plot_value_grid(
            #     outer_solution_mask,
            #     "ARGGG",
            #     markers=markers,
            # )

            new_items = ring_config.dual_solution.apply(
                rand,
                map_new,
                is_climbable_new,
                is_detail_important_new,
                inner_obstacle_weight_map - outer_obstacle_weight_map,
                inner_solution_mask,
                outer_solution_mask,
                inner_point_for_inner_solution,
                outer_point_for_outer_solution,
            )
            items.extend(new_items)

        for item in items:
            # flatten near each of the items, otherwise you end up with some pretty impossible stuff
            map_new.radial_flatten(to_2d_point(item.position), item.get_offset() * 2.0, island_mask)

        # validate that nothing went horribly wrong:
        assert (
            map_new.Z[island_mask].min() > WORLD_RAISE_AMOUNT - 1000
        ), "Hmmm, seems like we accidentally connected to a point outside of the island, not good"

        extended_island_mask = morphology.dilation(island_mask, morphology.disk(6))

        # keeps track of all of the places that we made into obstacles, so we can place things later
        if ring_config.inner_obstacle and ring_config.outer_obstacle:
            is_obstacle = (inner_obstacle_weight_map - outer_obstacle_weight_map) > 0.0
        elif ring_config.inner_obstacle:
            is_obstacle = np.logical_and(inner_obstacle_weight_map > 0.0, inner_obstacle_weight_map < 1.0)
            if not np.any(is_obstacle):
                is_obstacle = map_new.get_outline(inner_obstacle_weight_map, 1)
        else:
            is_obstacle = np.logical_and(outer_obstacle_weight_map > 0.0, outer_obstacle_weight_map < 1.0)
            if not np.any(is_obstacle):
                is_obstacle = map_new.get_outline(outer_obstacle_weight_map, 1)
        full_obstacle_mask_new = np.logical_or(self.full_obstacle_mask, is_obstacle)

        if self.is_debug_graph_printing_enabled:
            plot_value_grid(is_climbable_new, "New is climbable")
            plot_terrain(map_new.Z, "Terrain (after obstacle)")

        is_climbable_final = self.is_climbable.copy()
        is_climbable_final[extended_island_mask] = is_climbable_new[extended_island_mask]

        new_obstacle_zones = [x for x in self.obstacle_zones]
        new_obstacle_zones.append((inner_mask, outer_mask))

        return attr.evolve(
            self,
            map=map_new,
            full_obstacle_mask=full_obstacle_mask_new,
            is_detail_important=is_detail_important_new,
            is_climbable=is_climbable_final,
            obstacle_zones=tuple(new_obstacle_zones),
        ).add_items(items)

    def generate_terrain(self, rand: np.random.Generator, biome_map: BiomeMap) -> Terrain:
        is_climbable_fixed = self.is_climbable.copy()
        # plot_value_grid(is_climbable_fixed, "final is climbable")
        # if self.is_coast_unclimbable:
        #     is_climbable_fixed = np.logical_and(is_climbable_fixed, np.logical_not(biome_map.map.get_all_coast()))
        #     is_climbable_fixed[self.is_detail_important] = self.is_climbable[self.is_detail_important]
        return Terrain(
            biome_map,
            is_climbable_fixed,
            self.is_detail_important,
            self.config.point_density_in_points_per_square_meter,
            rand,
        )

    def add_building(self, building: Building, mask: MapBoolNP) -> Tuple[Building, "World"]:
        next_building_id = max([-1] + [x.id for x in self.buildings]) + 1
        building = attr.evolve(building, id=next_building_id)
        new_full_obstacle_mask = np.logical_or(self.full_obstacle_mask, mask)
        return building, attr.evolve(
            self, buildings=self.buildings + (building,), full_obstacle_mask=new_full_obstacle_mask
        )

    def plot_visibility(
        self,
        spawn_point: Point3DNP,
        point_offset: Point3DNP,
        markers: Iterable[Union[np.ndarray, Tuple[float, float]]],
    ):
        visibility_calculator = self.map.generate_visibility_calculator()
        visibility = np.zeros_like(self.map.Z)
        spawn_point_2d = np.array([spawn_point[0], spawn_point[2]])
        for i, j in np.ndindex(*self.map.Z.shape):
            x = self.map.X[i, j]
            y = self.map.Y[i, j]
            z = self.map.Z[i, j]
            # for the ocean
            if z < 0:
                z = 0.0
            point = np.array([x, z, y])
            if np.isclose(np.array([x, y]), spawn_point_2d).all():
                visibility[i, j] = 1.0
            else:
                visibility[i, j] = visibility_calculator.is_visible_from(
                    spawn_point + HALF_AGENT_HEIGHT_VECTOR, point + point_offset
                )
                # if (i, j) in markers:
                #     logger.debug(spawn_point + HALF_AGENT_HEIGHT_VECTOR, point + point_offset)
                #     logger.debug(visibility[i, j])
                # if (i, j) == (32, 22):
                #     logger.debug(point)
                #     visibility[i, j] = 1.0
                #     visibility_calculator.is_visible_from(spawn_point + HALF_AGENT_HEIGHT_VECTOR, point,
                #     is_plotted=True)
        plot_value_grid(visibility, markers=markers)
        return visibility

    def _apply_height_obstacle(
        self,
        mask: np.ndarray,
        ring: RingObstacle,
        island_mask: MapBoolNP,
        start_point: Optional[Point2DNP],
        end_point: Optional[Point2DNP],
    ) -> Tuple[HeightMap, np.ndarray]:
        map_new = self.map.copy()
        mask = mask.copy()
        mask[np.logical_not(island_mask)] = 0.0

        if ring.config.terrain_blurs:
            # make things a bit smoother around the traversal point first
            # plot_terrain(map_new.Z)
            assert start_point is not None and end_point is not None
            mid_point = (start_point + end_point) / 2.0
            water_mask = np.logical_not(island_mask)
            for mask_radius, blur_radius in ring.config.terrain_blurs:
                nearby = map_new.get_dist_sq_to(mid_point) < mask_radius**2
                blur_mask = np.logical_and(island_mask, nearby)
                rest_mask = np.logical_not(blur_mask)
                rest_mask = morphology.dilation(rest_mask, morphology.disk(3))
                blur_mask = np.logical_not(rest_mask)

                blur_map = map_new.copy()
                blur_map.Z[water_mask] = blur_map.Z[blur_mask].min()
                blur_map.blur(blur_radius)
                map_new.Z[blur_mask] = blur_map.Z[blur_mask]
                # plot_terrain(map_new.Z)

        if (
            ring.config.height_mode == HeightMode.MIDPOINT_ABSOLUTE
            or ring.config.height_mode == HeightMode.MIDPOINT_RELATIVE
        ):
            if ring.config.inner_obstacle and ring.config.outer_obstacle:
                assert ring.config.height_mode == HeightMode.MIDPOINT_ABSOLUTE
                assert start_point is not None and end_point is not None
                start_height = map_new.get_rough_height_at_point(start_point)
                end_height = map_new.get_rough_height_at_point(end_point)
                if ring.config.height < 0.0:
                    base_height = max([start_height, end_height])
                else:
                    base_height = min([start_height, end_height])
                # this is pretty gross, but required for current tasks...
                if ring.config.expansion_meters > 0.0:
                    expanded_mask = mask.copy()
                    cell_units = int(ring.config.expansion_meters * map_new.cells_per_meter) + 1
                    # logger.debug(f"{cell_units=}")
                    nearby = morphology.dilation(expanded_mask > 0.0, morphology.disk(cell_units))
                    expanded_mask[nearby] = np.clip(expanded_mask[nearby], 0.001, None)
                    expanded_mask[np.logical_not(island_mask)] = 0.0
                else:
                    expanded_mask = mask
                map_new.Z[expanded_mask > 0.0] = base_height
                # plot_value_grid(expanded_mask)
                # plot_value_grid(mask > 0.0)
                # plot_value_grid(expanded_mask > 0.0)
                map_new.apply_height_mask(expanded_mask, ring.config.height, HeightMode.RELATIVE)
            else:
                # only makes sense if this is not double sided
                assert start_point is not None
                boolean_obstacle_mask = np.logical_and(mask > 0.0, mask < 1.0)
                if ring.config.height_mode == HeightMode.MIDPOINT_ABSOLUTE:
                    # set everything to the start height
                    start_height = map_new.get_rough_height_at_point(start_point)
                    map_new.Z[boolean_obstacle_mask] = start_height
                else:
                    start_z = ring.z[map_new.point_to_index(start_point)]
                    near_start_z = map_new.get_outline(ring.z < start_z, 1)
                    near_start_z = np.logical_and(near_start_z, island_mask)
                    # plot_value_grid(near_start_z, "Near mid z for crazy heightmap thing")
                    ring_indices = np.argwhere(near_start_z)
                    assert ring_indices.shape[-1] == 2
                    ring_points = np.stack(
                        [map_new.X[0, ring_indices[:, 1]], map_new.Y[ring_indices[:, 0], 0]], axis=1
                    )
                    ring_heights = map_new.get_heights(ring_points)
                    # interp = NearestNDInterpolator(ring_points, ring_heights)
                    # # set everything to the height at the nearest midpoint
                    # plot_value_grid(boolean_obstacle_mask, "boolean_obstacle_mask")
                    # map_new.Z[boolean_obstacle_mask] = interp(
                    #     map_new.X[boolean_obstacle_mask], map_new.Y[boolean_obstacle_mask]
                    # )
                    ring_thetas = ring.theta[ring_indices[:, 0], ring_indices[:, 1]]
                    sorted_tuples = sorted(tuple(x) for x in zip(ring_thetas, ring_heights))
                    first_tuple = sorted_tuples[0]
                    last_tuple = sorted_tuples[-1]
                    twopi = np.pi * 2.0
                    sorted_tuples.insert(0, (last_tuple[0] - twopi, last_tuple[1]))
                    sorted_tuples.append((first_tuple[0] + twopi, first_tuple[1]))
                    sorted_tuples_np = np.array(sorted_tuples)
                    # wrap with 2pi and pi versions of the endpoints
                    height_by_theta = interp1d(sorted_tuples_np[:, 0], sorted_tuples_np[:, 1])
                    for idx in np.argwhere(boolean_obstacle_mask):
                        idx = tuple(idx)
                        theta = ring.theta[idx]
                        map_new.Z[idx] = height_by_theta(theta)
                # apply the mask to apply the delta
                map_new.apply_height_mask(mask, ring.config.height, HeightMode.RELATIVE)
        else:
            map_new.apply_height_mask(mask, ring.config.height, ring.config.height_mode)
        is_climbable_new = self.is_climbable.copy()
        return map_new, is_climbable_new

    def _create_ring(
        self, rand: np.random.Generator, config: RingObstacleConfig, safety_radius_min: float, safety_radius_max: float
    ) -> RingObstacle:

        # logger.debug(safety_radius_min)
        # logger.debug(safety_radius_max)

        edge_config = config.edge

        # shift to account for the center point
        x_diff = config.center_point[0] - self.map.X
        y_diff = config.center_point[1] - self.map.Y

        # convert to polar coordinates
        r = np.sqrt(x_diff * x_diff + y_diff * y_diff)
        traversal_distance = np.sqrt(np.sum((config.center_point - config.traversal_point) ** 2))
        if traversal_distance < safety_radius_min or traversal_distance > safety_radius_max:
            raise ImpossibleWorldError(
                f"Impossible to create a ring between {safety_radius_min} and {safety_radius_max} when the traversal point is {traversal_distance} meters away!"
            )
        normalized_r = r / traversal_distance
        theta = np.arctan2(y_diff, x_diff)

        # figure out where we cross
        traversal_indices = self.map.point_to_index(config.traversal_point)  # y, x

        outer_traversal_indices = None
        outer_mid_z = 0.0
        if config.inner_obstacle and config.outer_obstacle:
            outer_r = traversal_distance + config.chasm_bottom_size
            traversal_theta = theta[traversal_indices]
            outer_traversal_point = (
                -1.0 * np.array([outer_r * math.cos(traversal_theta), outer_r * math.sin(traversal_theta)])
                + config.center_point
            )
            outer_traversal_indices = self.map.point_to_index(outer_traversal_point)

        # create the "z" field of values.
        # we use various isolines from this z field to define the edges of the obstacle
        # do this repeatedly until we end up with a circle that fits between the safety margins
        for i in range(10):
            harmonics = edge_config.to_harmonics(traversal_distance)
            variation = create_harmonics(rand, theta, config=harmonics, is_normalized=False)
            z = np.sqrt(np.clip(normalized_r**2 + normalized_r * variation, 0.01, np.inf))

            mid_z = z[traversal_indices]
            z = (z / mid_z) * traversal_distance
            mid_z = traversal_distance
            if outer_traversal_indices:
                outer_mid_z = z[outer_traversal_indices]

                # plot_value_grid(
                #     np.logical_and(z < outer_mid_z, z > mid_z),
                #     "sigh world",
                #     markers=[traversal_indices, outer_traversal_indices],
                # )

            for margin in (0.1, 0.5):
                outline = np.bitwise_and(safety_radius_min + margin > r, r > safety_radius_min - margin)
                if np.any(outline):
                    break
            # plot_value_grid(outline)
            if np.any(outline):
                inner_z_at_safety_radius = z[outline].max()

                for margin in (0.1, 0.5):
                    outline = np.bitwise_and(safety_radius_max + margin > r, r > safety_radius_max - margin)
                    if np.any(outline):
                        break
                # plot_value_grid(outline)
                if np.any(outline):
                    outer_z_at_safety_radius = z[outline].min()
                    if mid_z - inner_z_at_safety_radius > 0.0 and outer_z_at_safety_radius - mid_z > 0.0:
                        break

            # if we get here, sigh... means that it was hard to make a circle like what we wanted, try again with something more circular
            edge_config = attr.evolve(
                edge_config, circularity=(edge_config.circularity + 1.0) / 2.0, noise=(edge_config.noise + 0.0) / 2.0
            )
        else:
            raise ImpossibleWorldError("Tried to make a circle many times, bounds are too tight!")

        # safety check: if you cross too close, it will be sad (divide by zero below)
        if self.is_safe_mode_enabled:
            center_indices = self.map.point_to_index(config.center_point)  # y, x
            if center_indices == traversal_indices:
                raise Exception(
                    f"Cannot set traversal point and center point too close: center={config.center_point} and traversal={config.traversal_point}"
                )

        # figure out where the middle of the chasm is (the z value) and rescale z
        # such that it roughly puts z into world units (ie, meters)
        if self.is_debug_graph_printing_enabled:
            margin = 2.0
            outline = np.bitwise_and(mid_z + margin > z, z > mid_z - margin).astype(np.float64)

            outline[r < safety_radius_min] = 0.5
            outline[r > safety_radius_max] = 0.5

            plot_value_grid(outline, title=f"Mid Z +/- {margin}m")

            plot_value_grid(z, title="Raw Z function")

        return RingObstacle(config=config, r=r, z=z, theta=theta, mid_z=mid_z, outer_mid_z=outer_mid_z)

    def reset_beaches(self, island: MapBoolNP) -> "World":
        # remove beaches for anything that has gone above the height on our island
        if self.special_biomes is None:
            return self
        self.special_biomes.swamp_mask[island] = self.map.Z[island] < self.biome_config.swamp_elevation_max
        self.special_biomes.fresh_water_mask[island] = self.map.Z[island] < 0.0
        self.special_biomes.beach_mask[island] = self.map.Z[island] < self.biome_config.max_shore_height
        squared_slope = self.map.get_squared_slope()
        cliffs = squared_slope > self.biome_config.force_cliff_square_slope
        cliffs = morphology.dilation(cliffs, morphology.disk(5))
        self.special_biomes.beach_mask[cliffs] = False
        return self

    def begin_height_obstacles(self, locations: WorldLocations) -> Tuple[WorldLocations, "World"]:
        map_new = self.map.copy()
        map_new.raise_island(locations.island, WORLD_RAISE_AMOUNT)
        delta = UP_VECTOR * WORLD_RAISE_AMOUNT

        items_new = []
        for item in self.items:
            position_new = item.position.copy()
            position_new[1] += WORLD_RAISE_AMOUNT
            items_new.append(attr.evolve(item, position=position_new))

        buildings_new = []
        for building in self.buildings:
            new_position = building.position.copy()
            new_position[1] = building.position[1] + WORLD_RAISE_AMOUNT
            buildings_new.append(attr.evolve(building, position=new_position))

        return attr.evolve(locations, goal=locations.goal + delta, spawn=locations.spawn + delta), attr.evolve(
            self, buildings=tuple(buildings_new), map=map_new, items=tuple(items_new)
        )

    def end_height_obstacles(
        self,
        locations: WorldLocations,
        is_accessible_from_water: bool,
        spawn_region: Optional[MapBoolNP] = None,
        is_spawn_region_climbable: bool = True,
    ) -> Tuple["World", WorldLocations]:
        # player can't jump up here. Is a little higher than stricly necessary bc dont want to grab stuff
        MIN_INACCESSIBLE_HEIGHT = 3.0

        if is_accessible_from_water:
            sea_height = 0.0
        else:
            sea_height = MIN_INACCESSIBLE_HEIGHT
        # lower the island back down
        map_new = self.map.copy()
        lowered_amount = map_new.lower_island(locations.island, sea_height)
        # lower all of the resulting positions as well
        fixed_items = []
        for item in self.items:
            position_new = item.position.copy()
            position_new[1] = position_new[1] - lowered_amount
            # TODO: remove this! Will certainly be annoying to debug when we add things underwater...
            if position_new[1] < 0.0:
                position_new[1] = 0.5
            item = attr.evolve(item, position=position_new)
            fixed_items.append(item)

        buildings_new = []
        for building in self.buildings:
            new_position = building.position.copy()
            new_position[1] = building.position[1] - lowered_amount
            buildings_new.append(attr.evolve(building, position=new_position))

        spawn_new = locations.spawn.copy()
        spawn_new[1] -= lowered_amount
        goal_new = locations.goal.copy()
        goal_new[1] -= lowered_amount
        new_locations = attr.evolve(locations, spawn=spawn_new, goal=goal_new)

        new_world = attr.evolve(self, items=tuple(fixed_items), buildings=tuple(buildings_new), map=map_new)

        if len(new_world.obstacle_zones) == 0:
            return new_world, new_locations

        assert len(new_world.obstacle_zones) >= 1
        spawn_region = new_world._get_spawn_zone(new_locations, spawn_region)

        assert spawn_region is not None

        # set unclimbability around the island (except our spawn zone, that can be climbable)
        is_climbable_fixed = new_world.is_climbable.copy()

        climbability_cell_radius = 4
        island_cliff_edge = new_world.map.get_outline(new_locations.island, climbability_cell_radius - 1)
        is_climbable_fixed[island_cliff_edge] = False
        # plot_value_grid(island_cliff_edge, "clif edge")
        # plot_value_grid(is_climbable_fixed, "is climb")
        # plot_value_grid(new_world.is_detail_important, "base is important")

        if is_spawn_region_climbable:
            # we want to be able to retry tasks by walking back to our spawn
            # but only makes sense to make the shore nearby unclimbable, not all cliffs
            # since otherwise tasks like climb and stack, which start you in a pit, are broken
            near_spawn_climbable = np.logical_and(
                spawn_region,
                np.logical_not(
                    morphology.dilation(new_world.full_obstacle_mask, morphology.disk(climbability_cell_radius + 1))
                ),
            )
            shore_mask = new_world.map.get_outline(new_world.map.get_land_mask(), 4)
            near_spawn_climbable = np.logical_and(near_spawn_climbable, shore_mask)
            near_spawn_climbable = morphology.dilation(near_spawn_climbable, morphology.disk(climbability_cell_radius))
            is_climbable_fixed[near_spawn_climbable] = True

        # no matter what, dont mess up important climbability
        # logger.debug(is_accessible_from_water)
        if not is_accessible_from_water:
            is_climbable_fixed[new_world.is_detail_important] = new_world.is_climbable[new_world.is_detail_important]

        if new_world.is_debug_graph_printing_enabled:
            plot_value_grid(is_climbable_fixed, "no but for realz")

        new_world = attr.evolve(new_world, is_climbable=is_climbable_fixed)

        return new_world.reset_beaches(new_locations.island), new_locations

    def _get_spawn_zone(
        self, locations: WorldLocations, spawn_region: Optional[MapBoolNP] = None
    ) -> Optional[MapBoolNP]:
        if len(self.obstacle_zones) == 1:
            # find the closest one to the spawn
            zone_a, zone_b = only(self.obstacle_zones)
            if zone_a is None:
                spawn_region = zone_b
            elif zone_b is None:
                spawn_region = zone_a
            else:
                dist_sq_map = self.map.get_dist_sq_to(to_2d_point(locations.spawn))
                spawn_region = zone_a if dist_sq_map[zone_a].min() < dist_sq_map[zone_b].min() else zone_b
        return spawn_region

    def get_critical_distance(
        self,
        locations: WorldLocations,
        min_distance: float,
        desired_distance: Optional[float] = None,
    ) -> Tuple[Optional[float], str]:
        max_distance = locations.get_2d_spawn_goal_distance() - 2 * DEFAULT_SAFETY_RADIUS - min_distance
        if max_distance < 0.0:
            return None, "World is too small for even the smallest obstacle of this type"
        if desired_distance is None:
            return max_distance, ""
        if max_distance < desired_distance:
            return max_distance, (
                f"The requested obstalce is too far for the space available: {desired_distance} jump does not fit "
                f"because goal and spawn are {locations.get_2d_spawn_goal_distance()} away (and there needs to be "
                f"room for margins)"
            )
        return desired_distance, ""

    def mask_flora(self, mask: MapFloatNP) -> "World":
        return attr.evolve(self, flora_mask=self.flora_mask * mask)

    def make_natural_biomes(
        self, rand: np.random.Generator, is_debug_graph_printing_enabled: bool = False
    ) -> Tuple[BiomeMap, "World"]:
        biome_config = self.biome_config
        special_biomes = self.special_biomes

        biome_map = make_biome(
            self.map,
            biome_config,
            rand,
            special_biomes,
            is_debug_graph_printing_enabled=is_debug_graph_printing_enabled,
        )

        if not biome_config.is_scenery_added:
            return biome_map, self

        global_foliage_noise = perlin(
            self.map.Z.shape,
            biome_config.global_foliage_noise_scale,
            rand,
            is_normalized=True,
            noise_min=biome_config.global_foliage_noise_min,
        )

        # plot_value_grid(global_foliage_noise, "global foliage noise")

        want_to_plot_noise = False
        if want_to_plot_noise:
            plot_value_grid(self.make_biome_noise(rand, "", 0.0), "example biome foliage noise")

        want_overall_plot = is_debug_graph_printing_enabled
        if want_overall_plot:
            plot_value_grid(self.map.Z)
            plot_biome_grid(biome_map.biome_id, biome_config, "final biome map")

        # apply the flora mask from the world so we can avoid sticking trees in buildings, etc
        global_foliage_noise *= self.flora_mask

        if want_overall_plot:
            # if True:
            plot_value_grid(global_foliage_noise)

        overall_density = biome_config.foliage_density_modifier

        # TESTING
        FOLIAGE_ON = True
        ANIMALS_ON = False
        INTERACTIVE_ON = False

        animal_items = (
            [
                ("res://entities/animals/predators/snake_170f.tscn", 0.001),
                ("res://entities/animals/predators/bee_218f.tscn", 0.001),
                ("res://entities/animals/prey/mouse_146f.tscn", 0.001),
                ("res://entities/animals/prey/rabbit_112f.tscn", 0.001),
                ("res://entities/animals/prey/crow_190f.tscn", 0.001),
                ("res://entities/animals/prey/crow_flying_182f.tscn", 0.001),
                ("res://entities/animals/prey/eagle_162f.tscn", 0.001),
                ("res://entities/animals/prey/eagle_flying_248f.tscn", 0.001),
                ("res://entities/animals/prey/hawk_304f.tscn", 0.001),
                ("res://entities/animals/prey/hawk_flying_370f.tscn", 0.001),
                ("res://entities/animals/prey/pigeon_178f.tscn", 0.001),
                ("res://entities/animals/prey/pigeon_flying_258f.tscn", 0.001),
                ("res://entities/animals/predators/snake_170f.tscn", 0.001),
                ("res://entities/animals/prey/squirrel_214f.tscn", 0.001),
                ("res://entities/animals/prey/turtle_184f.tscn", 0.001),
                ("res://entities/animals/predators/alligator_270f.tscn", 0.001),
                ("res://entities/animals/predators/hippo_204f.tscn", 0.001),
                ("res://entities/animals/prey/deer_250f.tscn", 0.001),
                ("res://entities/animals/prey/mouse_146f.tscn", 0.001),
                ("res://entities/animals/prey/frog_280f.tscn", 0.001),
                ("res://entities/animals/predators/wolf_246f.tscn", 0.001),
                ("res://entities/animals/predators/jaguar_348f.tscn", 0.001),
                ("res://entities/animals/predators/bear_230f.tscn", 0.001),
            ]
            if ANIMALS_ON
            else []
        )

        interactive_items = (
            [
                ("res://items/stone/boulder.tscn", 0.0005),
                ("res://items/stone/stone.tscn", 0.0005),
                ("res://items/stone/rock.tscn", 0.003),
                ("res://items/wood/log.tscn", 0.001),
                ("res://items/wood/stick.tscn", 0.002),
            ]
            if INTERACTIVE_ON
            else []
        )

        scenery_items: List[Scenery] = []

        # lowest next to coastal
        biome_id = 5
        biome_items = (
            animal_items
            + interactive_items
            + [
                ("res://scenery/bush.tscn", 0.001),
                ("res://scenery/trees/palm.tscn", 0.004),
                ("res://scenery/flower_pink.tscn", 0.005),
            ]
            if FOLIAGE_ON
            else []
        )
        for item in biome_items:
            file = item[0]
            density = item[1]
            scenery_items.extend(
                biome_map.get_random_points_in_biome(
                    rand,
                    config=SceneryConfig(
                        resource_file=file,
                        biome_id=biome_id,
                        density=density * overall_density,
                        border_distance=0.0,
                        border_mode=SceneryBorderMode.HARD,
                    ),
                    placement_noise=global_foliage_noise * self.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
                    is_debug_graph_printing_enabled=False,
                )
            )

        # low dirt
        biome_id = 6
        biome_items = (
            animal_items
            + interactive_items
            + [
                ("res://scenery/bush.tscn", 0.0015),
                ("res://scenery/trees/acacia.tscn", 0.003),
                ("res://scenery/flower_blue.tscn", 0.005),
                ("res://scenery/flower_pink.tscn", 0.005),
                ("res://scenery/flower_yellow.tscn", 0.005),
            ]
            if FOLIAGE_ON
            else []
        )
        for item in biome_items:
            file = item[0]
            density = item[1]
            scenery_items.extend(
                biome_map.get_random_points_in_biome(
                    rand,
                    config=SceneryConfig(
                        resource_file=file,
                        biome_id=biome_id,
                        density=density * overall_density,
                        border_distance=0.0,
                        border_mode=SceneryBorderMode.HARD,
                    ),
                    placement_noise=global_foliage_noise * self.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
                    is_debug_graph_printing_enabled=False,
                )
            )

        # second lowest
        biome_id = 7
        biome_items = (
            animal_items
            + interactive_items
            + [
                ("res://scenery/bush.tscn", 0.0025),
                ("res://scenery/trees/acacia.tscn", 0.004),
                ("res://scenery/flower_blue.tscn", 0.005),
                ("res://scenery/flower_pink.tscn", 0.005),
                ("res://scenery/flower_yellow.tscn", 0.005),
            ]
            if FOLIAGE_ON
            else []
        )
        for item in biome_items:
            file = item[0]
            density = item[1]
            scenery_items.extend(
                biome_map.get_random_points_in_biome(
                    rand,
                    config=SceneryConfig(
                        resource_file=file,
                        biome_id=biome_id,
                        density=density * overall_density,
                        border_distance=0.0,
                        border_mode=SceneryBorderMode.HARD,
                    ),
                    placement_noise=global_foliage_noise * self.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
                    is_debug_graph_printing_enabled=False,
                )
            )

        # third lowest
        biome_id = 8
        biome_items = (
            animal_items
            + interactive_items
            + [
                ("res://scenery/bush.tscn", 0.0025),
                ("res://scenery/trees/fir.tscn", 0.006),
                ("res://scenery/flower_blue.tscn", 0.003),
                ("res://scenery/flower_pink.tscn", 0.003),
                ("res://scenery/flower_yellow.tscn", 0.003),
            ]
            if FOLIAGE_ON
            else []
        )
        for item in biome_items:
            file = item[0]
            density = item[1]
            scenery_items.extend(
                biome_map.get_random_points_in_biome(
                    rand,
                    config=SceneryConfig(
                        resource_file=file,
                        biome_id=biome_id,
                        density=density * overall_density,
                        border_distance=0.0,
                        border_mode=SceneryBorderMode.HARD,
                    ),
                    placement_noise=global_foliage_noise * self.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
                    is_debug_graph_printing_enabled=False,
                )
            )

        # third highest
        biome_id = 9
        biome_items = (
            animal_items
            + interactive_items
            + [
                ("res://scenery/bush.tscn", 0.0025),
                ("res://scenery/trees/fir.tscn", 0.0004),
                ("res://scenery/trees/maple_red.tscn", 0.0006),
                ("res://scenery/trees/maple_orange.tscn", 0.0006),
                ("res://scenery/trees/maple_yellow.tscn", 0.0006),
                ("res://scenery/mushroom.tscn", 0.005),
            ]
            if FOLIAGE_ON
            else []
        )
        for item in biome_items:
            file = item[0]
            density = item[1]
            scenery_items.extend(
                biome_map.get_random_points_in_biome(
                    rand,
                    config=SceneryConfig(
                        resource_file=file,
                        biome_id=biome_id,
                        density=density * overall_density,
                        border_distance=0.0,
                        border_mode=SceneryBorderMode.HARD,
                    ),
                    placement_noise=global_foliage_noise * self.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
                    is_debug_graph_printing_enabled=False,
                )
            )

        # second highest
        biome_id = 10
        biome_items = (
            animal_items
            + interactive_items
            + [
                ("res://scenery/bush.tscn", 0.01),
                ("res://scenery/trees/fir.tscn", 0.0002),
                ("res://scenery/trees/maple_red.tscn", 0.00055),
                ("res://scenery/trees/maple_orange.tscn", 0.00055),
                ("res://scenery/trees/maple_yellow.tscn", 0.00055),
                ("res://scenery/mushroom.tscn", 0.001),
            ]
            if FOLIAGE_ON
            else []
        )
        for item in biome_items:
            file = item[0]
            density = item[1]
            scenery_items.extend(
                biome_map.get_random_points_in_biome(
                    rand,
                    config=SceneryConfig(
                        resource_file=file,
                        biome_id=biome_id,
                        density=density * overall_density,
                        border_distance=0.0,
                        border_mode=SceneryBorderMode.HARD,
                    ),
                    placement_noise=global_foliage_noise * self.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
                    is_debug_graph_printing_enabled=False,
                )
            )

        # highest, peak
        biome_id = 11
        biome_items = (
            animal_items
            + interactive_items
            + [
                ("res://scenery/bush.tscn", 0.01),
                ("res://scenery/trees/maple_red.tscn", 0.0004),
                ("res://scenery/trees/maple_orange.tscn", 0.0004),
                ("res://scenery/trees/maple_yellow.tscn", 0.0004),
                ("res://scenery/mushroom.tscn", 0.0005),
                ("res://scenery/flower_yellow.tscn", 0.0009),
            ]
            if FOLIAGE_ON
            else []
        )
        for item in biome_items:
            file = item[0]
            density = item[1]
            scenery_items.extend(
                biome_map.get_random_points_in_biome(
                    rand,
                    config=SceneryConfig(
                        resource_file=file,
                        biome_id=biome_id,
                        density=density * overall_density,
                        border_distance=0.0,
                        border_mode=SceneryBorderMode.HARD,
                    ),
                    placement_noise=global_foliage_noise * self.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
                    is_debug_graph_printing_enabled=False,
                )
            )

        new_items: Tuple[Entity, ...] = tuple([*self.items, *scenery_items])

        if self.export_config:
            if self.export_config.scenery_mode == "single":
                fixed_items: List[Entity] = []
                for i in range(len(new_items)):
                    new_item = new_items[i]
                    if isinstance(new_item, Scenery):
                        if "trees/" in new_item.resource_file:
                            fixed_items.append(attr.evolve(new_item, resource_file="res://scenery/trees/fir.tscn"))
                        else:
                            pass
                    else:
                        fixed_items.append(new_item)
                new_items = tuple(fixed_items)
            if self.export_config.scenery_mode == "tree":
                fixed_items = []
                for i in range(len(new_items)):
                    new_item = new_items[i]
                    if isinstance(new_item, Scenery):
                        if "trees/" in new_item.resource_file:
                            fixed_items.append(new_item)
                        else:
                            pass
                    else:
                        fixed_items.append(new_item)
                new_items = tuple(fixed_items)

        return biome_map, attr.evolve(self, items=new_items)


def get_spawn(
    rand: np.random.Generator,
    difficulty: float,
    spawn_location: np.ndarray,
    target_location: np.ndarray,
):
    direction_to_target = normalized(target_location - spawn_location)
    # noinspection PyTypeChecker
    target_yaw = np.angle(complex(direction_to_target[2], direction_to_target[0]), deg=True) + 180
    spawn_view_yaw = (target_yaw + rand.uniform(-180, 180) * difficulty) % 360  # type: ignore
    # noinspection PyTypeChecker

    dist_to_target_2d = np.linalg.norm(
        np.array([target_location[0], target_location[2]]) - np.array([spawn_location[0], spawn_location[2]])
    )
    eye_location = spawn_location + HALF_AGENT_HEIGHT_VECTOR
    height_dist_to_target = target_location[1] - eye_location[1]
    target_pitch = np.arctan2(height_dist_to_target, dist_to_target_2d) * 180.0 / np.pi
    spawn_view_pitch = np.clip(target_pitch + difficulty * rand.uniform(-60, 60), -70, 70)
    return SpawnPoint(
        position=spawn_location,
        yaw=spawn_view_yaw,
        pitch=float(spawn_view_pitch),
    )


def get_random_positions_along_path(
    visible_locations: Point3DListNP,
    start: Point3DNP,
    end: Point3DNP,
    target_location_distribution: stats.rv_continuous,
    rand: np.random.Generator,
    count: int,
) -> Point3DListNP:
    """Difficulty scales with how far away the point is from the straight line path between start and end"""
    path_length = float(np.linalg.norm(start - end))
    start_point = (start[0], start[2])
    end_point = (end[0], end[2])
    location_weights = np.array(
        [
            target_location_distribution.pdf(signed_line_distance((x[0], x[2]), start_point, end_point, path_length))
            for x in visible_locations
        ]
    )
    location_weights /= location_weights.sum()
    return rand.choice(visible_locations, p=location_weights, size=count)


def get_difficulty_based_value(
    difficulty: float, min_val: float, max_val: float, variability: float, rand: np.random.Generator
) -> float:
    total_delta = max_val - min_val
    delta = variability * total_delta
    remainder = total_delta - delta
    return min_val + (remainder * difficulty) + (rand.uniform() * delta)


def build_building(config: BuildingConfig, building_id: int, rand: np.random.Generator) -> Building:
    stories: List[Story] = []
    for story_num in range(config.story_count):
        footprint_below = None
        if story_num != 0:
            footprint_below = stories[story_num - 1].footprint.copy()
        footprint = config.footprint_builder.build(
            config.width, config.length, story_num, rand, footprint_below=footprint_below
        )

        rooms = config.room_builder.build(footprint, rand)
        hallways = config.hallway_builder.build(rooms, rand)
        story = Story(
            story_num,
            width=config.width,
            length=config.length,
            footprint=footprint,
            rooms=rooms,
            hallways=hallways,
            has_ceiling=True,
        )
        stories.append(story)

    if config.story_linker is not None:
        links = config.story_linker.link_stories(stories, rand)
        for link in links:
            stories[link.bottom_story_id].story_links.append(link)
            stories[link.top_story_id].story_links.append(link)
    else:
        links = []

    if config.entrance_builder is not None:
        stories_with_entrances = config.entrance_builder.build(stories, rand)
    else:
        stories_with_entrances = stories

    if config.window_builder is not None:
        stories_with_windows = config.window_builder.build(stories_with_entrances, rand, config.aesthetics)
    else:
        stories_with_windows = stories_with_entrances

    return Building(
        id=building_id,
        position=np.array([0, 0, 0]),
        stories=stories_with_windows,
        story_links=links,
        is_climbable=config.is_climbable,
    )


def _get_angle_value_for_difficulty(target_yaw: float, difficulty: float, rand: np.random.Generator):
    possible_yaws = range(0, 360, 10)
    difficulty_degrees = difficulty * 180
    difficulty_adjusted_target_yaw = round(target_yaw + rand.choice([1, -1]) * difficulty_degrees) % 360
    yaw_distribution = stats.norm(difficulty_adjusted_target_yaw, 5)
    angle_weights = np.array([yaw_distribution.pdf(x) for x in possible_yaws])
    angle_weights /= angle_weights.sum()
    spawn_view_yaw = rand.choice(possible_yaws, p=angle_weights)
    return spawn_view_yaw


def _get_spawn(
    rand: np.random.Generator,
    difficulty: float,
    spawn_location: np.ndarray,
    target_location: np.ndarray,
):
    direction_to_target = normalized(target_location - spawn_location)

    # noinspection PyTypeChecker
    target_yaw = np.angle(complex(direction_to_target[2], direction_to_target[0]), deg=True) + 180
    spawn_view_yaw = _get_angle_value_for_difficulty(target_yaw, difficulty, rand)  # type: ignore

    dist_to_target_2d = np.linalg.norm(
        np.array([target_location[0], target_location[2]]) - np.array([spawn_location[0], spawn_location[2]])
    )
    eye_location = spawn_location + HALF_AGENT_HEIGHT_VECTOR
    height_dist_to_target = target_location[1] - eye_location[1]
    target_pitch = np.arctan2(height_dist_to_target, dist_to_target_2d) * 180.0 / np.pi
    pitch_variance = 100
    spawn_view_pitch = np.clip(target_pitch + difficulty * pitch_variance * rand.uniform(), -70, 70)
    return SpawnPoint(
        position=spawn_location,
        yaw=spawn_view_yaw,
        pitch=float(spawn_view_pitch),
    )


def random_point_within_radius(rand: np.random.Generator, point: Point2DNP, radius: float) -> Point2DNP:
    assert_isinstance(point, Point2DNP)
    r = rand.uniform() * radius
    theta = rand.uniform() * np.pi * 2.0
    x = math.cos(theta) * r
    y = math.sin(theta) * r
    return np.array([x + point[0], y + point[1]])
