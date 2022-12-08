from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import cast

import attr
import numpy as np
from IPython.display import HTML
from IPython.display import display
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation
from skimage import morphology

from avalon.common.errors import SwitchError
from avalon.common.utils import only
from avalon.datagen.world_creation.configs.biome import BiomeConfig
from avalon.datagen.world_creation.configs.biome import _raw_biome_id_mapping
from avalon.datagen.world_creation.configs.scenery import SceneryConfig
from avalon.datagen.world_creation.constants import DARK_SHORE
from avalon.datagen.world_creation.constants import FRESH_WATER
from avalon.datagen.world_creation.constants import MAX_CLIFF_TERRAIN_ANGLE
from avalon.datagen.world_creation.constants import MIN_CLIFF_TERRAIN_ANGLE
from avalon.datagen.world_creation.constants import SWAMP
from avalon.datagen.world_creation.constants import UNCLIMBABLE_BIOME_ID
from avalon.datagen.world_creation.constants import UP_VECTOR
from avalon.datagen.world_creation.constants import WATER_LINE
from avalon.datagen.world_creation.debug_plots import plot_value_grid
from avalon.datagen.world_creation.entities.scenery import Scenery
from avalon.datagen.world_creation.noise import perlin
from avalon.datagen.world_creation.types import Biome
from avalon.datagen.world_creation.types import FloatListNP
from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.types import MapFloatNP
from avalon.datagen.world_creation.types import MapIntNP
from avalon.datagen.world_creation.types import Point2DListNP
from avalon.datagen.world_creation.types import Point3DListNP
from avalon.datagen.world_creation.types import SceneryBorderMode
from avalon.datagen.world_creation.utils import hex_to_rgb
from avalon.datagen.world_creation.worlds.height_map import HeightMap
from avalon.datagen.world_creation.worlds.height_map import SpecialBiomes
from avalon.datagen.world_creation.worlds.height_map import create_kd_tree
from avalon.datagen.world_creation.worlds.height_map import get_border_points
from avalon.datagen.world_creation.worlds.height_map import get_flora_config_by_file


def sidewaysness(normals: Point3DListNP) -> FloatListNP:
    xzlen = np.sqrt(normals[:, 0] ** 2 + normals[:, 2] ** 2)
    angle = np.abs(np.arctan2(normals[:, 1], xzlen))
    result = np.clip(((MAX_CLIFF_TERRAIN_ANGLE - angle)) / (MAX_CLIFF_TERRAIN_ANGLE - MIN_CLIFF_TERRAIN_ANGLE), 0, 1)
    result[angle < MIN_CLIFF_TERRAIN_ANGLE] = 1.0
    return cast(FloatListNP, result)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class BiomeMap:
    map: HeightMap
    biomes: Tuple[Biome, ...]
    biome_id: MapIntNP
    border_distances: MapFloatNP
    config: BiomeConfig

    def get_colors(
        self, rand: np.random.Generator, is_climbable: MapBoolNP, points: Point3DListNP, normals: Point3DListNP
    ) -> np.ndarray:
        grid_x, grid_y = self.map.region.get_linspace(self.map.cells_per_meter)
        biome_id_with_unclimb = self.biome_id.copy()
        is_steep = self.map.get_squared_slope() > self.config.high_slope_threshold
        biome_id_with_unclimb[np.logical_and(is_steep, np.logical_not(is_climbable))] = UNCLIMBABLE_BIOME_ID
        interp_spline = RegularGridInterpolator((grid_x, grid_y), biome_id_with_unclimb, method="nearest")
        points_2d = np.stack([points[:, 2], points[:, 0]], axis=1)
        biome_ids = interp_spline(points_2d)

        color_mapping = self.config.color_map
        neighbor_count = self.config.color_sampling_neighbor_count
        if neighbor_count == 0:
            colors = np.array([color_mapping[x] for x in biome_ids])
        else:
            min_val = self.map.region.x.min_ge
            max_val = self.map.region.x.max_lt
            neighbors_2d = points_2d.reshape((len(points_2d), 1, 2)).repeat(neighbor_count, 1)
            neighbors_2d += rand.standard_normal(neighbors_2d.shape) * self.config.color_sampling_std_dev
            neighbors_2d = neighbors_2d.reshape((len(neighbors_2d) * neighbor_count, 2))
            neighbors_2d = np.clip(neighbors_2d, min_val, max_val)
            neighbor_biome_ids = interp_spline(neighbors_2d)
            colors = np.array([color_mapping[x] for x in neighbor_biome_ids])
            colors = colors.reshape((len(points), neighbor_count, 3)).mean(axis=1)

        # coast is special
        is_coast = biome_ids == self.config.coastal_id
        colors[is_coast] = (
            np.array(color_mapping[self.config.coastal_id]) * (1.0 - self.config.coast_color_noise)
            + self.config.coast_color_noise * colors[is_coast]
        )

        # rocks are special too
        for rock_id in self.config.rock_biome_ids:
            is_rock = biome_ids == rock_id
            colors[is_rock] = (
                np.array(color_mapping[rock_id]) * (1.0 - self.config.rock_color_noise)
                + self.config.rock_color_noise * colors[is_rock]
            )

        unclimbable_color = color_mapping[UNCLIMBABLE_BIOME_ID]
        climb_spline = RegularGridInterpolator((grid_x, grid_y), is_climbable)
        is_climbable_factor = climb_spline(points_2d)
        unclimbability = ((1.0 - is_climbable_factor) * sidewaysness(normals)).reshape(-1, 1)
        colors = unclimbable_color * unclimbability + colors * (1.0 - unclimbability)

        # apply correlated color noise
        d = {x.id: x.correlated_color_jitter for x in self.biomes}
        u, inv = np.unique(biome_ids, return_inverse=True)
        jitters = np.array([d[x] for x in u])[inv].reshape(biome_ids.shape)
        jitters = jitters.reshape(*jitters.shape, 1)
        colors = np.clip(colors * 1.0 + rand.normal(0.0, jitters, jitters.shape).repeat(3, axis=-1), 0.0, 1.0)

        # apply uncorrelated color noise
        d = {x.id: x.color_jitter for x in self.biomes}
        u, inv = np.unique(biome_ids, return_inverse=True)
        jitters = np.array([d[x] for x in u])[inv].reshape(biome_ids.shape)
        jitters = jitters.reshape(*jitters.shape, 1).repeat(3, axis=-1)
        return np.clip(colors + rand.normal(0.0, jitters, colors.shape), 0.0, 1.0)  # type: ignore

    def copy(self) -> "BiomeMap":
        return BiomeMap(
            map=self.map.copy(),
            biome_id=self.biome_id.copy(),
            border_distances=self.border_distances.copy(),
            biomes=self.biomes,
            config=self.config,
        )

    def _resolve_desired_point_count(
        self,
        density: float,
        biome_weights: np.ndarray,
    ) -> int:
        square_meters_per_cell = (1.0 / self.map.cells_per_meter) ** 2
        square_meters = biome_weights.sum() * square_meters_per_cell
        desired_point_count = round(square_meters * density)
        return desired_point_count

    def get_random_points(
        self,
        rand: np.random.Generator,
        desired_point_count: int,
        biome_weights: np.ndarray,
        is_debug_graph_printing_enabled: bool = False,
    ) -> np.ndarray:
        # because of the way we randomize below, we just ignore the final colums
        biome_weights = biome_weights.copy()
        biome_weights[:, -1] = 0.0
        biome_weights[-1, :] = 0.0

        # TODO: how we efficiently grab a bunch of random points really depends on
        #  how many points we want, how big the region is, etc. If we're trying to get
        #  multiple points per grid cell, we're likely better off with a different approach
        cell_locations = np.argwhere(biome_weights)
        selected_cells = rand.choice(
            cell_locations,
            p=biome_weights[np.nonzero(biome_weights)] / biome_weights.sum(),
            size=desired_point_count,
            shuffle=False,
        )

        # TODO: actually, just set ones wherever there is anything
        if is_debug_graph_printing_enabled:
            item_placements = np.zeros_like(biome_weights)
            for location in selected_cells:
                item_placements[tuple(location)] += 1
            plot_value_grid(item_placements, "biome item placements")

        points_2d = np.zeros_like(selected_cells, dtype=float)
        cell_size = 1.0 / self.map.cells_per_meter
        points_2d[:, 1] = (
            self.map.region.x.min_ge
            + selected_cells[:, 0] * cell_size
            + rand.uniform(0, cell_size, (desired_point_count))
        )
        points_2d[:, 0] = (
            self.map.region.z.min_ge
            + selected_cells[:, 1] * cell_size
            + rand.uniform(0, cell_size, (desired_point_count))
        )
        heights = self.map.get_heights(points_2d)
        return np.stack([points_2d[:, 0], heights, points_2d[:, 1]], axis=1)

    def get_random_points_in_biome(
        self,
        rand: np.random.Generator,
        config: SceneryConfig,
        placement_noise: Optional[MapFloatNP] = None,
        is_debug_graph_printing_enabled: bool = False,
        custom_weights: Optional[MapFloatNP] = None,
    ) -> List[Scenery]:
        if not self.config.is_scenery_added:
            return []
        if custom_weights is None:
            weights = np.ones_like(self.biome_id, dtype=np.float32)
            weights[self.biome_id != config.biome_id] = 0.0
            weights *= self.border_distances
            if config.border_mode == SceneryBorderMode.HARD:
                weights[weights < config.border_distance] = 0.0
                weights[weights > 0.0] = 1.0
            elif config.border_mode == SceneryBorderMode.LINEAR:
                weights = np.clip(weights / config.border_distance, 0, 1)
            elif config.border_mode == SceneryBorderMode.SQUARED:
                weights = np.clip(weights / config.border_distance, 0, 1) ** 2
            else:
                raise SwitchError("Mode not implemented yet: " + str(config.border_mode))
        else:
            weights = custom_weights

        if placement_noise is not None:
            weights *= placement_noise

        if is_debug_graph_printing_enabled:
            plot_value_grid(weights, "biome item placement weights")

        desired_point_count = self._resolve_desired_point_count(config.density, weights)
        if desired_point_count == 0:
            return []

        points = self.get_random_points(rand, desired_point_count, weights, is_debug_graph_printing_enabled)

        scales = np.ones((desired_point_count, 3), dtype=np.float32)
        correlated_scale_noise = rand.uniform(
            -config.correlated_scale_range, config.correlated_scale_range, size=(desired_point_count, 1)
        ).repeat(3, axis=1)
        uncorrelated_scale_noise = rand.uniform(
            -config.skewed_scale_range, config.skewed_scale_range, (desired_point_count, 3)
        )
        scales = scales + correlated_scale_noise + uncorrelated_scale_noise

        # bump things down into the ground, must consider scale as well
        flora_config, resource_name = get_flora_config_by_file(self.config.flora_config, config.resource_file)
        if flora_config is not None:
            points[:, 1] += flora_config.height_offset * scales[:, 1] * flora_config.default_scale

        results = []

        if config.is_oriented_to_surface:
            normals = self.map.get_normals(points)
            for point, scale, normal in zip(points, scales, normals):
                rotation = rotation_matrix_from_vectors(normal).flatten()
                results.append(
                    Scenery(resource_file=config.resource_file, position=point, rotation=rotation, scale=scale)
                )
        else:
            rotation_indices = rand.integers(0, 360, (desired_point_count))
            for point, scale, rotation_index in zip(points, scales, rotation_indices):
                rotation = _cached_rotations[rotation_index]
                results.append(
                    Scenery(resource_file=config.resource_file, position=point, rotation=rotation, scale=scale)
                )

        return results

    def create_extra_height_points(
        self,
        rand: np.random.Generator,
        is_climbable: MapBoolNP,
        is_detail_important: MapBoolNP,
    ) -> Point2DListNP:
        meters_per_cell = 1.0 / self.map.cells_per_meter
        square_meters_per_cell = meters_per_cell**2
        # points_per_cell = round(point_density_in_points_per_square_meter * square_meters_per_cell)
        # if points_per_cell == 0:
        #     points_per_cell = 1

        is_detail_important = morphology.dilation(is_detail_important, morphology.disk(2))
        # plot_value_grid(is_detail_important)

        points_2d = np.stack([self.map.X, self.map.Y], axis=2)
        base_points = points_2d[is_detail_important].reshape((-1, 2))
        noise_scale = meters_per_cell * 0.2
        half_point = np.array([meters_per_cell / 2.0, meters_per_cell / 2.0])
        noised_important_points = np.concatenate(
            [
                base_points + rand.uniform(-noise_scale, noise_scale, base_points.shape),
                base_points + rand.uniform(-noise_scale, noise_scale, base_points.shape) + half_point,
            ],
            axis=0,
        )

        # create extra points where it is very steep as well
        squared_slope = self.map.get_squared_slope()
        # plot_value_grid(squared_slope >= 16)
        # plot_value_grid(squared_slope >= 9)
        # plot_value_grid(squared_slope >= 4)
        # plot_value_grid(squared_slope >= 1)
        # this actually works out pretty reasonably, perhaps could use a bit more tuning
        steep_points = squared_slope >= 4

        # we could select more or fewer points here, but just selecting them all seems to work out for nwo
        unclimbable_mask = np.logical_or(steep_points, np.logical_not(is_climbable))
        unclimbable_points = points_2d[np.logical_and(unclimbable_mask, np.logical_not(is_detail_important))].reshape(
            (-1, 2)
        )
        rand.shuffle(unclimbable_points, axis=0)
        unclimbable_point_count = round(len(unclimbable_points) * 0.2)
        selected_unclimbable_points = unclimbable_points[:unclimbable_point_count]
        noise_scale = meters_per_cell * 0.3
        noised_unclimbable_points = selected_unclimbable_points + rand.uniform(
            -noise_scale, noise_scale, selected_unclimbable_points.shape
        )

        result = np.concatenate([noised_important_points, noised_unclimbable_points], axis=0)

        result[:, 0] = np.clip(result[:, 0], self.map.region.x.min_ge, self.map.region.x.max_lt)
        result[:, 1] = np.clip(result[:, 1], self.map.region.z.min_ge, self.map.region.z.max_lt)
        return cast(Point2DListNP, result)

    def plot(self) -> None:
        self.map.plot()


_IDENTITY = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
_IDENTITY_MATRIX = np.eye(3)


# TODO: could be vectorized
def rotation_matrix_from_vectors(b: np.ndarray):
    a = UP_VECTOR
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = _IDENTITY_MATRIX + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


_cached_rotations = [Rotation.from_euler("y", i, degrees=True).as_matrix().flatten() for i in range(360)]


def make_fast_biome(map: HeightMap, natural_biome_config: BiomeConfig):
    border_distances = np.ones_like(map.Z)
    biome_id = np.ones(map.Z.shape, dtype=np.int8)
    coast_biome_id = 1
    biomes = tuple(
        [
            Biome(coast_biome_id, "coastal", "#FFF18E", color_jitter=0.02),
            Biome(UNCLIMBABLE_BIOME_ID, "unclimbable", "#FF0000", color_jitter=0.02),
        ]
    )

    config = BiomeConfig(
        low_slope_threshold=1.0,
        high_slope_threshold=4.0,
        min_elevation=0.0,
        max_elevation=0.2,
        min_dryness=0.3,
        max_dryness=1.0,
        water_cutoff=0.9,
        noise_scale=0.01,
        noise_magnitude=1.0,
        is_color_per_vertex=False,
        is_normal_per_vertex=False,
        color_map={x.id: hex_to_rgb(x.color) for x in biomes},
        biome_matrix=_raw_biome_id_mapping,
        biomes=biomes,
        flora_config=natural_biome_config.flora_config,
        godot_sky_config=natural_biome_config.godot_sky_config,
        godot_sun_config=natural_biome_config.godot_sun_config,
        godot_env_config=_disable_env_settings(natural_biome_config.godot_env_config),
        coastal_id=coast_biome_id,
        rock_biome_ids=tuple(),
        color_sampling_neighbor_count=0,
        is_scenery_added=natural_biome_config.is_scenery_added,
    )
    return BiomeMap(map, biomes, biome_id, border_distances, config)


def _disable_env_settings(env_settings: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: (v if (not isinstance(v, bool) or not k.endswith("_enabled")) else False) for k, v in env_settings.items()
    }


def make_biome(
    map: HeightMap,
    config: BiomeConfig,
    rand: np.random.Generator,
    special_biomes: Optional[SpecialBiomes] = None,
    is_debug_graph_printing_enabled: bool = False,
) -> BiomeMap:
    water_distance = map.get_water_distance(
        rand, config.is_fresh_water_included_in_moisture, max_points=config.max_kd_points
    )
    if water_distance is None:
        moisture = np.zeros_like(map.Z)
        water_distance = np.full_like(map.Z, np.inf)
    else:
        moisture = 1.0 - (water_distance / water_distance.max(initial=0.001))

    if is_debug_graph_printing_enabled:
        height_with_water = np.clip(map.Z, WATER_LINE, map.Z.max(initial=1.0))
        plot_value_grid(height_with_water, "height map")
        plot_value_grid(moisture, "moisture map")

    elevation = map.get_elevation()
    min_elevation = WATER_LINE
    normalized_elevation = (elevation - min_elevation) / (elevation.max(initial=0.001) - min_elevation)
    elevation_range = config.max_elevation - config.min_elevation
    normalized_elevation = normalized_elevation * elevation_range + config.min_elevation

    dryness = 1.0 - moisture.copy()
    noise_data = perlin(elevation.shape, config.noise_scale, rand)
    dryness = np.clip(dryness * (1 + noise_data * config.noise_magnitude), 0, 1)
    _clamp_water(dryness, moisture, config)

    squared_slope = map.get_squared_slope()

    biome_id = _map_to_biome_id(
        normalized_elevation,
        dryness,
        moisture,
        elevation,
        water_distance,
        squared_slope,
        config,
        special_biomes=special_biomes,
    )

    border_distances = _get_border_distances(rand, biome_id, map.cells_per_meter, config.max_kd_points)

    biome_map = BiomeMap(map, config.biomes, biome_id, border_distances, config)

    if is_debug_graph_printing_enabled:
        raw_dryness = 1.0 - moisture.copy()
        _clamp_water(raw_dryness, moisture, config)
        biome_id_without_noise = _map_to_biome_id(
            normalized_elevation,
            raw_dryness,
            moisture,
            elevation,
            water_distance,
            squared_slope,
            config,
            special_biomes=special_biomes,
        )
        plot_biome_grid(biome_id_without_noise, config, "biome map (without noise)")

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        ax.set_title("noise texture")
        plt.imshow(noise_data, origin="lower")
        plt.show()
        fig.clf()

        plot_biome_grid(biome_map.biome_id, config, "biome map (with noise)")

        plot_value_grid(border_distances, "distance from biome edge")

        plot_biome_grid(config.biome_matrix[0], config, "low steepness biome mapping by dryness (X) and elevation (Y)")
        plot_biome_grid(
            config.biome_matrix[1], config, "medium steepness biome mapping by dryness (X) and elevation (Y)"
        )
        plot_biome_grid(
            config.biome_matrix[2], config, "high steepness biome mapping by dryness (X) and elevation (Y)"
        )
        print_biome_legend(config)

    return biome_map


def _clamp_water(dryness: MapFloatNP, moisture: MapFloatNP, config: BiomeConfig) -> None:
    water_cutoff = config.water_cutoff
    dryness[moisture < water_cutoff] = np.clip(
        dryness[moisture < water_cutoff], config.min_dryness, config.max_dryness
    )
    dryness[moisture >= water_cutoff] = 0.0


def _get_border_distances(
    rand: np.random.Generator, biome_id: MapIntNP, cells_per_meter: float, max_points: Optional[int]
) -> MapFloatNP:
    border, border_points = get_border_points(biome_id, rand)

    distances = np.ones_like(biome_id, dtype=np.float32)
    if len(border_points) == 0:
        return distances

    distances[border] = 0.0
    tree = create_kd_tree(border_points, "borders", max_points=max_points, rand=rand)
    distances[np.nonzero(distances)] = tree.query(np.argwhere(distances), k=1)[0]

    return distances * (1.0 / cells_per_meter)


def _map_to_biome_id(
    normalized_elevation: MapFloatNP,
    dryness: MapFloatNP,
    moisture: MapFloatNP,
    elevation: MapFloatNP,
    water_distance: MapFloatNP,
    squared_slope: MapFloatNP,
    config: BiomeConfig,
    special_biomes: Optional[SpecialBiomes] = None,
) -> MapIntNP:
    elevation_bins = config.biome_matrix[0].shape[0]
    elevation_indices = np.clip((normalized_elevation * elevation_bins - 0.5).astype(int), 0, elevation_bins - 1)
    dryness_bins = config.biome_matrix[0].shape[1]
    dryness_indices = np.clip((dryness * dryness_bins - 0.5).astype(int), 0, dryness_bins - 1)

    low_mask = squared_slope < config.low_slope_threshold
    med_mask = np.logical_and(config.low_slope_threshold <= squared_slope, squared_slope < config.high_slope_threshold)
    high_mask = config.high_slope_threshold <= squared_slope
    biome_id = np.zeros_like(dryness_indices)
    for i, mask in enumerate([low_mask, med_mask, high_mask]):
        biome_id[mask] = config.biome_matrix[i][elevation_indices, dryness_indices][mask]

    if special_biomes:
        biome_id[special_biomes.swamp_mask] = SWAMP

    beach_mask = np.full_like(biome_id, False)
    if special_biomes:
        beach_mask = special_biomes.beach_mask
        biome_id[beach_mask] = 3

    # the dark sand region is everything within a pretty small distance of the beach line
    dark_beach_mask = np.logical_and(
        beach_mask,
        np.logical_and(water_distance < config.dark_shore_distance, elevation < WATER_LINE + config.dark_shore_height),
    )
    biome_id[dark_beach_mask] = DARK_SHORE

    # reset anything that is completely wet to be water
    biome_id[moisture == 1.0] = 2

    # set fresh water and swamp masks
    if special_biomes:
        biome_id[special_biomes.fresh_water_mask] = FRESH_WATER

    # expand and contract rocks so that they are not scattered
    rock_id = only(config.rock_biome_ids)
    base_rock_ids = biome_id == rock_id
    expander = morphology.disk(4)
    true_rock_ids = np.logical_and(
        base_rock_ids,
        morphology.dilation(
            np.logical_and(
                base_rock_ids,
                np.logical_not(morphology.dilation(np.logical_not(base_rock_ids), expander)),
            ),
            expander,
        ),
    )
    removed_rock_ids = np.logical_and(base_rock_ids, np.logical_not(true_rock_ids))
    biome_id[removed_rock_ids] = config.biome_matrix[0][elevation_indices, dryness_indices][removed_rock_ids]

    # and reset things that are just always cliffs
    cliffs = squared_slope > config.high_slope_threshold
    # cliffs = squared_slope > config.force_cliff_square_slope
    # cliffs = morphology.dilation(cliffs, morphology.disk(1))
    biome_id[cliffs] = rock_id

    # plot_value_grid(biome_id == rock_id, "ROCK IDS")

    return cast(MapIntNP, biome_id)


def plot_biome_grid(data: np.ndarray, config: BiomeConfig, title: str = "") -> None:
    data = np.array(data)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    if title:
        ax.set_title(title)

    image_data = []
    for row in data:
        new_row = []
        for value in row:
            new_row.append(config.color_map[value])
        image_data.append(new_row)

    image_data_np = np.array(image_data)
    plt.imshow(image_data_np, origin="lower")
    plt.show()


def print_biome_legend(config: BiomeConfig) -> None:
    display(HTML(f"<h1>Biomes:</h1>"))

    for biome in config.biomes:
        display(HTML(f'<h2 style="background: {biome.color}">{biome.name}</h2>'))
