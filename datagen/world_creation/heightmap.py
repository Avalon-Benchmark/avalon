import math
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import attr
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from nptyping import Bool
from nptyping import Double
from nptyping import Int32
from nptyping import NDArray
from nptyping import Shape
from nptyping import assert_isinstance
from scipy import stats
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import measurements
from scipy.spatial import KDTree
from skimage import morphology
from skimage.morphology import flood_fill

from common.errors import SwitchError
from contrib.serialization import Serializable
from datagen.world_creation.indoor.objects import Building
from datagen.world_creation.region import EdgedRegion
from datagen.world_creation.region import Region
from datagen.world_creation.types import RawGDObject
from datagen.world_creation.utils import IS_DEBUG_VIS
from datagen.world_creation.utils import plot_points
from datagen.world_creation.utils import plot_value_grid

WATER_LINE = 0.0

MapBoolNP = NDArray[Shape["MapY, MapX"], Bool]
MapFloatNP = NDArray[Shape["MapY, MapX"], Double]
MapIntNP = NDArray[Shape["MapY, MapX"], Int32]
FloatListNP = NDArray[Shape["ValueIndex"], Double]
Point2DNP = NDArray[Shape["2"], Double]
Point2DListNP = NDArray[Shape["PointIndex, 2"], Double]
Point3DNP = NDArray[Shape["3"], Double]
Point3DListNP = NDArray[Shape["PointIndex, 3"], Double]


class HeightMode(Enum):
    RELATIVE = "RELATIVE"
    ABSOLUTE = "ABSOLUTE"
    MIDPOINT_RELATIVE = "MIDPOINT_RELATIVE"
    MIDPOINT_ABSOLUTE = "MIDPOINT_ABSOLUTE"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class SpecialBiomes:
    beach_mask: MapBoolNP
    swamp_mask: MapBoolNP
    fresh_water_mask: MapBoolNP


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FloraConfig:
    # which 3D file to use for the geometry
    resource: str
    # whether or not to use the noise shader with perlin noise
    is_noise_shader_enabled: bool = False
    # whether this scenery should cast shadows
    is_shadowed: bool = True
    # the default scale for this 3D model
    default_scale: float = 1.0
    # how large the collision box is at the default scale. Only applies to trees
    collision_extents: np.ndarray = np.array([1.0, 5.0, 1.0])
    # how much to move the object in the vertical access when placing.
    # helps ensures that things are not floating off of highly sloped terrain
    # only applies to trees
    height_offset: float = 0.0


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Biome:
    id: int
    name: str
    color: str
    color_jitter: float = 0.02
    correlated_color_jitter: float = 0.2


def Color(r, g, b, a):
    return RawGDObject(f"Color( {r}, {g}, {b}, {a} )")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DebugVisualizationConfig:
    spawn_mode: str = "perspective"  # perspective, normal
    is_2d_graph_drawn: bool = False


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ExportConfig:
    name: str
    num_tiles: int
    is_terrain_shader_enabled: bool
    is_flora_shader_enabled: bool
    is_sun_enabled: bool
    is_sun_shadow_enabled: bool
    is_ssao_enabled: bool
    is_fog_enabled: bool
    is_blur_enabled: bool
    is_postprocessing_enabled: bool
    scenery_mode: str
    is_single_collision_mesh: bool
    is_biome_fast: bool
    is_exported_with_absolute_paths: bool
    is_border_calculation_detailed: bool
    is_meta_data_exported: bool
    is_biome_based_on_natural_height: bool
    world_id: Optional[str] = None
    debug_visualization_config: Optional[DebugVisualizationConfig] = None


def get_oculus_export_config(world_id: Optional[str] = None) -> ExportConfig:
    return ExportConfig(
        name="oculus",
        num_tiles=7,
        is_terrain_shader_enabled=False,
        is_flora_shader_enabled=False,
        is_sun_enabled=True,
        is_sun_shadow_enabled=True,
        is_ssao_enabled=True,
        is_fog_enabled=True,
        is_blur_enabled=False,
        is_postprocessing_enabled=True,
        scenery_mode="normal",
        is_single_collision_mesh=False,
        is_biome_fast=False,
        is_exported_with_absolute_paths=False,
        is_border_calculation_detailed=True,
        is_meta_data_exported=True,
        is_biome_based_on_natural_height=False,
        world_id=world_id,
    )


def get_agent_export_config() -> ExportConfig:
    return ExportConfig(
        name="agent",
        num_tiles=1,
        is_terrain_shader_enabled=False,
        is_flora_shader_enabled=False,
        is_sun_enabled=True,
        is_sun_shadow_enabled=True,
        is_ssao_enabled=False,
        is_fog_enabled=False,
        is_blur_enabled=False,
        is_postprocessing_enabled=False,
        scenery_mode="tree",
        is_single_collision_mesh=False,
        is_biome_fast=False,
        is_exported_with_absolute_paths=True,
        is_border_calculation_detailed=False,
        is_meta_data_exported=False,
        is_biome_based_on_natural_height=False,
    )


def get_eval_agent_export_config() -> ExportConfig:
    return ExportConfig(
        name="agent",
        num_tiles=1,
        is_terrain_shader_enabled=False,
        is_flora_shader_enabled=False,
        is_sun_enabled=True,
        is_sun_shadow_enabled=True,
        is_ssao_enabled=False,
        is_fog_enabled=False,
        is_blur_enabled=False,
        is_postprocessing_enabled=False,
        scenery_mode="tree",
        is_single_collision_mesh=False,
        is_biome_fast=False,
        is_exported_with_absolute_paths=True,
        is_border_calculation_detailed=True,
        is_meta_data_exported=False,
        is_biome_based_on_natural_height=False,
    )


def get_pretty_export_config() -> ExportConfig:
    return ExportConfig(
        name="pretty",
        num_tiles=1,
        is_terrain_shader_enabled=True,
        is_flora_shader_enabled=True,
        is_sun_enabled=True,
        is_sun_shadow_enabled=True,
        is_ssao_enabled=True,
        is_fog_enabled=True,
        is_blur_enabled=True,
        is_postprocessing_enabled=True,
        scenery_mode="normal",
        is_single_collision_mesh=False,
        is_biome_fast=False,
        is_exported_with_absolute_paths=False,
        is_border_calculation_detailed=True,
        is_meta_data_exported=True,
        is_biome_based_on_natural_height=True,
    )


def get_mouse_and_keyboard_export_config() -> ExportConfig:
    return ExportConfig(
        name="mouse_and_keyboard",
        num_tiles=1,
        is_terrain_shader_enabled=True,
        is_flora_shader_enabled=True,
        is_sun_enabled=True,
        is_sun_shadow_enabled=True,
        is_ssao_enabled=True,
        is_fog_enabled=True,
        is_blur_enabled=True,
        is_postprocessing_enabled=True,
        scenery_mode="normal",
        is_single_collision_mesh=False,
        is_biome_fast=False,
        is_exported_with_absolute_paths=False,
        is_border_calculation_detailed=True,
        is_meta_data_exported=True,
        is_biome_based_on_natural_height=False,
    )


# create the "biomes". Considers elevation and moisture (distance from water), looks that up in a table,
# and does some adjustments
@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class BiomeConfig:
    # a mapping to the config for each scenery object
    flora_config: Dict[str, FloraConfig]
    # corresponds directly to attributes for the WorldEnvironment object in Godot
    # in order to discover the name of a setting, simply modify it, save it, and look at the resulting file
    godot_env_config: Dict[str, Any]
    # corresponds directly to attributes for the ProceduralSky object in Godot
    # in order to discover the name of a setting, simply modify it, save it, and look at the resulting file
    godot_sky_config: Dict[str, Any]
    # corresponds directly to attributes for the DirectionalLight object in Godot
    # in order to discover the name of a setting, simply modify it, save it, and look at the resulting file
    godot_sun_config: Dict[str, Any]
    # elevaations will be rescaled to this range
    min_elevation: float
    max_elevation: float
    # min and max dryness values allowed on the land.
    min_dryness: float
    max_dryness: float
    # moisture values greater than this will be set to min_dryness
    water_cutoff: float
    # the spatial scale of the noise pattern to be used to make the biome edges less regular
    noise_scale: float
    # how much to scale the noise data by when changing biome edges.
    # high values will make the biomes more random (and less strictly determined by height and moisture)
    noise_magnitude: float
    # where to cut off the slope. Is technically in units of "squared slope"
    low_slope_threshold: float  # max of low matrix
    high_slope_threshold: float  # max of middle matrix
    # just sets the colors based on the above mapping
    color_map: Dict[int, Tuple[float, float, float]]
    # mapping for biomes. See the doc where this is defined and passed in.
    biome_matrix: np.ndarray
    # list of all possible biomes. Must define one biome for each id used in the biome_matrix
    biomes: Tuple[Biome, ...]
    # controls the noise that is used to globally vary the density of scenery.
    # this noise texture crosses biome borders, so it can make consistent patterns across them
    # this is the usual spatial scale parameter for noise
    global_foliage_noise_scale: float = 0.01  # bigger = noisier, patterns bigger
    # this is the minimum for the noise. The max will always be 1.0
    # the noise is MULTIPLIED with the other weights
    # so the lower you set this, the less foliage there will be in general
    global_foliage_noise_min: float = 0.7  # closer to 1 = no effect from global noise
    # max meters away from the ocean to assign to the dark shore biome
    dark_shore_distance: float = 3.0
    # max meters of elevation above sea level to assign to the dark shore biome
    dark_shore_height: float = 1.0
    # this controls whether lakes will end up creating coasts or not (if False they will not)
    is_fresh_water_included_in_moisture: bool = False  # True = interior sandy beaches
    # controls whether normals are set per vertex (True) or per face (False)
    is_normal_per_vertex: float = False
    # controls whether colors are set per vertex (True) or per face (False)
    is_color_per_vertex: float = False
    # max elevation above sea level to consider "beach"
    max_shore_height: float = 8.0
    # maximum distance from the ocean to consider "beach"
    # I've set this pretty high right now just to help you see what is going on
    max_shore_distance: float = 20.0
    # any coastal points that have a higher slope than this cutoff will NOT be considered beach
    # and will thus retain their original biome (per the appropriate matrix)
    beach_slope_cutoff: float = 1.0  # same squared slope unit
    # helps to prevent the beach from reaching into other adjacent biomes along the shoreline
    # 1.0 means "dont reach into other biomes at all", while 0.0 means "go crazy and reach in as far
    # as is determined by the max shore distance and height params
    beach_shrink_fraction: float = 0.75
    # over how many meters do we blend the height from the non-beach terrain into the beach
    beach_fade_distance: float = 4.0
    # there must be at least this many meters of contiguous shore points of a sufficiently gentle slope
    # in order for it to be counted as a beach
    min_beach_length: float = 10.0
    # much like the beach shore, except for the swamp biome around bodies of fresh water
    # (elevation and distance maximums)
    swamp_elevation_max: float = 3.0
    swamp_distance_max: float = 10.0
    # extra noise parameters that can alter the shape of swamp biomes to make them not so regular in their appearance
    # normal scale (spatial) and magnitude (amount) params
    swamp_noise_scale: float = 0.01
    swamp_noise_magnitude: float = 1.0
    # controls whether scenery items are added at all
    is_scenery_added: bool = True
    # controls whether the perlin noise terrain shader is enabled. It looks nice, but is a bit slow on the oculus
    is_terrain_noise_shader_enabled: bool = True
    # how many neighbors to sample when mixing colors with adjacent neighbors. Set to 0 to disable
    color_sampling_neighbor_count: int = 10
    # controls the radius for what counts as a neighbor for color mixing
    color_sampling_std_dev: float = 3.0
    # how much noise there is in the coastal biome
    coast_color_noise: float = 0.0
    # how much noise there is in the rock biome
    rock_color_noise: float = 0.0
    # specifies which biome is the coastal one
    coastal_id: int = 3
    # specifies which biome is the rock one
    rock_biome_ids: Tuple[int, ...] = (0,)
    # can add independent noise to each scenery object, but this is slow
    is_independent_noise_per_scenery: bool = False
    # just a reference to the ExportConfig
    export_config: Optional[ExportConfig] = None
    # to generate levels more quickly, can set this to restrict the number of points for distance
    # calculations. Something like 2000 is plenty for good calculations without slowing things down
    # Something like 500 will be really fast, but with minor visual quality issues (noise)
    max_kd_points: Optional[int] = None
    # at this slope, the biome is guaranteed to be cliff
    force_cliff_square_slope: float = 100.0
    # a global modifier on how many scenery objects will be placed
    foliage_density_modifier: float = 1.0
    # whether to do the more complex beach and shore erosion calculations.
    is_beach_computed: bool = True


# from: https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy
def perlin(
    shape: Tuple[int, int],
    scale: float,
    rand: np.random.Generator,
    is_normalized: bool = False,
    noise_min: float = 0.0,
) -> MapFloatNP:
    cache_key = (scale,) + shape
    # create the grid
    y_lin = np.linspace(0, scale * shape[0], shape[0], endpoint=False)
    x_lin = np.linspace(0, scale * shape[1], shape[1], endpoint=False)
    x, y = np.meshgrid(x_lin, y_lin)

    # permutation table
    p = np.arange(max([round(scale * shape[0]), round(scale * shape[1])]) + 1, dtype=int)
    rand.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
    # fade factors
    u, v = fade(xf), fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    result = lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

    if is_normalized:
        result -= np.min(result)
        result /= np.max(result)

        if noise_min > 0.0:
            result /= 1.0 - noise_min
            result += noise_min

    return result


def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)


def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ChasmParams(Serializable):
    start: Tuple[float, float]
    end: Tuple[float, float]
    width: float
    depth: float
    is_depth_relative_to_min: bool = False
    extra_unclimbable_width: float = 0.0

    def get_start_end_dist(self) -> float:
        return math.sqrt((self.start[0] - self.end[0]) ** 2 + (self.start[1] - self.end[1]) ** 2)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class VisibilityCalculator:
    cells_per_meter: float
    height_interpolator: RegularGridInterpolator

    def is_visible_from(self, a: Point3DNP, b: Point3DNP, is_plotted: bool = False) -> bool:
        if np.isclose(a, b).all():
            return True
        a2 = np.array([a[0], a[2]])
        b2 = np.array([b[0], b[2]])
        total_distance_2d = np.linalg.norm((b2 - a2))
        # TODO: parameterize and tune this I guess, is a performance vs correctness tradeoff
        # looking at every other cell seems relatively sasfe, the heights are usually not too variable
        point_count = max(round((self.cells_per_meter * total_distance_2d) * 0.5), 5)
        points = np.linspace(a2, b2, point_count)
        distances = np.linspace(0.0, total_distance_2d, point_count)
        # it's idiotic that these need to be reversed, whatever
        swapped_points = points.copy()
        swapped_points[:, [1, 0]] = swapped_points[:, [0, 1]]
        heights = self.height_interpolator(swapped_points)
        a_height = a[1]
        b_height = b[1]
        # always go from the lowest point to the highest, makes it easier to think about
        if b_height < a_height:
            heights = np.flip(heights)
            b_height, a_height = a_height, b_height
        height_deltas = heights - a_height
        end_height = b_height - a_height
        if is_plotted:
            plt.plot(distances, height_deltas)
            plt.show()
        final_height_dist_ratio = end_height / distances[-1]
        height_dist_ratios = height_deltas[1:-1] / distances[1:-1]
        result = not np.any(height_dist_ratios > final_height_dist_ratio)
        return result


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HeightMap:
    region: Region
    cells_per_meter: float
    X: MapFloatNP
    Y: MapFloatNP
    Z: MapFloatNP
    _simplicity_warnings: List[str]

    @staticmethod
    def create(region: Region, cells_per_meter: float) -> "HeightMap":
        x, y = region.get_linspace(cells_per_meter)
        X, Y = np.meshgrid(x, y)
        Z = 0.0 * X
        return HeightMap(region, cells_per_meter, X, Y, Z, [])

    def plot(self):
        plt.figure(figsize=(20, 20))
        ax = plt.subplot(projection="3d")
        meshcount = 129
        gridcount = len(self.X[0, :])
        if gridcount > meshcount:
            logger.warning(f"WARNING: Mesh is {gridcount / meshcount}x finer than can be displayed")
        ax.plot_wireframe(self.X, self.Y, self.Z, color="k", rcount=meshcount, ccount=meshcount)
        rough_meters = (self.region.x.size + self.region.z.size) / 2.0
        ax.set_zlim(-rough_meters / 2.0, rough_meters / 2.0)
        ax.set_axis_off()
        # print(ax.elev)
        # print(ax.azim)
        plt.show()
        return self

    def add_noise(self, scale: float, rand: np.random.Generator):
        self.Z += rand.uniform(-scale / 2.0, scale / 2.0, self.Z.shape)

    def add_center_biased_noise(
        self, mountain_center: Tuple[float, float], scale: float, mountain_radius: float, rand: np.random.Generator
    ):
        noise = rand.normal(scale, scale / 3.0, self.Z.shape)
        max_dist = mountain_radius * self.region.x.size
        dist_sq_to_mountain = self.get_dist_sq_to(np.array(mountain_center))
        weight = 1.0 - (dist_sq_to_mountain / (max_dist ** 2))
        self.Z = self.Z + np.clip(weight, 0.0, 1.0) * noise

    def copy(self) -> "HeightMap":
        return HeightMap(
            self.region,
            self.cells_per_meter,
            self.X.copy(),
            self.Y.copy(),
            self.Z.copy(),
            self._simplicity_warnings[:],
        )

    def change_height(self, delta: float):
        self.Z += delta

    def get_land_mask(self) -> MapBoolNP:
        return self.Z > 0.0

    def apply_height_mask(self, mask: np.ndarray, height: float, height_mode: HeightMode):
        # basically, sinking one part of the world is equivalent to raising the rest, EXCEPT at the water
        # if there is a negative height difference that would put any part of the region underwater
        # we have to raise the whole rest of the landmass, and mark the coast as unclimbable (if not already done)
        # otherwise, you could simply walk around the island to avoid having to cross chasms

        # figure out everywhere the mask applies to land:
        land_mask = self.get_land_mask()
        if height_mode == HeightMode.RELATIVE:
            if height < 0:
                # plot_value_grid(self.Z, "height when applying")
                # plot_value_grid(mask, "mask")
                effect_mask = np.logical_and(land_mask, mask > 0)
                if not np.any(effect_mask):
                    # plot_value_grid(mask, "doesn't matter mask")
                    return
                min_height = self.Z[effect_mask].min()
                height_delta = min_height + height
                if height_delta < 0:
                    # shift the world up
                    self.Z[land_mask] += -height_delta
                    assert (
                        False
                    ), "Should no longer be able to get here--use the world begin_height_obstacles and end_height_obstacles methods instead"
            self.Z += height * mask
        elif height_mode == HeightMode.ABSOLUTE:
            assert height > 0
            self.Z[np.logical_and(land_mask, mask >= 1.0)] = height
            interpolation_mask = np.logical_and(land_mask, np.logical_and(mask < 1.0, mask > 0.0))
            self.Z[interpolation_mask] = (self.Z[interpolation_mask] - height) * (
                1.0 - mask[interpolation_mask]
            ) + height
        else:
            raise SwitchError(f"Unknown HeightMode: {height_mode}")

    def upsample(self, factor: float = 2.0) -> "HeightMap":
        x, y = self.region.get_linspace(self.cells_per_meter)
        interp_spline = RectBivariateSpline(x, y, self.Z)

        x2 = np.linspace(
            self.region.x.min_ge, self.region.x.max_lt, round(self.region.x.size * self.cells_per_meter * factor) + 1
        )
        y2 = np.linspace(
            self.region.z.min_ge, self.region.z.max_lt, round(self.region.z.size * self.cells_per_meter * factor) + 1
        )

        X2, Y2 = np.meshgrid(x2, y2)
        Z2 = interp_spline(x2, y2)

        return HeightMap(self.region, self.cells_per_meter * factor, X2, Y2, Z2, self._simplicity_warnings)

    def get_rough_height_at_point(self, point: Point2DNP) -> float:
        assert_isinstance(point, Point2DNP)
        return self.Z[self.point_to_index(point)]

    def get_2d_points(self):
        return np.stack([self.X, self.Y], axis=2)

    def get_3d_points(self):
        return np.stack([self.X, self.Z, self.Y], axis=2)

    def get_lineseg_distances(self, start: Point2DNP, end: Point2DNP):
        a = start.reshape((1, 2))
        b = end.reshape((1, 2))
        points_2d = self.get_2d_points()
        line_seg_distances = lineseg_dists(points_2d.reshape((-1, 2)), a, b).reshape(points_2d.shape[:2])
        return line_seg_distances

    def get_heights(self, points: np.ndarray) -> np.ndarray:
        assert_isinstance(points, Point2DListNP)
        points_2d = points.copy()
        # y before x internally
        points_2d[:, [1, 0]] = points_2d[:, [0, 1]]
        grid_x, grid_y = self.region.get_linspace(self.cells_per_meter)
        interp_spline = RegularGridInterpolator((grid_y, grid_x), self.Z)
        return interp_spline(points_2d)

    def restrict_points_to_region(self, points: Point2DListNP) -> Point2DListNP:
        assert_isinstance(points, Point2DListNP)
        mask = np.logical_and(
            np.logical_and(points[:, 0] >= self.region.x.min_ge, points[:, 0] <= self.region.x.max_lt),
            np.logical_and(points[:, 1] >= self.region.z.min_ge, points[:, 1] <= self.region.z.max_lt),
        )
        return points[mask]

    def points_to_indices(self, points: Point2DListNP) -> Tuple[np.ndarray, np.ndarray]:
        assert_isinstance(points, Point2DListNP)
        max_x_index = self.Z.shape[1] - 1
        x = np.rint(np.interp(points[:, 0], [self.region.x.min_ge, self.region.x.max_lt], [0, max_x_index])).astype(
            np.int32
        )
        max_y_index = self.Z.shape[0] - 1
        y = np.rint(np.interp(points[:, 1], [self.region.z.min_ge, self.region.z.max_lt], [0, max_y_index])).astype(
            np.int32
        )
        return x, y

    def get_dist_sq_to(self, point: Point2DNP):
        assert_isinstance(point, Point2DNP)
        locations = self.get_2d_points()
        dist_sq = (locations[:, :, 0] - point[0]) ** 2 + (locations[:, :, 1] - point[1]) ** 2
        return dist_sq

    def radial_flatten(self, point: Point2DNP, radius: float, extra_mask: Optional[MapBoolNP] = None):
        # height = only(self.get_heights(np.array([point])))
        height = self.get_rough_height_at_point(point)
        assert height > WATER_LINE, "Cannot flatten points in the ocean"
        dist_sq = self.get_dist_sq_to(point)

        mask = dist_sq < radius * radius
        if extra_mask is not None:
            mask = np.logical_and(extra_mask, mask)
        mixing = np.clip(((radius - np.sqrt(dist_sq[mask])) / radius) * 2, 0, 1)
        self.Z[mask] = self.Z[mask] * (1.0 - mixing) + height * mixing

        # plot_value_grid(self.Z)

        return dist_sq

    def get_random_land_point(self, rand: np.random.Generator, ideal_shore_dist: float) -> Point2DNP:
        """:returns: a random point that is guaranteed to be on the land"""
        land_mask = self.get_land_mask()
        if ideal_shore_dist > 0.0:
            original_land_mask = land_mask
            shore_cell_dist = int(self.cells_per_meter * ideal_shore_dist) + 1
            land_mask = np.logical_not(
                morphology.dilation(np.logical_not(land_mask), morphology.disk(shore_cell_dist))
            )
            if not np.any(land_mask):
                land_mask = original_land_mask
        possible_points = np.argwhere(land_mask)
        indices = tuple(rand.choice(possible_points))
        x = self.X[indices]
        y = self.Y[indices]
        return np.array([x, y])

    def get_island(self, point: Point2DNP) -> Tuple[MapBoolNP, MapBoolNP]:
        assert_isinstance(point, Point2DNP)
        index = self.point_to_index(point)
        mask = self.get_land_mask().astype(int)
        flood_fill(mask, index, 2, in_place=True)
        island_mask = mask == 2

        water_mask = np.logical_not(self.get_land_mask())
        ocean_mask = water_mask.copy().astype(int)
        flood_fill(ocean_mask, (0, 0), 2, in_place=True)
        ocean_mask = ocean_mask == 2
        fresh_water_mask = np.logical_and(water_mask, np.logical_not(ocean_mask))

        island_mask_without_fresh_water = island_mask.astype(int).copy()
        island_mask_without_fresh_water[fresh_water_mask] = 2
        island_mask_without_fresh_water[ocean_mask] = -100
        flood_fill(island_mask_without_fresh_water, index, 3, tolerance=3, in_place=True)
        island_mask_without_fresh_water = island_mask_without_fresh_water == 3
        # plot_value_grid(island_mask_without_fresh_water, "Island without fresh water")

        return island_mask, island_mask_without_fresh_water

    def generate_visibility_calculator(self) -> VisibilityCalculator:
        """Optimization. Here so we dont have to make this mesh grid multiple times if checking visibility many times"""
        grid_x, grid_y = self.region.get_linspace(self.cells_per_meter)
        height_interpolator = RegularGridInterpolator((grid_y, grid_x), self.Z)
        return VisibilityCalculator(self.cells_per_meter, height_interpolator)

    # TODO: is gross that this, unlike everythign else, accepts only 3d points...
    def get_normals(self, points: np.ndarray) -> np.ndarray:
        delta = 0.5
        points_2d = np.stack([points[:, 2], points[:, 0]], axis=1)

        grid_x, grid_y = self.region.get_linspace(self.cells_per_meter)
        interp_spline = RegularGridInterpolator((grid_y, grid_x), self.Z)
        heights_at_points = interp_spline(points_2d)
        heights_right = interp_spline(points_2d + np.array([0.0, delta]))
        heights_up = interp_spline(points_2d + np.array([delta, 0.0]))
        points_3d = np.stack([points[:, 0], heights_at_points, points[:, 1]], axis=1)
        points_right = np.stack([points[:, 0] + delta, heights_right, points[:, 1]], axis=1)
        points_up = np.stack([points[:, 0], heights_up, points[:, 1] + delta], axis=1)
        right_vector = points_right - points_3d
        up_vector = points_up - points_3d
        right_vector /= np.linalg.norm(right_vector, axis=1, keepdims=True)
        up_vector /= np.linalg.norm(up_vector, axis=1, keepdims=True)
        return np.cross(up_vector, right_vector)

    def create_edge_region(
        self, region: Region, points_per_meter: float, rand: np.random.Generator, noise_scale: float = 0.5
    ) -> EdgedRegion:
        grid_x, grid_y = self.region.get_linspace(self.cells_per_meter)
        interp_spline = RegularGridInterpolator((grid_x, grid_y), self.Z)

        # x/y between two points of the region are fully determined by the two endpoints
        region_points = region.points
        line_num = 0
        all_vertices = []
        for i, start_point in enumerate(region_points):
            end_point = region_points[(i + 1) % len(region_points)]
            meters = ((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2) ** 0.5
            num_points = round(meters * points_per_meter) + 1
            points = np.linspace(start_point, end_point, num_points)
            meters_per_point = 1.0 / points_per_meter
            width = noise_scale * (meters_per_point * 0.5) * 0.999
            noise = rand.uniform(-width, width, points.shape)
            if line_num in (1, 3):
                noise[:, 0] = 0.0
            else:
                noise[:, 1] = 0.0
            noise[0, 0] = 0.0
            noise[0, 1] = 0.0
            noise[-1, 0] = 0.0
            noise[-1, 1] = 0.0

            # TODO: put back, if I can get the random generation working properly...
            # noised_points = points + noise
            points[:, [1, 0]] = points[:, [0, 1]]
            noised_points = points
            heights = interp_spline(noised_points)

            vertices = np.stack((noised_points[:, 1], heights, noised_points[:, 0]), axis=1)

            all_vertices.append(vertices[1:])

            line_num += 1

        all_vertices = np.concatenate(all_vertices, axis=0)

        if IS_DEBUG_VIS:
            plot_points(all_vertices, 0, 2)

        return EdgedRegion(region.x, region.z, all_vertices)

    # TODO: could localize these
    # TODO: they should not be able to overlap each other, or other features (dont apply unless all of the points are roughly the same level)
    # TODO: parameterize
    def add_plateaus(
        self,
        rand: np.random.Generator,
        num_plateaus: int,
        plateau_height_dist_sampler: Callable[[np.random.Generator], float],
        plateau_radius_dist_sampler: Callable[[np.random.Generator], float],
    ):
        logger.info(f"Adding {num_plateaus} plateaus")
        mins = (self.region.x.min_ge, self.region.z.min_ge)
        maxs = (self.region.x.max_lt, self.region.z.max_lt)
        plateau_locations = rand.uniform(mins, maxs, (num_plateaus, 2))
        plateau_radii = [plateau_radius_dist_sampler(rand) for _ in range(num_plateaus)]
        plateau_heights = [plateau_height_dist_sampler(rand) for _ in range(num_plateaus)]
        # TODO: better vectorize...
        for k, location in enumerate(plateau_locations):
            radius = plateau_radii[k]
            height = plateau_heights[k]
            logger.info(f"Plateau {k} is radius={radius} at {location} with height={height}")
            min_sq_dist = radius ** 2
            matching_coords = []
            total_height = 0
            for i, j in np.ndindex(*self.Z.shape):
                x = self.X[i, j]
                y = self.Y[i, j]
                delta_x = location[0] - x
                delta_y = location[1] - y
                sq_dist = delta_x ** 2 + delta_y ** 2
                if sq_dist < min_sq_dist:
                    matching_coords.append((i, j))
                    self.Z[i, j] += height
                    total_height += self.Z[i, j]
            if len(matching_coords) > 0:
                average_height = total_height / len(matching_coords)
                for coord in matching_coords:
                    self.Z[coord] = average_height

    def clamp_heights(self, smooth_factor: int = 1):
        # TODO: calculate
        scale_factor = 3.5
        for i, j in np.ndindex(*self.Z.shape):
            value = self.Z[i, j] / scale_factor
            # roughly from here: https://gamedev.stackexchange.com/questions/116205/terracing-mountain-features/
            self.Z[i, j] = (
                round(value) + 0.5 * (2 * (value - round(value))) ** (1 + 2 * smooth_factor)
            ) * scale_factor

    def blur(self, blur_meters_std_dev: float):
        if blur_meters_std_dev > 0:
            blur = blur_meters_std_dev * self.cells_per_meter
            self.Z = gaussian_filter(self.Z, sigma=blur)

    def tilt(self, x_delta: float, y_delta: float):
        for i, j in np.ndindex(*self.Z.shape):
            x = self.X[i, j]
            y = self.Y[i, j]
            self.Z[i, j] += x * x_delta + y * y_delta

    def sink_edges(
        self,
        rand: np.random.Generator,
        is_circular: bool,
        fade_fraction: float,
        fade_noise_scale: float,
        fade_noise_min: float,
        extra_weight: float = 1.0,
    ):
        fade_weights = np.zeros_like(self.Z)
        assert fade_weights.shape[0] == fade_weights.shape[1], "Doesnt support rectangular arrays"

        mid_idx = np.clip(fade_weights.shape[0] // 2, 0, fade_weights.shape[0] - 1)
        points = self.get_2d_points()
        center_point = points[mid_idx, mid_idx]
        deltas = points - center_point
        if is_circular:
            dist = np.sqrt(deltas[:, :, 0] ** 2 + deltas[:, :, 1] ** 2)
        else:
            dist = np.max(np.abs(deltas), axis=2)
        edge_start_idx = np.clip(round((1.0 - fade_fraction) * fade_weights.shape[0]), 0, fade_weights.shape[0] - 1)
        edge_start_dist = dist[mid_idx, edge_start_idx]
        edge_end_dist = dist[mid_idx, -1]
        falloff_dist = edge_end_dist - edge_start_dist
        if falloff_dist < 0.001:
            return
        fade_weights = (dist - edge_start_dist) / falloff_dist
        fade_weights[dist < edge_start_dist] = 0.0
        # plot_value_grid(fade_weights)

        noise = perlin(fade_weights.shape, fade_noise_scale, rand)
        # plot_value_grid(noise)
        noise -= np.min(noise)
        noise /= np.max(noise) * (1 / fade_noise_min)
        noise += fade_noise_min

        fade_weights *= noise
        # plot_value_grid(fade_weights, title="FADE WEIGHTS")
        max_height = np.max(self.Z)
        self.Z -= max_height * fade_weights * extra_weight

    def lower_edges(self, final_height: float, fade_over_cells: int):
        Z2 = self.Z.copy()
        _min_helper(Z2[0, :], final_height)
        _min_helper(Z2[-1, :], final_height)
        _min_helper(Z2[:, 0], final_height)
        _min_helper(Z2[:, -1], final_height)
        for i in range(1, 1 + fade_over_cells):
            factor = i / fade_over_cells
            Z2[i, :] *= factor
            Z2[-1 + -i, :] *= factor
            Z2[:, i] *= factor
            Z2[:, -1 + -i] *= factor
        self.Z = Z2

    # TODO: could be made more efficient
    def add_buildings(self, buildings: Tuple[Building, ...]):
        Z2 = self.Z.copy()

        # figures out which buildings are relevant to which points
        x_relevant_building_ids = {}
        for i, x in enumerate(self.X[0, :]):
            x_relevant_building_ids[i] = {i for i, b in enumerate(buildings) if b.region.x.contains(x)}
        z_relevant_building_ids = {}
        for i, z in enumerate(self.Y[:, 0]):
            z_relevant_building_ids[i] = {i for i, b in enumerate(buildings) if b.region.z.contains(z)}

        # set heights to maximums
        empty_set = set()
        for i, j in np.ndindex(*self.Z.shape):
            x = self.X[i, j]
            y = self.Y[i, j]
            building_ids_at_location = x_relevant_building_ids.get(i, empty_set).intersection(
                z_relevant_building_ids.get(j, empty_set)
            )
            if len(building_ids_at_location) > 0:
                max_building_height = max(
                    [buildings[building_id].height.max_lt for building_id in building_ids_at_location]
                )
                Z2[i, j] = max([max_building_height, Z2[i, j]])

        self.Z = Z2

    def get_elevation(self) -> MapFloatNP:
        return self.Z.copy()

    def get_squared_slope(self) -> MapFloatNP:
        """:returns: squared slope. Squared for efficiency (dont have to do as m any square roots"""
        meters_per_cell = 1.0 / self.cells_per_meter
        horizontal_slope = _calculate_horizontal_delta(self.Z) / (2 * meters_per_cell)
        vertical_slope = _calculate_vertical_delta(self.Z) / (2 * meters_per_cell)
        return horizontal_slope * horizontal_slope + vertical_slope * vertical_slope

    def _get_ocean_mask(self, water_mask: MapBoolNP) -> Optional[MapBoolNP]:
        for edge, ind in (
            (water_mask[0, :], (0, None)),
            (water_mask[-1, :], (-1, None)),
            (water_mask[:, 0], (None, 0)),
            (water_mask[:, -1], (None, -1)),
        ):
            ocean_indices = np.argwhere(edge)
            if len(ocean_indices) > 0:
                value = ocean_indices[0][0]
                ocean_index = tuple([(value if x is None else x) for x in ind])
                break
        # nothing to do if there is no ocean
        else:
            return None
        ocean_mask = water_mask.copy().astype(int)
        flood_fill(ocean_mask, ocean_index, 2, in_place=True)
        return ocean_mask == 2

    def add_debug_circle_lake(self, position: Point2DNP, radius: float, fade_radius: float):
        points = self.get_2d_points()
        deltas = points - position
        distances = np.linalg.norm(deltas, axis=2)
        weights = np.zeros_like(distances)
        fade_range = fade_radius - radius
        weights[distances < fade_radius] = (fade_radius - distances[distances < fade_radius]) / fade_range
        weights[distances < radius] = 1.0

        plot_value_grid(weights, "Debug circle lake (before)")

        height = -1.0
        land_mask = self.get_land_mask()
        self.Z[np.logical_and(land_mask, weights >= 1.0)] = height

        plot_value_grid(self.Z, "Debug circle lake (mid)")

        interpolation_mask = np.logical_and(land_mask, np.logical_and(weights < 1.0, weights > 0.0))
        self.Z[interpolation_mask] = (self.Z[interpolation_mask] - height) * (
            1.0 - weights[interpolation_mask]
        ) + height

        plot_value_grid(self.Z, "Debug circle lake (after)")

    def erode_shores(
        self,
        rand: np.random.Generator,
        biome_config: BiomeConfig,
        is_debug_graph_printing_enabled: bool = False,
    ) -> SpecialBiomes:
        """
        What is a beach, really?

        This function is of the opinion that a beach is:
        - On the ocean
        - Gently sloped
        - Sufficiently large

        Such regions are found, and then eroded (reduce land heights near the water for a flatter beach)

        :returns: two masks (for dark and light beach, ie, near and far from the water)
        """

        # figure out all shore points on the ocean
        water_mask = self.Z < WATER_LINE
        ocean_mask = self._get_ocean_mask(water_mask)
        fresh_water_mask = np.logical_and(water_mask, np.logical_not(ocean_mask))

        if not biome_config.is_beach_computed:
            beach_mask = np.zeros_like(fresh_water_mask)
            swamp_mask = np.zeros_like(fresh_water_mask)
        else:
            # get the beach mask, if any
            beach_mask = self._get_beach_mask(
                rand, biome_config, ocean_mask, water_mask, is_debug_graph_printing_enabled
            )

            # figure out the other biomes

            fresh_water_shore_mask = np.logical_and(
                morphology.dilation(fresh_water_mask, morphology.square(3)), np.logical_not(fresh_water_mask)
            )
            if np.any(fresh_water_shore_mask):
                tree = create_kd_tree(
                    np.argwhere(fresh_water_shore_mask),
                    "fresh water shore",
                    max_points=biome_config.max_kd_points,
                    rand=rand,
                )
                shore_distance = np.zeros_like(self.Z)
                is_land = self.Z >= WATER_LINE
                shore_distance[is_land] = tree.query(np.argwhere(is_land), k=1)[0]
                shore_distance = shore_distance * (1.0 / self.cells_per_meter)
                swamp_noise = (
                    perlin(shore_distance.shape, biome_config.swamp_noise_scale, rand)
                    * biome_config.swamp_noise_magnitude
                    * biome_config.swamp_distance_max
                )
                inland_area_below_swamp_line = np.logical_and(
                    self.Z < WATER_LINE + biome_config.swamp_elevation_max, np.logical_not(ocean_mask)
                )
                swamp_mask = np.logical_and(
                    inland_area_below_swamp_line, shore_distance < biome_config.swamp_distance_max + swamp_noise
                )
                if is_debug_graph_printing_enabled:
                    plot_value_grid(swamp_noise, "swamp noise")
                    plot_value_grid(
                        np.logical_and(inland_area_below_swamp_line, shore_distance < biome_config.swamp_distance_max),
                        "swamp without noise",
                    )
                    plot_value_grid(swamp_mask, "swamp with noise")
            else:
                swamp_mask = fresh_water_shore_mask.copy()

        return SpecialBiomes(beach_mask=beach_mask, swamp_mask=swamp_mask, fresh_water_mask=fresh_water_mask)

    def get_all_coast(self, extra_unclimbable_distance: float = 1.5) -> MapBoolNP:
        water_mask = self.Z < WATER_LINE
        shore_mask = np.logical_and(morphology.dilation(water_mask, morphology.square(3)), np.logical_not(water_mask))
        cell_distance = round(0.5 * extra_unclimbable_distance * self.cells_per_meter)
        return morphology.dilation(shore_mask, morphology.disk(cell_distance))

    def _get_beach_mask(
        self,
        rand: np.random.Generator,
        biome_config: BiomeConfig,
        ocean_mask: MapBoolNP,
        water_mask: MapBoolNP,
        is_debug_graph_printing_enabled: bool = True,
    ):
        if ocean_mask is None:
            empty_mask = np.zeros_like(water_mask)
            return empty_mask

        shore_mask = np.logical_and(morphology.dilation(ocean_mask, morphology.square(3)), np.logical_not(ocean_mask))

        if is_debug_graph_printing_enabled:
            plot_value_grid(ocean_mask, "ocean")
            plot_value_grid(shore_mask, "ocean shore")

        # nothing to do if there is no shore:
        if not np.any(shore_mask):
            empty_mask = np.zeros_like(water_mask)
            return empty_mask

        # compute slopes at shore points
        squared_slope = self.get_squared_slope()
        # remove shore points where the slope into the water is too high
        gently_sloped_shore_mask = np.zeros_like(shore_mask)
        gently_sloped_shore_mask[shore_mask] = squared_slope[shore_mask] < biome_config.beach_slope_cutoff

        if is_debug_graph_printing_enabled:
            plot_value_grid(gently_sloped_shore_mask, "gently sloped ocean shore")

        # expand the shore indices. This helps connect nearby regions
        expansion_cell_distance = round((biome_config.max_shore_distance / 2.0) * self.cells_per_meter)
        expanded_shore_mask = np.logical_and(
            shore_mask,
            morphology.dilation(gently_sloped_shore_mask, morphology.square(1 + (2 * expansion_cell_distance))),
        )

        if is_debug_graph_printing_enabled:
            plot_value_grid(expanded_shore_mask, "expanded shore mask")

        # contract the shore indices by more than we expanded. This prevents us from eating into nearby regions
        contraction_cell_distance = round(
            biome_config.beach_shrink_fraction * biome_config.max_shore_distance * self.cells_per_meter
        )
        rest_of_shore_mask = np.logical_and(shore_mask, np.logical_not(expanded_shore_mask))
        expanded_rest_of_shore_mask = np.logical_and(
            shore_mask,
            morphology.dilation(rest_of_shore_mask, morphology.square(1 + (2 * contraction_cell_distance))),
        )
        contracted_shore_mask = np.logical_and(expanded_shore_mask, np.logical_not(expanded_rest_of_shore_mask))

        if is_debug_graph_printing_enabled:
            plot_value_grid(contracted_shore_mask, "contracted shore mask")

        # create connected beach components
        lw, num = measurements.label(contracted_shore_mask, structure=generate_binary_structure(2, 2))
        # remove beach components that are too small
        all_areas = measurements.sum(contracted_shore_mask, lw, index=np.arange(np.max(lw) + 1))
        selected_area_ids = []
        min_cells_for_beach = biome_config.min_beach_length * self.cells_per_meter
        for area_id, area in enumerate(all_areas):
            if area < min_cells_for_beach:
                continue
            selected_area_ids.append(area_id)
        large_patch_shore_indices = np.argwhere(np.isin(lw, tuple(selected_area_ids), assume_unique=True))

        if is_debug_graph_printing_enabled:
            debug_large_patch_shore = np.zeros_like(self.Z)
            for index in list(large_patch_shore_indices):
                debug_large_patch_shore[tuple(index)] = 1.0
            plot_value_grid(debug_large_patch_shore, "large patch gentle slope ocean shore")

        # compute beach line distance matrix
        tree = create_kd_tree(
            large_patch_shore_indices, "large patch", max_points=biome_config.max_kd_points, rand=rand
        )
        shore_distance = np.zeros_like(self.Z)
        is_land = self.Z >= WATER_LINE
        shore_distance[is_land] = tree.query(np.argwhere(is_land), k=1)[0]
        shore_distance = shore_distance * (1.0 / self.cells_per_meter)

        if is_debug_graph_printing_enabled:
            plot_value_grid(shore_distance, title="shore distances")

        # the beach is anything below a (noisy) height AND within a (noisy) distance of the beach line
        beach_mask = np.logical_and(
            np.logical_and(self.Z < biome_config.max_shore_height, shore_distance < biome_config.max_shore_distance),
            is_land,
        )

        if is_debug_graph_printing_enabled:
            plot_value_grid(beach_mask, title="beach mask")

        # sometimes can end up with little islands of terrain isolated on the beach.
        # to deal with this, blend them in
        non_beach_land = np.logical_and(is_land, np.logical_not(beach_mask))
        # dilate once. this will be removed from the beach mask at the very end so that the region ids are set correctly
        beach_fade_cell_distance = round(0.5 * biome_config.beach_fade_distance * self.cells_per_meter)
        expanded_non_beach_land = morphology.dilation(non_beach_land, morphology.disk(beach_fade_cell_distance))
        # dilate again. the delta here defines the region over which we will blend heights
        full_non_beach_land = morphology.dilation(expanded_non_beach_land, morphology.disk(beach_fade_cell_distance))
        fade_region = np.logical_and(full_non_beach_land, np.logical_not(non_beach_land))
        fade_region = np.logical_and(fade_region, beach_mask)
        if np.any(fade_region):
            # figure out the distance to make this mask
            non_beach_edge = np.logical_and(
                morphology.dilation(non_beach_land, morphology.square(3)), np.logical_not(non_beach_land)
            )
            land_tree = create_kd_tree(
                np.argwhere(non_beach_edge), "non beach", max_points=biome_config.max_kd_points, rand=rand
            )
            land_distance = np.zeros_like(self.Z)
            land_distance[fade_region] = land_tree.query(np.argwhere(fade_region), k=1)[0]

            # TODO check with josh about right way to return
            if np.max(land_distance) == 0:
                return beach_mask

            fade_weights = 1.0 - (land_distance / np.max(land_distance))
            fade_weights[np.logical_not(fade_region)] = 0.0

            if is_debug_graph_printing_enabled:
                plot_value_grid(fade_region, title="fade region")
                plot_value_grid(fade_weights, title="fade weights")

            # compute the "ideal" heights for each point on the beach (from the distance to the beach line)
            # target_height = max_shore_height * 0.5
            # TODO: RESUME: maybe like a squared falloff here instead?
            target_height = 0.1
            ideal_heights = target_height * shore_distance

            if is_debug_graph_printing_enabled:
                old_Z = self.Z.copy()

            # mix the ideal heights with the true heights
            mixing_weights = np.clip(shore_distance[beach_mask] / biome_config.max_shore_distance, 0, 1)
            mixing_weights = np.max(np.stack([fade_weights[beach_mask], mixing_weights], axis=1), axis=1)
            self.Z[beach_mask] = (
                mixing_weights * self.Z[beach_mask] + (1.0 - mixing_weights) * ideal_heights[beach_mask]
            )

            if is_debug_graph_printing_enabled:
                mixing_weights_debug = np.zeros_like(self.Z)
                mixing_weights_debug[beach_mask] = mixing_weights
                plot_value_grid(mixing_weights_debug, "mixing weights")
                plot_value_grid(self.Z, "new heights")
                plot_value_grid(self.Z - old_Z, "height delta")

            # finally, remove the stuff near the edges
            beach_mask = np.logical_and(beach_mask, np.logical_not(expanded_non_beach_land))

        if is_debug_graph_printing_enabled:
            plot_value_grid(beach_mask, "beach")

        return beach_mask

    def get_water_distance(
        self,
        rand: np.random.Generator,
        is_fresh_water_included_in_moisture: bool,
        max_points: Optional[int] = None,
        for_points: Optional[MapBoolNP] = None,
    ) -> Optional[MapFloatNP]:

        moisture = self.Z.copy()
        water_points = self.Z < WATER_LINE
        moisture[water_points] = 0.0
        land_points = self.Z >= WATER_LINE
        moisture[land_points] = 1.0

        if is_fresh_water_included_in_moisture:
            shore_points = _get_shore_points(rand, water_points)
        else:
            ocean_mask = self._get_ocean_mask(water_points)
            if ocean_mask is None:
                return None
            shore_points = _get_shore_points(rand, ocean_mask)

        num_points = len(shore_points)
        if num_points == 0:
            return None
        tree = create_kd_tree(shore_points, "shore", max_points=max_points, rand=rand)

        if for_points is not None:
            result = np.zeros_like(moisture)
            result[np.nonzero(for_points)] = tree.query(np.argwhere(for_points), k=1)[0]
            return result * (1.0 / self.cells_per_meter)

        moisture[np.nonzero(moisture)] = tree.query(np.argwhere(moisture), k=1)[0]
        return moisture * (1.0 / self.cells_per_meter)

    def distances_from_points(
        self,
        point_indices: np.ndarray,
        mask: MapBoolNP,
        name: str,
        max_points: Optional[int] = None,
        rand: Optional[np.random.Generator] = None,
    ):
        distances = np.zeros_like(self.Z)
        tree = create_kd_tree(point_indices, name, max_points, rand)
        distances[np.nonzero(mask)] = tree.query(np.argwhere(mask), k=1)[0]
        return distances * (1.0 / self.cells_per_meter)

    def _full_distances_from_points(
        self,
        point_indices: np.ndarray,
        mask: MapBoolNP,
        name: str,
        max_points: Optional[int] = None,
        rand: Optional[np.random.Generator] = None,
    ):
        distances = np.zeros_like(self.Z)
        tree = create_kd_tree(point_indices, name, max_points, rand)
        distances[np.nonzero(mask)], indices = tree.query(np.argwhere(mask), k=1)
        return distances * (1.0 / self.cells_per_meter), indices

    def point_to_index(self, point: Point2DNP) -> Tuple[float, float]:
        assert_isinstance(point, Point2DNP)
        max_x_index = self.Z.shape[1] - 1
        one_cell_x = self.region.x.size / (max_x_index)
        x = round((point[0] - self.region.x.min_ge) / one_cell_x)
        max_y_index = self.Z.shape[0] - 1
        one_cell_y = self.region.z.size / (max_y_index)
        y = round((point[1] - self.region.z.min_ge) / one_cell_y)
        return clamp(y, 0, max_y_index), clamp(x, 0, max_x_index)

    def index_to_point_2d(self, indices: Tuple[int, int]) -> Point2DNP:
        return np.array([self.X[indices], self.Y[indices]])

    def raise_island(self, island: MapBoolNP, delta: float):
        self.Z[island] += delta

    def lower_island(self, island: MapBoolNP, set_to_height: float):
        min_height = self.Z[island].min()
        amount = min_height - set_to_height
        self.Z[island] -= amount
        return amount

    def get_outline(self, mask: MapBoolNP, cell_thickness: int):
        outline = np.logical_and(morphology.dilation(mask, morphology.disk(1)), np.logical_not(mask))
        if cell_thickness > 1:
            outline = morphology.dilation(outline, morphology.disk(cell_thickness))
        return outline

    def shrink_region(self, mask: MapBoolNP, num_cells: int):
        return np.logical_and(
            mask, np.logical_not(morphology.dilation(np.logical_not(mask), morphology.disk(num_cells)))
        )

    def interpolate_heights(self, region: MapBoolNP, border: MapBoolNP):
        expanded_region = morphology.dilation(region, morphology.disk(1))
        border = np.logical_and(expanded_region, border)

        # plot_value_grid(region, "region")
        # plot_value_grid(border, "border")

        point_ind = np.argwhere(border)
        border_dist, border_ind = self._full_distances_from_points(point_ind, region, "border a")
        border_dist = border_dist[region]
        near_point_inds = point_ind[border_ind]
        heights = self.Z[near_point_inds[:, 0], near_point_inds[:, 1]]

        # TODO: fix this--should be based on how much space we have to the safety radius, really
        max_dist = 10.0
        mix_factor = np.clip((2 * border_dist / max_dist) - 1.0, 0, 1)

        # old_z = self.Z.copy()

        self.Z[region] = mix_factor * self.Z[region] + (1.0 - mix_factor) * heights

        # plot_value_grid(self.Z - old_z)

    def log_simplicity_warning(self, message: str):
        self._simplicity_warnings.append(message)

    def clear_simplicity_warnings(self):
        self._simplicity_warnings.clear()

    def get_simplicity_warnings(self) -> Tuple[str, ...]:
        return tuple(self._simplicity_warnings)

    def create_radial_mask(self, point: Point2DNP, radius) -> MapFloatNP:
        assert_isinstance(point, Point2DNP)
        dist_sq = self.get_dist_sq_to(point)

        result = np.zeros_like(self.Z)
        mask = dist_sq < radius * radius
        result[mask] = np.clip(((radius - np.sqrt(dist_sq[mask])) / radius) * 2, 0, 1)

        # plot_value_grid(result)

        return result

    def add_hill(self, point: Point2DNP, scale: float, radius: float, mask: MapBoolNP):
        assert_isinstance(point, Point2DNP)
        dist_sq = self.get_dist_sq_to(point)

        result = np.zeros_like(self.Z)
        nearby_mask = np.logical_and(dist_sq < radius * radius, mask)
        result[nearby_mask] = np.clip(((radius - np.sqrt(dist_sq[nearby_mask])) / radius), 0, 1)
        # plot_value_grid(result)
        temp = self.Z[nearby_mask] * scale
        self.Z[nearby_mask] = (1.0 - result[nearby_mask]) * self.Z[nearby_mask] + result[nearby_mask] * temp


def create_kd_tree(points, name: str, max_points: Optional[int] = None, rand: Optional[np.random.Generator] = None):
    reduction = 1.0
    if max_points is not None:
        reduction = max_points / len(points)
        if reduction < 1.0:
            points = np.logical_and(points, rand.uniform(size=points.shape) < reduction)
    # print(f"Creating {name} KDTree with {round(len(points) * reduction)} points")
    return KDTree(points)


def _get_shore_points(rand: np.random.Generator, water_points: np.ndarray):
    border, border_points = get_border_points(water_points, rand)
    return border_points


def _min_helper(value: np.ndarray, min_value: float):
    value[value > min_value] = min_value


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def _calculate_horizontal_delta(Z):
    first_col = np.expand_dims(Z[:, 0] - (Z[:, 1] - Z[:, 0]), 1)
    last_col = np.expand_dims(Z[:, -1] - (Z[:, -2] - Z[:, -1]), 1)
    combined = np.concatenate([first_col, Z, last_col], axis=1)
    return combined[:, 2:] - combined[:, :-2]


def _calculate_vertical_delta(Z):
    first_row = np.expand_dims(Z[0, :] - (Z[1, :] - Z[0, :]), 0)
    last_row = np.expand_dims(Z[-1, :] - (Z[-2, :] - Z[-1, :]), 0)
    combined = np.concatenate([first_row, Z, last_row], axis=0)
    return combined[2:, :] - combined[:-2, :]


def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    # if these two points are the same, we just care about distance from a point
    if abs(d_ba.sum()) < 0.001:
        return np.linalg.norm(p - a, axis=1)
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))
    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)
    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])
    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
    return np.hypot(h, c)


def get_border_points(data, rand: np.random.Generator, reduction: float = 0.3):
    data = data.astype(np.int)
    kernel = np.array(
        [
            [0, 1, 0],
            [16, 0, 16 ** 2],
            [0, 16 ** 3, 0],
        ]
    )
    sums = convolve(data, kernel)
    expected_sums = data + data * 16 + data * 16 ** 2 + data * 16 ** 3
    border = sums != expected_sums
    # plot_value_grid(border, "BORDER")
    # make it go 3x faster and add a little noise...  we are bad people
    # might even get limited later as well... (by the KD tree construction)
    border = np.logical_and(border, rand.uniform(size=border.shape) < reduction)
    border_points = np.argwhere(border)
    return border, border_points


def selected_distance_weighted_points(
    rand: np.random.Generator,
    possible_points: Point2DListNP,
    center_point: Point2DNP,
    goal_distance_weight: stats.norm,
    point_count: int,
):
    distances = np.linalg.norm(possible_points - center_point, axis=1)
    location_weights = goal_distance_weight.pdf(distances)
    location_probabilities = location_weights / location_weights.sum()
    return rand.choice(possible_points, point_count, p=location_probabilities)


def get_flora_config_by_file(flora_config, resource_file):
    resource_name = resource_file.split("/")[-1].split(".")[0]
    if resource_name in flora_config:
        config = flora_config[resource_name]
    else:
        config = flora_config.get(resource_name.replace("_low", ""), None)
    return config, resource_name
