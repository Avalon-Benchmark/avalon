from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import attr
import numpy as np

from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.flora import FloraConfig
from avalon.datagen.world_creation.constants import DARK_SHORE
from avalon.datagen.world_creation.constants import FRESH_WATER
from avalon.datagen.world_creation.constants import SWAMP
from avalon.datagen.world_creation.constants import UNCLIMBABLE_BIOME_ID
from avalon.datagen.world_creation.types import Biome
from avalon.datagen.world_creation.types import Color
from avalon.datagen.world_creation.utils import hex_to_rgb


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


# there are 3 grids below, 1 for each level of steepness (from low to high)
# fmt: off
# 20 x 20 grid based on this: http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/
#    /\
#    |
#    |
#  elevation
#
#            dryness ------->
_raw_biome_id_mapping = np.array(
    [
        # low slope biomes
        np.flip([
            [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],  #
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],  #
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],  #
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        ], 0),
        # medium slope biomes
        np.flip([
            [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],  #
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],  #
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], 0),
        # high slope biomes
        np.flip([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], 0),
    ]
)
# fmt: on


def generate_biome_config(
    rand: np.random.Generator,
    export_config: ExportConfig,
    max_height: float,
    foliage_density_modifier: float = 1.0,
):
    COASTAL_ID = 3
    ROCK_IDS = (0,)

    COLORS = {
        "white": "#FFFFFF",
        "light-green": "#9BC57A",
        "med-green": "#89B764",
        "dark-green": "#7CAA57",
        "dark-brown": "#B09B57",
        "light-brown": "#D0B771",
        "dark-grey": "#757069",
        "sand": "#E7E0A7",
        "swamp": "#A48A30",
        "bare": "#B6B9BC",
    }

    BIOMES = (
        Biome(2, "water", COLORS["sand"], color_jitter=0.0, correlated_color_jitter=0.0),  # ocean blue
        Biome(COASTAL_ID, "coastal", COLORS["sand"], color_jitter=0.01, correlated_color_jitter=0.02),  # light brown
        Biome(ROCK_IDS[0], "bare", COLORS["bare"], color_jitter=0.0, correlated_color_jitter=0.02),
        # Biome(ROCK_IDS[1], "scorched", COLORS["dark-grey"], color_jitter=0.0, correlated_color_jitter=0.025),
        Biome(5, "tropical rain forest", COLORS["light-green"], color_jitter=0.01, correlated_color_jitter=0.02),
        Biome(6, "tropical seasonal forest", COLORS["med-green"], color_jitter=0.01, correlated_color_jitter=0.02),
        Biome(7, "temperate rain forest", COLORS["med-green"], color_jitter=0.01, correlated_color_jitter=0.02),
        Biome(8, "temperate deciduous forest", COLORS["dark-green"], color_jitter=0.01, correlated_color_jitter=0.02),
        Biome(9, "grassland", COLORS["dark-brown"], color_jitter=0.01, correlated_color_jitter=0.02),
        Biome(10, "temperate desert", COLORS["dark-brown"], color_jitter=0.005, correlated_color_jitter=0.02),
        Biome(11, "subtropical desert", COLORS["light-brown"], color_jitter=0.005, correlated_color_jitter=0.02),
        # other stuff
        Biome(
            UNCLIMBABLE_BIOME_ID, "unclimbable", COLORS["dark-grey"], color_jitter=0.0, correlated_color_jitter=0.01
        ),
        Biome(DARK_SHORE, "dark shore", COLORS["sand"], color_jitter=0.0, correlated_color_jitter=0.01),
        Biome(FRESH_WATER, "fresh water", COLORS["swamp"], color_jitter=0.0, correlated_color_jitter=0.01),
        Biome(SWAMP, "swamp", COLORS["swamp"], color_jitter=0.0, correlated_color_jitter=0.01),
    )

    flora_config = {
        # scenery
        "acacia": FloraConfig(
            "res://scenery/trees/acacia.tscn",
            default_scale=0.6,
            is_noise_shader_enabled=False,
            height_offset=-1.5,
            collision_extents=np.array([0.507, 6, 0.533]),
        ),
        "fir": FloraConfig(
            "res://scenery/trees/fir.tscn",
            default_scale=0.6,
            is_noise_shader_enabled=False,
            height_offset=-2.0,
            collision_extents=np.array([1.0, 5.56902, 1.0]),
        ),
        "maple_red": FloraConfig(
            "res://scenery/trees/maple_red.tscn",
            default_scale=0.8,
            is_noise_shader_enabled=False,
            height_offset=-2.25,
            collision_extents=np.array([0.65, 6.0, 0.65]),
        ),
        "maple_orange": FloraConfig(
            "res://scenery/trees/maple_orange.tscn",
            default_scale=0.8,
            is_noise_shader_enabled=False,
            height_offset=-2.25,
            collision_extents=np.array([0.65, 6.0, 0.65]),
        ),
        "maple_yellow": FloraConfig(
            "res://scenery/trees/maple_yellow.tscn",
            default_scale=0.8,
            is_noise_shader_enabled=False,
            height_offset=-2.25,
            collision_extents=np.array([0.65, 6.0, 0.65]),
        ),
        "palm": FloraConfig(
            "res://scenery/trees/palm.tscn",
            default_scale=0.8,
            is_noise_shader_enabled=False,
            height_offset=-2.5,
            collision_extents=np.array([0.507, 5.75, 0.533]),
        ),
        "fruit_tree_normal": FloraConfig(
            "res://scenery/trees/fruit_tree_normal.tscn", default_scale=0.8, is_noise_shader_enabled=False
        ),
        "fruit_tree_tropical": FloraConfig(
            "res://scenery/trees/fruit_tree_tropical.tscn", default_scale=0.8, is_noise_shader_enabled=True
        ),
        "bush": FloraConfig("res://scenery/bush.tscn", default_scale=0.4, is_shadowed=False, height_offset=-0.25),
        "flower_blue": FloraConfig(
            "res://scenery/flower_blue.tscn", default_scale=0.7, is_shadowed=False, height_offset=-0.075
        ),
        "flower_pink": FloraConfig(
            "res://scenery/flower_pink.tscn", default_scale=0.7, is_shadowed=False, height_offset=-0.075
        ),
        "flower_yellow": FloraConfig(
            "res://scenery/flower_yellow.tscn", default_scale=0.7, is_shadowed=False, height_offset=-0.075
        ),
        "mushroom": FloraConfig(
            "res://scenery/mushroom.tscn", default_scale=0.5, is_shadowed=False, height_offset=-0.05
        ),
        # items
        "boulder": FloraConfig("res://items/stone/boulder.tscn", default_scale=0.5),
        "stone": FloraConfig("res://items/stone/stone.tscn", default_scale=0.5),
        "rock": FloraConfig("res://items/stone/rock.tscn", default_scale=1.1),
        "log": FloraConfig("res://items/wood/log.tscn", default_scale=1.0),
        "stick": FloraConfig("res://items/wood/stick.tscn", default_scale=1.0),
        # animals, predators
        "wolf": FloraConfig("res://entities/animals/predators/wolf_246f.tscn", default_scale=1.0),
        "jaguar": FloraConfig("res://entities/animals/predators/jaguar_348f.tscn", default_scale=1.0),
        "hippo": FloraConfig("res://entities/animals/predators/hippo_204f.tscn", default_scale=1.0),
        "bee": FloraConfig("res://entities/animals/predators/bee_218f.tscn", default_scale=1.0),
        "bear": FloraConfig("res://entities/animals/predators/bear_230f.tscn", default_scale=1.0),
        "alligator": FloraConfig("res://entities/animals/predators/alligator_270f.tscn", default_scale=1.0),
        "snake": FloraConfig("res://entities/animals/predators/snake_170f.tscn", default_scale=1.0),
        "eagle": FloraConfig("res://entities/animals/predators/eagle.tscn", default_scale=1.0),
        "hawk": FloraConfig("res://entities/animals/predators/hawk.tscn", default_scale=1.0),
        # animals, prey
        "deer": FloraConfig("res://entities/animals/prey/deer.tscn", default_scale=1.0),
        "frog": FloraConfig("res://entities/animals/prey/frog.tscn", default_scale=1.0),
        "rabbit": FloraConfig("res://entities/animals/prey/rabbit.tscn", default_scale=1.0),
        "pigeon": FloraConfig("res://entities/animals/prey/pigeon.tscn", default_scale=1.0),
        "squirrel": FloraConfig("res://entities/animals/prey/squirrel.tscn", default_scale=1.0),
        "turtle": FloraConfig("res://entities/animals/prey/turtle.tscn", default_scale=1.0),
        "mouse": FloraConfig("res://entities/animals/prey/mouse.tscn", default_scale=1.0),
        "crow": FloraConfig("res://entities/animals/prey/crow.tscn", default_scale=1.0),
    }

    # see the godot docs for each of these attributes.
    # they are effectively passed straight through to the generated .tscn files.
    sky_config = {
        "resource_local_to_scene": True,
        "sky_top_color": Color(0.705882, 0.87451, 1, 1),
        "sky_horizon_color": Color(1, 0.858824, 0.694118, 1),
        "sky_curve": 0.206137,
        "ground_bottom_color": Color(1, 0.858824, 0.694118, 1),
        "ground_horizon_color": Color(1, 0.858824, 0.694118, 1),
        "ground_curve": 0.112084,
        "sun_color": Color(0.886275, 0.576471, 0.545098, 1),
        "sun_latitude": 160.0,
        "sun_longitude": 180.0,
        "sun_angle_max": 10.0,
        "sun_curve": 0.186603,
        "sun_energy": 5.0,
    }

    # see the godot docs for each of these attributes.
    # they are effectively passed straight through to the generated .tscn files.
    env_config = {
        "resource_local_to_scene": True,
        "background_mode": 2,
        "ambient_light_color": Color(1, 1, 1, 1),
        "ambient_light_energy": 0.7,
        "ambient_light_sky_contribution": 0.0,
        "fog_enabled": export_config.is_fog_enabled,
        "fog_color": Color(0.847059, 0.901961, 0.960784, 1),
        "fog_sun_color": Color(0.992157, 0.960784, 0.847059, 1),
        "fog_sun_amount": 1.0,
        "fog_depth_begin": 20.0,
        "fog_depth_end": 4000.0,
        "fog_depth_curve": 0.404918,
        "fog_transmit_enabled": True,
        "fog_height_min": 100.0,
        "fog_height_max": -90.0,
        "fog_height_curve": 3.03143,
        "tonemap_mode": 2 if export_config.is_postprocessing_enabled else 0,
        "tonemap_white": 12.0,
        "ssao_enabled": export_config.is_ssao_enabled,
        "dof_blur_far_enabled": export_config.is_blur_enabled,
        "dof_blur_far_distance": 200.0,
        "dof_blur_far_transition": 10.0,
        "dof_blur_far_amount": 0.06,
        "dof_blur_far_quality": 2,
        "adjustment_enabled": export_config.is_postprocessing_enabled,
        "adjustment_brightness": 1.1,
        "adjustment_contrast": 1.1,
        "adjustment_saturation": 1.05,
    }

    # see the godot docs for each of these attributes.
    # they are effectively passed straight through to the generated .tscn files.
    sun_config = {
        "light_energy": 0.5,
        "light_color": Color(0.988235, 0.945098, 0.72549, 1),
        "light_indirect_energy": 0.0,
        "light_specular": 1.0,
        "shadow_enabled": export_config.is_sun_shadow_enabled,
        "shadow_color": Color(0.141176, 0.164706, 0.360784, 1),
        "directional_shadow_max_distance": 500.0,
    }

    is_scenery_added = True
    if export_config.scenery_mode == "none":
        is_scenery_added = False

    if export_config.is_legacy_biome_based_on_natural_height_set is not None:
        is_biome_based_on_natural_height = export_config.is_legacy_biome_based_on_natural_height_set
    else:
        is_biome_based_on_natural_height = rand.uniform() < 0.5
    if is_biome_based_on_natural_height:
        min_elevation = 0.0
        max_elevation = max_height / 40.0
        max_shore_height = 8.0
        max_shore_distance = 8.0
        is_beach_computed = True
    else:
        is_beach_computed = rand.uniform() < 0.5

        max_shore_height = rand.uniform(0.0, 8.0)
        max_shore_distance = rand.uniform(0.0, 8.0)

        # randomize the biomes--we're sometimes in the natural world, other times it's all one biome
        is_biome_height_dependent = rand.uniform() < 0.5
        if is_biome_height_dependent:
            max_height_for_biomes = rand.uniform(20.0, 60.0)
            max_elevation = min([max_height / max_height_for_biomes, 1.0])
            min_elevation = 0.0
        else:
            min_elevation = rand.uniform(0.0, 0.9)
            max_elevation = min_elevation + rand.uniform(0.0, 0.4)
            if max_elevation > 1.0:
                max_elevation = 1.0

    # randomize the scale of the noise. The first setting is more local variation, second is more global
    if rand.uniform() < 0.3:
        global_foliage_noise_scale = 0.005
    else:
        global_foliage_noise_scale = 0.05

    biome_config = BiomeConfig(
        flora_config=flora_config,
        godot_env_config=env_config,
        godot_sky_config=sky_config,
        godot_sun_config=sun_config,
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        min_dryness=0.3,
        max_dryness=1.0,
        water_cutoff=0.9,
        noise_scale=0.05,
        noise_magnitude=4.0,
        is_color_per_vertex=False,
        is_normal_per_vertex=False,
        color_map={x.id: hex_to_rgb(x.color) for x in BIOMES},
        biome_matrix=_raw_biome_id_mapping,
        biomes=BIOMES,
        low_slope_threshold=0.5,
        high_slope_threshold=1.5,
        global_foliage_noise_scale=global_foliage_noise_scale,
        global_foliage_noise_min=0.0,
        is_fresh_water_included_in_moisture=False,
        dark_shore_distance=2.0,
        dark_shore_height=1.0,
        max_shore_distance=max_shore_distance,
        max_shore_height=max_shore_height,
        beach_slope_cutoff=0.5,
        min_beach_length=10.0,
        beach_shrink_fraction=0.0,
        beach_fade_distance=1.0,
        swamp_elevation_max=3.0,
        swamp_distance_max=3.0,
        swamp_noise_scale=0.01,
        swamp_noise_magnitude=1.0,
        color_sampling_neighbor_count=5,
        color_sampling_std_dev=2.0,
        coast_color_noise=0.0,
        rock_color_noise=0.0,
        coastal_id=COASTAL_ID,
        rock_biome_ids=ROCK_IDS,
        foliage_density_modifier=foliage_density_modifier,
        max_kd_points=None if export_config.is_border_calculation_detailed else 500,
        is_terrain_noise_shader_enabled=export_config.is_terrain_shader_enabled,
        is_scenery_added=is_scenery_added,
        is_beach_computed=is_beach_computed,
    )

    return biome_config
