from typing import Optional

import attr
import numpy as np

from datagen.world_creation.biome_map import DARK_SHORE
from datagen.world_creation.biome_map import FRESH_WATER
from datagen.world_creation.biome_map import SWAMP
from datagen.world_creation.biome_map import UNCLIMBABLE_BIOME_ID
from datagen.world_creation.biome_map import BorderMode
from datagen.world_creation.biome_map import SceneryConfig
from datagen.world_creation.biome_map import hex_to_rgb
from datagen.world_creation.biome_map import make_biome
from datagen.world_creation.biome_map import plot_biome_grid
from datagen.world_creation.heightmap import Biome
from datagen.world_creation.heightmap import BiomeConfig
from datagen.world_creation.heightmap import Color
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.heightmap import FloraConfig
from datagen.world_creation.heightmap import perlin
from datagen.world_creation.heightmap import plot_value_grid
from datagen.world_creation.indoor.objects import BuildingAestheticsConfig
from datagen.world_creation.items import Scenery
from datagen.world_creation.new_world import NewWorld


def generate_biome_config(
    rand: np.random.Generator,
    export_config: ExportConfig,
    max_height: float,
    foliage_density_modifier: Optional[float] = 1.0,
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

    if export_config.is_biome_based_on_natural_height:
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


def generate_aesthetics_config() -> BuildingAestheticsConfig:
    return BuildingAestheticsConfig()


def make_natural_biomes(rand: np.random.Generator, world: NewWorld, is_debug_graph_printing_enabled: bool = False):
    biome_config = world.biome_config
    special_biomes = world.special_biomes

    biome_map = make_biome(
        world.map, biome_config, rand, special_biomes, is_debug_graph_printing_enabled=is_debug_graph_printing_enabled
    )

    if not biome_config.is_scenery_added:
        return biome_map

    global_foliage_noise = perlin(
        world.map.Z.shape,
        biome_config.global_foliage_noise_scale,
        rand,
        is_normalized=True,
        noise_min=biome_config.global_foliage_noise_min,
    )

    # plot_value_grid(global_foliage_noise, "global foliage noise")

    want_to_plot_noise = False
    if want_to_plot_noise:
        plot_value_grid(world.make_biome_noise(rand, "", 0.0), "example biome foliage noise")

    want_overall_plot = is_debug_graph_printing_enabled
    if want_overall_plot:
        plot_value_grid(world.map.Z)
        plot_biome_grid(biome_map.biome_id, biome_config, "final biome map")

    # apply the flora mask from the world so we can avoid sticking trees in buildings, etc
    global_foliage_noise *= world.flora_mask

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
        world.items.extend(
            biome_map.get_random_points_in_biome(
                rand,
                config=SceneryConfig(
                    resource_file=file,
                    biome_id=biome_id,
                    density=density * overall_density,
                    border_distance=0.0,
                    border_mode=BorderMode.HARD,
                ),
                placement_noise=global_foliage_noise * world.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
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
        world.items.extend(
            biome_map.get_random_points_in_biome(
                rand,
                config=SceneryConfig(
                    resource_file=file,
                    biome_id=biome_id,
                    density=density * overall_density,
                    border_distance=0.0,
                    border_mode=BorderMode.HARD,
                ),
                placement_noise=global_foliage_noise * world.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
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
        world.items.extend(
            biome_map.get_random_points_in_biome(
                rand,
                config=SceneryConfig(
                    resource_file=file,
                    biome_id=biome_id,
                    density=density * overall_density,
                    border_distance=0.0,
                    border_mode=BorderMode.HARD,
                ),
                placement_noise=global_foliage_noise * world.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
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
        world.items.extend(
            biome_map.get_random_points_in_biome(
                rand,
                config=SceneryConfig(
                    resource_file=file,
                    biome_id=biome_id,
                    density=density * overall_density,
                    border_distance=0.0,
                    border_mode=BorderMode.HARD,
                ),
                placement_noise=global_foliage_noise * world.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
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
        world.items.extend(
            biome_map.get_random_points_in_biome(
                rand,
                config=SceneryConfig(
                    resource_file=file,
                    biome_id=biome_id,
                    density=density * overall_density,
                    border_distance=0.0,
                    border_mode=BorderMode.HARD,
                ),
                placement_noise=global_foliage_noise * world.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
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
        world.items.extend(
            biome_map.get_random_points_in_biome(
                rand,
                config=SceneryConfig(
                    resource_file=file,
                    biome_id=biome_id,
                    density=density * overall_density,
                    border_distance=0.0,
                    border_mode=BorderMode.HARD,
                ),
                placement_noise=global_foliage_noise * world.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
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
        world.items.extend(
            biome_map.get_random_points_in_biome(
                rand,
                config=SceneryConfig(
                    resource_file=file,
                    biome_id=biome_id,
                    density=density * overall_density,
                    border_distance=0.0,
                    border_mode=BorderMode.HARD,
                ),
                placement_noise=global_foliage_noise * world.make_biome_noise(rand, file, 0.5, noise_scale=0.05),
                is_debug_graph_printing_enabled=False,
            )
        )

    if world.export_config:
        if world.export_config.scenery_mode == "single":
            new_items = []
            for i in range(len(world.items)):
                item = world.items[i]
                if isinstance(item, Scenery):
                    if "trees/" in item.resource_file:
                        new_items.append(attr.evolve(item, resource_file="res://scenery/trees/fir.tscn"))
                    else:
                        pass
                else:
                    new_items.append(item)
            world.items = new_items
        if world.export_config.scenery_mode == "tree":
            new_items = []
            for i in range(len(world.items)):
                item = world.items[i]
                if isinstance(item, Scenery):
                    if "trees/" in item.resource_file:
                        new_items.append(item)
                    else:
                        pass
                else:
                    new_items.append(item)
            world.items = new_items

    return biome_map
