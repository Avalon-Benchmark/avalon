from typing import Optional

import attr

from avalon.contrib.serialization import Serializable
from avalon.datagen.world_creation.types import DebugVisualizationConfig


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ExportConfig(Serializable):
    name: str
    is_tiled: bool
    is_terrain_shader_enabled: bool
    is_flora_shader_enabled: bool
    is_sun_enabled: bool
    is_sun_shadow_enabled: bool
    is_ssao_enabled: bool
    is_fog_enabled: bool
    is_blur_enabled: bool
    is_postprocessing_enabled: bool
    is_indoor_lighting_enabled: bool
    scenery_mode: str
    is_minor_scenery_hidden: bool
    is_biome_fast: bool
    is_exported_with_absolute_paths: bool
    is_border_calculation_detailed: bool
    is_meta_data_exported: bool
    world_id: Optional[str] = None
    debug_visualization_config: Optional[DebugVisualizationConfig] = None
    # TODO(mjr) added back for regression tests. Should probably disentangle them from world generation
    is_legacy_biome_based_on_natural_height_set: Optional[bool] = None


def get_oculus_export_config(world_id: Optional[str] = None) -> ExportConfig:
    return ExportConfig(
        name="oculus",
        is_tiled=True,
        is_terrain_shader_enabled=False,
        is_flora_shader_enabled=False,
        is_sun_enabled=True,
        is_sun_shadow_enabled=True,
        is_ssao_enabled=True,
        is_fog_enabled=True,
        is_blur_enabled=False,
        is_postprocessing_enabled=True,
        is_indoor_lighting_enabled=True,
        scenery_mode="normal",
        is_minor_scenery_hidden=False,
        is_biome_fast=False,
        is_exported_with_absolute_paths=False,
        is_border_calculation_detailed=True,
        is_meta_data_exported=True,
        world_id=world_id,
    )


def get_eval_agent_export_config() -> ExportConfig:
    return attr.evolve(
        get_oculus_export_config(),
        # evaluation is very similar to the oculus config, with only minor (basically graphical) differences
        name="agent",
        # the biggest difference is that the levels are never "tiled" for agents.
        # the tiling takes MUCH longer to export for larger levels, and is likely only marginally faster
        # at runtime, and thus isn't really worth the trade-off (unless you really want to scale your world generation
        # workers out like crazy)
        is_tiled=False,
        # we disable ssao, fog, postprocessing, and lighting so that the simulator is faster
        # none of thes options makes a meaningful difference graphically, but they look nicer for players
        # also, postprocessing is used for players to indicate health
        is_ssao_enabled=False,
        is_fog_enabled=False,
        is_postprocessing_enabled=False,
        is_indoor_lighting_enabled=False,
        # minor scenery (ie, not trees, ie, things that are not collidable) should not be visible to the agent
        # becauase it does not train with them (because they don't make a big difference, and takes much longer
        # to generate all of them). For the evaluation mode, we generate them then hide them so that the level gen
        # code will work out to produce exactly the same levels as with the oculus human export config
        # (while in training we dont even bother generating them at all in the first place)
        is_minor_scenery_hidden=True,
        # the way godot loads levels is different for the agent and the human players, and requires absolute paths
        is_exported_with_absolute_paths=True,
    )


def get_agent_export_config() -> ExportConfig:
    return attr.evolve(
        get_eval_agent_export_config(),
        # training only differs in a very small number of ways
        # first, we just avoid generating any of the non-tree (non-collision) scenery), since the agents don't even
        # see that anyway in the evaluation settings (it's just to make the worlds look nicer for humans)
        scenery_mode="tree",
        # next, we dont care about making super accurate border calculations. They take a while, but dont really
        # change the world that much. Again, just makes it look nicer for players basically.
        is_border_calculation_detailed=False,
        # finally, no need to export the meta data file during training, because we dont use it at all
        is_meta_data_exported=False,
    )


def get_pretty_export_config() -> ExportConfig:
    return ExportConfig(
        name="pretty",
        is_tiled=False,
        is_terrain_shader_enabled=True,
        is_flora_shader_enabled=True,
        is_sun_enabled=True,
        is_sun_shadow_enabled=True,
        is_ssao_enabled=True,
        is_fog_enabled=True,
        is_blur_enabled=True,
        is_postprocessing_enabled=True,
        is_indoor_lighting_enabled=True,
        scenery_mode="normal",
        is_minor_scenery_hidden=False,
        is_biome_fast=False,
        is_exported_with_absolute_paths=False,
        is_border_calculation_detailed=True,
        is_meta_data_exported=True,
    )


def get_mouse_and_keyboard_export_config() -> ExportConfig:
    return ExportConfig(
        name="mouse_and_keyboard",
        is_tiled=False,
        is_terrain_shader_enabled=True,
        is_flora_shader_enabled=True,
        is_sun_enabled=True,
        is_sun_shadow_enabled=True,
        is_ssao_enabled=True,
        is_fog_enabled=True,
        is_blur_enabled=True,
        is_postprocessing_enabled=True,
        is_indoor_lighting_enabled=False,
        scenery_mode="normal",
        is_minor_scenery_hidden=False,
        is_biome_fast=False,
        is_exported_with_absolute_paths=False,
        is_border_calculation_detailed=True,
        is_meta_data_exported=True,
    )
