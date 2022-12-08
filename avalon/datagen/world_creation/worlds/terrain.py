from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union
from typing import cast

import numpy as np
from godot_parser import GDObject
from godot_parser import Node as GDNode
from godot_parser.tree import Tree
from numpy.random import Generator
from scipy.spatial import Delaunay

from avalon.common.utils import first
from avalon.datagen.godot_base_types import FloatRange
from avalon.datagen.world_creation.configs.flora import FloraConfig
from avalon.datagen.world_creation.debug_plots import IS_DEBUG_VIS
from avalon.datagen.world_creation.debug_plots import plot_points
from avalon.datagen.world_creation.debug_plots import plot_triangulation
from avalon.datagen.world_creation.entities.scenery import Scenery
from avalon.datagen.world_creation.region import EdgedRegion
from avalon.datagen.world_creation.region import Region
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.types import Point2DListNP
from avalon.datagen.world_creation.types import RawGDObject
from avalon.datagen.world_creation.utils import ARRAY_MESH_TEMPLATE
from avalon.datagen.world_creation.worlds.biome_map import BiomeMap
from avalon.datagen.world_creation.worlds.height_map import get_flora_config_by_file


class Terrain:
    def __init__(
        self,
        biome_map: BiomeMap,
        is_climbable: MapBoolNP,
        is_detail_important: MapBoolNP,
        point_density_in_points_per_square_meter: float,
        rand: np.random.Generator,
        special_regions: Tuple[Region, ...] = tuple(),
    ) -> None:
        self.biome_map = biome_map
        self.height_map = biome_map.map
        self.fine_triangulation = TerrainTriangulation(
            biome_map,
            is_climbable,
            point_density_in_points_per_square_meter,
            rand,
            special_regions,
            extra_points=biome_map.create_extra_height_points(rand, is_climbable, is_detail_important),
        )
        self.coarse_triangulation = TerrainTriangulation(
            biome_map, is_climbable, point_density_in_points_per_square_meter / 4.0, rand, special_regions
        )

    # TODO: reorganize the order of calls to make a bit more sense
    def export(
        self,
        fine_output_path: Path,
        coarse_output_path: Path,
        region: Region,
        tile_x: int,
        tile_z: int,
        tile_radius: int,
        building_names: List[str],
        trees_by_tile_ids_and_resource: Dict[Tuple[int, int], Dict[str, List[Scenery]]],
        flora_by_tile_ids_and_resource: Dict[Tuple[int, int], Dict[str, List[Scenery]]],
        distant_tile_ids: Set[Tuple[int, int]],
        neighboring_and_this_tile_ids: Set[Tuple[int, int]],
    ):
        (
            global_triangle_indices_for_region,
            region_triangles,
            region_triangle_normals,
            region_vertex_normals,
            region_vertices,
            region_colors,
        ) = self._get_export_data_for_region(region)

        scene = GodotScene()

        with scene.use_tree() as tree:
            tree.root = _create_terrain_info_node(
                scene,
                tile_x,
                tile_z,
                building_names,
            )

            tree.root.add_child(
                _create_terrain_mesh_node(
                    scene,
                    region_triangles,
                    region_vertices,
                    region_vertex_normals,
                    region_colors,
                    self.biome_map.config.is_terrain_noise_shader_enabled,
                    is_fine=True,
                )
            )

            # add trees for this tile
            our_tile_id_set = {(tile_z, tile_x)}
            tree_list_by_resource = get_scenery_with_tile_ids(trees_by_tile_ids_and_resource, our_tile_id_set)
            for tree_resource, tree_list in tree_list_by_resource.items():
                first_tree = first(tree_list)
                assert first_tree is not None
                tree.root.add_child(
                    create_multimesh_instance(
                        scene,
                        f"high_tree_{tile_x}_{tile_z}",
                        tree_list,
                        first_tree.resource_file.replace(".tscn", ".res"),
                        self.biome_map.config.flora_config,
                    )
                )

            # if this is a single-tile export scenario, include the flora in here
            if len(neighboring_and_this_tile_ids) == 1:
                self._add_tile_local_flora(
                    flora_by_tile_ids_and_resource, neighboring_and_this_tile_ids, scene, tile_x, tile_z, tree
                )

        scene.write(str(fine_output_path.absolute()))

        # create the file that shows the whole rest of the world
        distant_export_data = self._get_export_data_outside_of_region(region, tile_radius)

        scene = GodotScene()
        with scene.use_tree() as tree:
            if distant_export_data is not None:
                (
                    selected_triangle_indices,
                    distant_triangles,
                    distant_triangle_normals,
                    distant_vertex_normals,
                    distant_vertices,
                    distant_colors,
                ) = distant_export_data

                tree.root = _create_terrain_mesh_node(
                    scene,
                    distant_triangles,
                    distant_vertices,
                    distant_vertex_normals,
                    distant_colors,
                    self.biome_map.config.is_terrain_noise_shader_enabled,
                    is_fine=False,
                )
            else:
                tree.root = GDNode(
                    "DistantData",
                    type="Spatial",
                    properties={"transform": GDObject("Transform", 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)},
                )

            tree_list_by_resource = get_scenery_with_tile_ids(trees_by_tile_ids_and_resource, distant_tile_ids)
            for tree_resource, tree_list in tree_list_by_resource.items():
                first_tree = first(tree_list)
                assert first_tree is not None
                tree.root.add_child(
                    create_multimesh_instance(
                        scene,
                        f"high_tree_{tile_x}_{tile_z}",
                        tree_list,
                        first_tree.resource_file.replace(".tscn", "_low.res"),
                        self.biome_map.config.flora_config,
                    )
                )

            if len(neighboring_and_this_tile_ids) > 1:
                # get the other scenery from the neighboring tiles and combine with our own
                self._add_tile_local_flora(
                    flora_by_tile_ids_and_resource, neighboring_and_this_tile_ids, scene, tile_x, tile_z, tree
                )

        scene.write(str(coarse_output_path.absolute()))

        return region_triangles, region_vertices

    def _add_tile_local_flora(
        self,
        flora_by_tile_ids_and_resource: Dict[Tuple[int, int], Dict[str, List[Scenery]]],
        neighboring_and_this_tile_ids: Set[Tuple[int, int]],
        scene: GodotScene,
        tile_x: int,
        tile_z: int,
        tree: Tree,
    ) -> None:
        flora_list_by_resource = get_scenery_with_tile_ids(
            flora_by_tile_ids_and_resource, neighboring_and_this_tile_ids
        )
        for flora_resource, flora_list in flora_list_by_resource.items():
            first_flora = first(flora_list)
            assert first_flora is not None
            tree.root.add_child(
                create_multimesh_instance(
                    scene,
                    f"high_tree_{tile_x}_{tile_z}",
                    flora_list,
                    first_flora.resource_file.replace(".tscn", ".res"),
                    self.biome_map.config.flora_config,
                )
            )

    def _get_export_data_for_region(
        self, region: Region
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        global_triangle_indices = self._are_triangles_in_tile(
            self.fine_triangulation.triangle_points, region
        ).nonzero()[0]

        result = _get_export_data_inner(global_triangle_indices, self.fine_triangulation)
        assert result is not None, "No triangles in region?"
        return result

    def _get_export_data_outside_of_region(
        self, region: Region, num_tiles: int
    ) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        x_min = region.x.min_ge - num_tiles * region.x.size
        x_max = x_min + region.x.size * (2 * num_tiles + 1)
        z_min = region.z.min_ge - num_tiles * region.z.size
        z_max = z_min + region.z.size * (2 * num_tiles + 1)
        loaded_region = Region(
            x=FloatRange(x_min, x_max),
            z=FloatRange(z_min, z_max),
        )

        global_triangle_indices = self._are_triangles_outside_tile(
            self.coarse_triangulation.triangle_points, loaded_region
        ).nonzero()[0]

        return _get_export_data_inner(global_triangle_indices, self.coarse_triangulation)

    def _are_triangles_in_tile(self, triangle_points: np.ndarray, region: Region) -> np.ndarray:
        """Returns a boolean numpy array indicating the triangles that fall in the region"""
        return cast(
            np.ndarray,
            (
                (triangle_points[:, :, 0] >= region.x.min_ge).any(axis=-1)
                & (triangle_points[:, :, 0] < region.x.max_lt).any(axis=-1)
                & (triangle_points[:, :, 2] >= region.z.min_ge).any(axis=-1)
                & (triangle_points[:, :, 2] < region.z.max_lt).any(axis=-1)
            ),
        )

    def _are_triangles_outside_tile(self, triangle_points: np.ndarray, region: Region) -> np.ndarray:
        """Returns a boolean numpy array indicating the triangles that fall outside of region"""
        return cast(
            np.ndarray,
            (
                (triangle_points[:, :, 0] < region.x.min_ge).any(axis=-1)
                | (triangle_points[:, :, 0] >= region.x.max_lt).any(axis=-1)
                | (triangle_points[:, :, 2] < region.z.min_ge).any(axis=-1)
                | (triangle_points[:, :, 2] >= region.z.max_lt).any(axis=-1)
            ),
        )


#
# def create_triset(
#     geom: Geometry, input_list: InputList, triangle_indices: np.ndarray, material_ref: str
# ) -> TriangleSet:
#     flat_triangle_indices = triangle_indices.flatten()
#     flat_triangle_indices = np.reshape(flat_triangle_indices, (len(flat_triangle_indices), 1))
#     flat_normal_indices = flat_triangle_indices
#     indices = np.concatenate((flat_triangle_indices, flat_normal_indices), axis=1).flatten()
#     triset = geom.createTriangleSet(indices, input_list, material_ref)
#     return triset


def normalize_v3(arr: np.ndarray) -> np.ndarray:
    """Normalize a numpy array of 3 component vectors shape=(n,3)"""
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


# roughly from here: https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
def calculate_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]

    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0]) * -1.0

    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)

    return cast(np.ndarray, n)


def calculate_normals_for_faces_and_vertices(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)

    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]

    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])

    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)

    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    # normalize_v3(norm)
    # so I replaced with this for now, which is significantly slower (10%)
    lens = np.sqrt(norm[:, 0] ** 2 + norm[:, 1] ** 2 + norm[:, 2] ** 2)
    norm[:, 0] = np.divide(norm[:, 0], lens, out=np.zeros_like(lens), where=lens != 0)
    norm[:, 1] = np.divide(norm[:, 1], lens, out=np.ones_like(lens), where=lens != 0)
    norm[:, 2] = np.divide(norm[:, 2], lens, out=np.zeros_like(lens), where=lens != 0)

    return norm, n


def _create_triangulation(
    rand: Generator,
    edge_region: EdgedRegion,
    point_density_in_points_per_square_meter: float,
    special_regions: Tuple[Region, ...] = tuple(),
    extra_points: Optional[np.ndarray] = None,
) -> Delaunay:
    """
    Create random points within a region and Delaunay triangulate them to create a Terrain object

    Helper for Terrain
    """
    area = edge_region.x.size * edge_region.z.size
    num_points = round(area * point_density_in_points_per_square_meter)
    points = edge_region.get_randomish_points(num_points)

    if IS_DEBUG_VIS:
        plot_points(points, 0, 1)

    if extra_points is not None:
        points = np.concatenate([points, extra_points])

    if IS_DEBUG_VIS:
        plot_points(points, 0, 1)

    for region in special_regions:
        assert edge_region.contains_region(
            region
        ), f"Some special regions are not contained in the tile region: {region}"
        points = np.array([x for x in points if not region.contains_point_2d(x)] + region.points)
    edge_points = np.array([(x[0], x[2]) for x in edge_region.edge_vertices])
    points = np.concatenate((edge_points, points))
    if IS_DEBUG_VIS:
        plot_points(points, 0, 1)
    return Delaunay(points)


class TerrainTriangulation:
    def __init__(
        self,
        biome_map: BiomeMap,
        is_climbable: MapBoolNP,
        point_density_in_points_per_square_meter: float,
        rand: np.random.Generator,
        special_regions: Tuple[Region, ...] = tuple(),
        extra_points: Optional[Point2DListNP] = None,
    ) -> None:
        self.config = biome_map.config
        height_map = biome_map.map
        points_per_meter = point_density_in_points_per_square_meter**0.5
        edge_region = height_map.create_edge_region(height_map.region, points_per_meter, rand)
        triangulation = _create_triangulation(
            rand,
            edge_region,
            point_density_in_points_per_square_meter,
            special_regions,
            extra_points,
        )
        self.neighbors = triangulation.neighbors
        if IS_DEBUG_VIS:
            plot_triangulation(triangulation)
        heights = height_map.get_heights(triangulation.points)
        height_by_id = {i: heights[i] for i in range(len(triangulation.points))}
        # reset the edge vertices to whatever heights are specified
        for i, vertex in enumerate(edge_region.edge_vertices):
            height_by_id[i] = vertex[1]

        # convert to 3D points from Delaunay triangulation
        self.vertices = np.array([(x[0], height_by_id[i], x[1]) for i, x in enumerate(triangulation.points)])

        self.triangles = triangulation.simplices

        self.triangle_points = self.vertices[self.triangles]

        # only calculate vertex normals if required
        if biome_map.config.is_normal_per_vertex or biome_map.config.is_color_per_vertex:
            vertex_normals, triangle_normals = calculate_normals_for_faces_and_vertices(self.vertices, self.triangles)
            vertex_normals *= -1

        # set per vertex or per face normals, as configured
        if biome_map.config.is_normal_per_vertex:
            self.vertex_normals, self.triangle_normals = vertex_normals, triangle_normals
        else:
            self.triangle_normals = calculate_normals(self.vertices, self.triangles)

        self.midpoints = np.array([self.vertices[x].sum(axis=0) / 3.0 for x in self.triangles])

        # set per vertex or per face colors, as configured
        if biome_map.config.is_color_per_vertex:
            self.vertex_colors = biome_map.get_colors(rand, is_climbable, self.vertices, vertex_normals)
        else:
            self.triangle_colors = biome_map.get_colors(rand, is_climbable, self.midpoints, self.triangle_normals)


def get_scenery_with_tile_ids(
    scenery_by_tile_ids_and_resource: Dict[Tuple[int, int], Dict[str, List[Scenery]]], tile_ids: Set[Tuple[int, int]]
) -> Dict[str, List[Scenery]]:
    results: Dict[str, List[Scenery]] = {}
    for tile_id, inner_dict in scenery_by_tile_ids_and_resource.items():
        if tile_id in tile_ids:
            for resource, scenery in inner_dict.items():
                if resource not in results:
                    results[resource] = []
                results[resource].extend(scenery)
    return results


def _get_export_data_inner(
    global_triangle_indices: np.ndarray, triangulation: TerrainTriangulation
) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if len(global_triangle_indices) == 0:
        return None
    triangles = triangulation.triangles[global_triangle_indices]
    triangle_normals = triangulation.triangle_normals[global_triangle_indices]

    is_color_per_vertex = triangulation.config.is_color_per_vertex
    is_normal_per_vertex = triangulation.config.is_normal_per_vertex

    new_triangles = np.array([(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(triangles))])
    vertices = []
    colors = []
    normals = []
    for triangle_index in global_triangle_indices:
        triangle = triangulation.triangles[triangle_index]
        for vertex in triangle:
            vertices.append(triangulation.vertices[vertex])
            # colors.append(triangulation.triangle_colors[triangle_index])
            # normals.append(triangulation.triangle_normals[triangle_index])
            colors.append(
                triangulation.vertex_colors[vertex]
                if is_color_per_vertex
                else triangulation.triangle_colors[triangle_index]
            )
            normals.append(
                triangulation.vertex_normals[vertex]
                if is_normal_per_vertex
                else triangulation.triangle_normals[triangle_index]
            )

    return (
        global_triangle_indices,
        new_triangles,
        triangle_normals,
        np.array(normals),
        np.array(vertices),
        np.array(colors),
    )


def _create_terrain_mesh_node(
    scene: GodotScene,
    material_triangles: np.ndarray,
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    colors: np.ndarray,
    is_terrain_noise_shader_enabled: bool,
    is_fine: bool,
) -> GDNode:
    surface_kwargs = {}
    surface_id = 0
    if is_terrain_noise_shader_enabled:
        material_resource = scene.add_ext_resource("res://materials/noise_terrain.material", "Material")
    else:
        material_resource = scene.add_ext_resource("res://materials/terrain.material", "Material")
    aabb_corner = [x.min() for x in [vertices[:, 0], vertices[:, 1], vertices[:, 2]]]
    aabb_maxes = [x.max() for x in [vertices[:, 0], vertices[:, 1], vertices[:, 2]]]
    aabb_size = tuple(aabb_maxes[i] - aabb_corner[i] for i in range(3))
    aabb = (*aabb_corner, *aabb_size)
    alpha_colors = np.concatenate([colors, np.ones((len(colors), 1))], axis=1)
    surface_kwargs[f"surfaces/{surface_id}"] = RawGDObject(
        ARRAY_MESH_TEMPLATE.format(
            aabb=", ".join(str(x) for x in aabb),
            vertex_floats=_print_array(vertices.flatten(), 3),
            vertex_normal_floats=_print_array(vertex_normals.flatten(), 3),
            color_floats=_print_array(alpha_colors.flatten(), 5),
            triangle_indices=_print_array(material_triangles.flatten(), 1),
            index_count=str(len(material_triangles)),
            vertex_count=str(len(vertices)),
            mesh_name="terrain",
            material_resource_type="ExtResource",
            material_id=material_resource.id,
        )
    )
    surface_id += 1
    terrain_mesh = scene.add_sub_resource("ArrayMesh", resource_name="terrain", **surface_kwargs)
    terrain_mesh_node = GDNode(
        "terrain_mesh" if is_fine else "distant_mesh",
        type="MeshInstance",
        properties={"mesh": terrain_mesh.reference, "material/0": "null"},
    )
    return terrain_mesh_node


def _create_static_body(index: int, scene: GodotScene, triangles: np.ndarray, vertices: np.ndarray) -> GDNode:

    collision_mesh_vertices = []
    for triangle in triangles:
        collision_mesh_vertices.append(vertices[triangle[0]])
        collision_mesh_vertices.append(vertices[triangle[1]])
        collision_mesh_vertices.append(vertices[triangle[2]])
    collision_mesh_floats = _print_array(np.array(collision_mesh_vertices).flatten(), 3)
    collision_shape = scene.add_sub_resource(
        "ConcavePolygonShape",
        data=RawGDObject(
            "PoolVector3Array( {collision_mesh_floats} )".format(collision_mesh_floats=collision_mesh_floats)
        ),
    )
    collision_shape_node = GDNode(
        f"static_collision_shape_{index}", type="CollisionShape", properties={"shape": collision_shape.reference}
    )
    physics_material = scene.add_sub_resource("PhysicsMaterial", bounce=0.0, absorbent="false")
    static_body = GDNode(
        f"static_collision_{index}",
        type="StaticBody",
        properties={"physics_material_override": physics_material.reference},
    )
    static_body.add_child(collision_shape_node)
    return static_body


def _create_terrain_info_node(
    scene: GodotScene,
    tile_x: int,
    tile_z: int,
    building_names: List[str],
) -> GDNode:
    quoted_building_names = [f'"{x}"' for x in building_names]
    info_node = GDNode(
        "terrain",
        type="Spatial",
        properties={
            "script": scene.add_ext_resource("res://terrain/terrain_tile.gd", "Script").reference,
            "tile_x": tile_x,
            "tile_z": tile_z,
            "building_names": RawGDObject("[ " + ", ".join(quoted_building_names) + " ]"),
        },
    )
    return info_node


def _print_array(array: np.ndarray, precision: int) -> str:
    # annoyingly, this is MUCH slower than the below...
    # return np.array2string(
    #     array,
    #     threshold=np.inf,
    #     precision=precision,
    #     max_line_width=np.inf,
    #     suppress_small=True,
    #     floatmode="fixed",
    #     separator=", ",
    # )[1:-1]
    return ", ".join([f"{x:.{precision}f}" for x in array.flatten()])


def create_multimesh_instance(
    scene: GodotScene,
    name: str,
    instances: list[Scenery],
    resource_file: str,
    flora_config: Dict[str, FloraConfig],
):

    config, resource_name = get_flora_config_by_file(flora_config, resource_file)
    assert config is not None
    if config.is_noise_shader_enabled:
        material_resource = scene.add_ext_resource("res://materials/flora.material", "Material")
    else:
        material_resource = None

    transform_data = np.array(
        [
            np.array([*((x.rotation.reshape((3, 3)) * x.scale * config.default_scale).flatten()), *x.position])
            for x in instances
        ]
    ).flatten()

    transform_floats = _print_array(np.array(transform_data).flatten(), 3)

    mesh_resource = scene.add_ext_resource(resource_file, "ArrayMesh")
    multimesh_resource = scene.add_sub_resource(
        "MultiMesh",
        transform_format=1,
        instance_count=len(instances),
        mesh=mesh_resource.reference,
        transform_array=RawGDObject(
            "PoolVector3Array( {transform_floats} )".format(transform_floats=transform_floats)
        ),
    )
    return GDNode(
        f"{name}_{resource_name}",
        type="MultiMeshInstance",
        properties={
            "transform": GDObject("Transform", 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
            "multimesh": multimesh_resource.reference,
            "material_override": material_resource.reference if material_resource is not None else None,
            "cast_shadow": 1 if config.is_shadowed else 0,
        },
    )
