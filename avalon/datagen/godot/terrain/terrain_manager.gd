extends Node

class_name TerrainManager

export var export_path: String
export var x_tile_count: int
export var z_tile_count: int
export var x_min: float
export var z_min: float
export var x_max: float
export var z_max: float
export var tile_radius: float
export var climb_map: PoolByteArray
export var climb_map_x: int
export var climb_map_y: int

var current_tile: Vector2
var x_tile_size: float
var z_tile_size: float
var num_tiles_to_loaded_edge: int = 1
var world_wall_dist: float

var PERF_SHADOWS_ENABLED: bool = ProjectSettings.get_setting("avalon/shadows_enabled")


func _ready():
	# TODO: is weird that there are two sizes, we assume they are square
	x_tile_size = (x_max - x_min) / x_tile_count
	z_tile_size = (z_max - z_min) / z_tile_count

	var start_tile = $tiles.get_child(0)
	on_tile_entered(start_tile.tile_x, start_tile.tile_z)

	world_wall_dist = num_tiles_to_loaded_edge * x_tile_size * 1.5
	# creates walls around the entire world to prevent anything from falling off
	var world_size = x_max - x_min
	for wall_array in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
		var wall = StaticBody.new()
		wall.translation = Vector3(wall_array[0] * world_size, 0.0, wall_array[1] * world_size)
		var box = BoxShape.new()
		box.extents = Vector3(world_size / 2.0, world_size, world_size / 2.0)
		var collision = CollisionShape.new()
		collision.shape = box
		wall.add_child(collision)
		self.add_child(wall)

	var sun: DirectionalLight = get_parent().get_node("Sun")
	sun.shadow_enabled = PERF_SHADOWS_ENABLED


func _physics_process(_delta):
	if (x_tile_count <= 1) and (z_tile_count <= 1):
		return
	# TODO call `player.get_position()`
	var pos = get_tree().root.find_node("physical_body", true, false).global_transform.origin
	var tile_pos = _pos_to_tile_pos(pos)
	if tile_pos != current_tile:
		on_tile_entered(tile_pos[0], tile_pos[1])
		current_tile = tile_pos


func is_position_climbable(pos: Vector3) -> bool:
	if climb_map.size() == 0:
		return false
	# TODO: move this logic somewhere that makes more sense
	var bytes_per_row = int(floor(climb_map_x / 8.0))
	if climb_map_x % 8 > 0:
		bytes_per_row += 1
	var climb_map_row = int(((min(pos.z, z_max - 0.001) - z_min) / (z_max - z_min)) * climb_map_y)
	var climb_map_col = int(((min(pos.x, x_max - 0.001) - x_min) / (x_max - x_min)) * climb_map_x)
#		printt(climb_map_row, climb_map_col)
	var climb_map_col_base = climb_map_col / 8
	var climb_map_col_offset = climb_map_col % 8
	var entry = climb_map[bytes_per_row * climb_map_row + climb_map_col_base]
	if get_bit(climb_map_col_offset, entry):
#		print("Player is in unclimbable region")
		return false
	else:
		return true


func get_bit(n: int, value: int) -> int:
	if value < 0 || n < 0:
		print("warning: get_bit args out of bounds. n: ", n, ", value: ", value)
		return 0
	var _state: bool = value & (1 << n) != 0
	return 1 if _state else 0


func _pos_to_tile_pos(pos: Vector3) -> Vector2:
	var x = floor((pos.x - x_min) / x_tile_size)
	if x < 0.0:
		x = 0.0
	if x > x_tile_count - 1:
		x = x_tile_count - 1
	var z = floor((pos.z - z_min) / z_tile_size)
	if z < 0.0:
		z = 0.0
	if z > z_tile_count - 1:
		z = z_tile_count - 1
	return Vector2(x, z)


func _get_desired_tiles(x_idx, z_idx):
	var neighboring_and_this_tile_ids = []
	for i in range(-tile_radius, tile_radius + 1):
		var other_tile_x = x_idx + i
		if other_tile_x < 0 or other_tile_x >= x_tile_count:
			continue
		for j in range(-tile_radius, tile_radius + 1):
			var other_tile_z = z_idx + j
			if other_tile_z < 0 or other_tile_z >= z_tile_count:
				continue
			neighboring_and_this_tile_ids.append(Vector2(other_tile_x, other_tile_z))
	return neighboring_and_this_tile_ids


func on_tile_entered(x: int, z: int):
	# prints("Entering", x, z)
	var loaded_tiles = get_loaded_tiles()
	var desired_tiles = _get_desired_tiles(x, z)

	# load anything not already loaded
	for tile_pos in desired_tiles:
		if tile_pos[0] < 0 or tile_pos[0] >= x_tile_count:
			continue
		if tile_pos[1] < 0 or tile_pos[1] >= z_tile_count:
			continue
		if not loaded_tiles.has(tile_pos):
			load_tile(tile_pos)

	# unload anything no longer needed
	for tile_pos in loaded_tiles:
		if not tile_pos in desired_tiles:
			unload_tile(tile_pos)

	# figure out all buildings that should be loaded
	var desired_buildings = {}
	for tile in $tiles.get_children():
		if tile is TerrainTile:
			for building in tile.building_names:
				desired_buildings[building] = true

	# toggle the visibility of each building to be correct
	for building in $buildings.get_children():
		var high_poly = building.find_node("high_poly", true, false)
		var low_poly = building.find_node("low_poly", true, false)
		if building.name in desired_buildings:
			high_poly.show()
			low_poly.hide()
		else:
			high_poly.hide()
			low_poly.show()

	var current_distant_mesh = self.find_node("distant_mesh", false, false)
	if current_distant_mesh:
		self.remove_child(current_distant_mesh)

	var distant_mesh_name = "%s/distant_%s_%s.tscn" % [export_path, x, z]
	if ResourceLoader.exists(distant_mesh_name):
		var scene: PackedScene = ResourceLoader.load(distant_mesh_name, "", true)
		current_distant_mesh = scene.instance()
		current_distant_mesh.name = "distant_mesh"
		self.add_child(current_distant_mesh)


#func load_building(building_name: String):
#	var resource_name = "%s/%s.tscn" % [export_path, building_name]
#	var scene: PackedScene = ResourceLoader.load(resource_name, "", true)
#	var building_node = scene.instance()
#	$buildings.add_child(building_node)
#
#
#func unload_building(building: Building):
#	$buildings.add_child(building)
#	building.queue_free()


func get_loaded_tiles():
	var loaded_tiles = []
	for tile in $tiles.get_children():
		if tile is TerrainTile:
			loaded_tiles.push_back(tile.get_pos())
	return loaded_tiles


func load_tile(tile: Vector2):
	prints("Loading tile", tile)
	var level_name = "%s/tile_%s_%s" % [export_path, tile[0], tile[1]]
	var scene: PackedScene
	scene = ResourceLoader.load(level_name + ".tscn", "", true)
	var tile_node = scene.instance()
	tile_node.name = "tile_%s_%s" % [tile[0], tile[1]]
	$tiles.add_child(tile_node)


func unload_tile(tile_pos: Vector2):
	prints("Unloading tile", tile_pos)
	var tile_node: Node
	for tile in $tiles.get_children():
		if tile is TerrainTile:
			if tile.get_pos() == tile_pos:
				tile_node = tile
	$tiles.remove_child(tile_node)
	tile_node.queue_free()


func set_owner_recursive(node: Node, owner: Node):
	for child in node.get_children():
		child.owner = owner
		set_owner_recursive(child, owner)
