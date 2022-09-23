extends Spatial

class_name TerrainTile

export var tile_x: int
export var tile_z: int
export var building_names: Array


func get_pos() -> Vector2:
	return Vector2(tile_x, tile_z)


func get_class():
	# godot `get_class` frustratingly enough only returns base class names
	return "TerrainTile"
