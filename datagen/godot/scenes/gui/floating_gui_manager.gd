extends Spatial

class_name FloatingGUIManager

var _node_by_name := {}


func _ready():
	for child in get_children():
		_node_by_name[child.name] = child


func get_by_name(name: String) -> FloatingGUI:
	return _node_by_name[name]


func has(name: String):
	return name in _node_by_name


func get_all() -> Array:
	return _node_by_name.values()


func set_positions() -> void:
	for node in get_all():
		node.set_position()


func open_by_name(name: String) -> void:
	if has(name):
		get_by_name(name).open()


func close_by_name(name: String) -> void:
	if has(name):
		get_by_name(name).close()
