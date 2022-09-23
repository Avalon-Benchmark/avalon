extends StaticBody

class_name LabelGUI

export var text: String = "" setget set_text

onready var label: Label = find_node("Label", true, false)
onready var viewport: Viewport = find_node("Viewport", true, false)
onready var quad: MeshInstance = find_node("Quad", true, false)

export var height_in_real_world_units := 0.5
export var width_in_real_world_units := 1.0
export var pixel_per_real_world_unit := 300
export var font_size := 48


func _ready():
	var size_in_real_world_units = Vector2(width_in_real_world_units, height_in_real_world_units)
	quad.mesh.size = size_in_real_world_units
	viewport.size = size_in_real_world_units * pixel_per_real_world_unit
	var font = label.get_font("default_font")
	if font is DynamicFont:
		font.size = font_size

	self.text = text


func set_text(new_text: String):
	text = new_text
	if is_inside_tree():
		label.text = new_text.to_upper()
		label.update()
