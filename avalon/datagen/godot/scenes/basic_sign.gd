extends Spatial

class_name BasicSign

export var text: String = "" setget set_text
onready var text_panel = get_node("text_panel")


func _ready():
	self.text = text


func set_text(new_text: String):
	text = new_text

	if is_inside_tree():
		text_panel.text = new_text
