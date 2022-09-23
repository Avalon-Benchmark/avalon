extends Spatial

class_name TextSelect

onready var text_panel = get_node("text_panel")

export var option_prefix := ""
export var options := [] setget set_options
export var curr_option_index := 0 setget _update_text_on_index_change

const PREV_NODE_NAME = "prev"
const NEXT_NODE_NAME = "next"


func _ready():
	self.options = options
	_update_text_on_index_change(curr_option_index)


func set_options(new_options):
	options = new_options
	_update_text_on_index_change(curr_option_index)


func get_current_option() -> String:
	return options[curr_option_index]


func _get_option_string() -> String:
	if len(options) == 0:
		return ""
	var option = get_current_option()
	if not option_prefix:
		return option
	else:
		return "%s: %s" % [option_prefix, option]


func _update_text_on_index_change(index: int):
	var next_index = curr_option_index + index
	if next_index < 0:
		curr_option_index = len(options) - 1
	elif next_index >= len(options):
		curr_option_index = 0
	else:
		curr_option_index = next_index
	text_panel.text = _get_option_string()


func set_next_text(node: Node) -> void:
	match node.name:
		PREV_NODE_NAME:
			_update_text_on_index_change(-1)
		NEXT_NODE_NAME:
			_update_text_on_index_change(1)
		_:
			HARD.assert(false, "Got unexpected node name %s" % node.name)
