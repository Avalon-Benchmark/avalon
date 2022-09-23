extends Teleporter

class_name TaskTeleporter

onready var title: LabelGUI = get_node("teleporter_gui/title/text_panel")
onready var runtime: LabelGUI = get_node("teleporter_gui/runtime/text_panel")
onready var last_world: LabelGUI = get_node("teleporter_gui/last_world/text_panel")


func set_message(next_world_text: String, time_to_complete: float, last_world_text: String):
	title.text = next_world_text
	if time_to_complete:
		runtime.text = "Time limit: %s mins" % time_to_complete
	else:
		runtime.text = ""

	if last_world_text:
		last_world.text = last_world_text
