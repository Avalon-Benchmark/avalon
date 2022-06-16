extends Spatial

class_name Teleporter

var is_player_on_portal: bool
var is_teleport_button_pressed: bool
var is_disabled := false


func _ready():
	add_to_group("teleporters")

	var area = get_node("teleporter_area/area")
	area.connect("body_entered", self, "_on_body_entered")
	area.connect("body_exited", self, "_on_body_exited")

	var button = get_node("teleporter_button/button")
	button.connect("on_button_pressed", self, "_on_teleport_button_pressed")


func _on_teleport_button_pressed(_button: Node):
	if is_player_on_portal:
		is_teleport_button_pressed = true


func _on_body_entered(body: PhysicsBody):
	if Tools.is_physical_player(body):
		is_player_on_portal = true


func _on_body_exited(body: PhysicsBody):
	if Tools.is_physical_player(body) and not is_disabled:
		is_player_on_portal = false


func can_teleport():
	return is_player_on_portal and is_teleport_button_pressed and not is_disabled


func get_world_path() -> String:
	return HARD.assert(false, "Not implemented")


func disable():
	is_disabled = true
	is_teleport_button_pressed = false
