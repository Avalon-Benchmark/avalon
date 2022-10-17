extends Area

class_name NaturalWeapon

export var is_able_to_attack := false


func _ready():
	HARD.assert(OK == connect("body_entered", self, "_on_body_entered"), "Failed to connect signal")
	HARD.assert(OK == connect("body_exited", self, "_on_body_exited"), "Failed to connect signal")


func attack(damage: float):
	Globals.get_player().take_damage(damage)


func _on_body_entered(body: Node):
	if is_player_body(body):
		is_able_to_attack = true


func _on_body_exited(body: Node):
	if is_player_body(body):
		is_able_to_attack = false


func is_player_body(body: Node) -> bool:
	return body == Globals.get_player().physical_body
