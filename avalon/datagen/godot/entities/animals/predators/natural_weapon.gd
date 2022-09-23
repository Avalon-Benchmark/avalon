extends Area

class_name NaturalWeapon

var is_able_to_attack := false
var _player: Player


func _ready():
	HARD.assert(OK == connect("body_entered", self, "_on_body_entered"), "Failed to connect signal")
	HARD.assert(OK == connect("body_exited", self, "_on_body_exited"), "Failed to connect signal")


func attack(damage: float):
	_player.take_damage(damage)


func _on_body_entered(body: Node):
	if is_player_body(body):
		is_able_to_attack = true


func _on_body_exited(body: Node):
	if is_player_body(body):
		is_able_to_attack = false


func is_player_body(body: Node) -> bool:
	if _player == null:
		_player = get_tree().root.find_node("player", true, false)
	return body == _player.physical_body
