extends Area

class_name NaturalWeapon

var _player  #: Player TODO(mjr) type hint causing a cyclical dependency/parser issue. Maybe just dirty state


func _ready():
	HARD.assert(OK == connect("body_entered", self, "_on_body_entered"), "Failed to connect signal")
	HARD.assert(OK == connect("body_exited", self, "_on_body_exited"), "Failed to connect signal")


func is_able_to_attack() -> bool:
	return _player != null


func attack(damage: float):
	_player.take_damage(damage)


func _on_body_entered(body: Node):
	# TODO make more specific
	# TODO may be some bug with large detection radiuses due to load/unload
	if body is KinematicBody and body.name == "physical_body":
		_player = body.get_parent().get_parent()


func _on_body_exited(body: Node):
	if body is KinematicBody and body.name == "physical_body":
		_player = null
