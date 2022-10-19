extends KinematicBody

var _player
const MAX_SPEED = 2.5


func get_player_position() -> Vector3:
	if _player == null:
		_player = get_tree().root.find_node("player", true, false)
	return _player.physical_body.global_transform.origin


func _physics_process(_delta):
	var away_from_player = -global_transform.origin.direction_to(get_player_position()) * MAX_SPEED
	away_from_player.y = 0
	away_from_player.z += randf() * 0.5
	away_from_player.x += randf() * 0.5
	var _velocity = .move_and_slide(away_from_player)
