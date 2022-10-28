extends ProgressBar

class_name HealthBar

onready var label: Label = get_node("hit_point_label")


func _physics_process(_delta: float):
	var player = Globals.get_player()
	if player:
		value = player.hit_points
		label.text = "%.2f" % clamp(player.hit_points, 0, INF)
