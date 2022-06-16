extends ProgressBar

onready var label: Label = get_node("hit_point_label")


func _physics_process(_delta: float):
	var player = Globals.player as Player
	value = player.hit_points
	label.text = "%s" % clamp(player.hit_points, 0, INF)
