extends Camera

class_name TrackingCamera

export var is_enabled := true
export var is_tracking_relative_to_initial_transform: bool

var initial_transform: Transform


func _ready():
	initial_transform = global_transform
