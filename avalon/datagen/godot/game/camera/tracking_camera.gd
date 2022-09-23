extends Camera

class_name TrackingCamera

export var is_enabled := true
export var is_tracking_relative_to_initial_transform: bool
export var is_automatically_looking_at_tracked: bool

var initial_transform: Transform
var tracked: Spatial


func _ready():
	initial_transform = global_transform
