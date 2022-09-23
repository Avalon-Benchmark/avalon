extends Pushable

class_name Boulder


func _ready():
	# making it slightly easier to tip over boulders when on a ledge
	linear_damp_factor = 1.0
	angular_damp_factor = 1.0
	max_damping = 0.5
