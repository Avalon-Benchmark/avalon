extends Animal
class_name Frog

export var inactive_speed := Vector2(3, 2.5)
export var inactive_movement_frequency: float = 1.0 / 16
export var inactive_movement_steps := 1


func _ready():
	inactive_behavior = HopRandomly.new(
		_rng_key("inactive"), inactive_movement_frequency, inactive_speed, inactive_movement_steps
	)

	avoid_ocean_behavior = AvoidOcean.new(_rng_key("avoid_ocean"), 8, inactive_speed / 2)


func select_next_behavior() -> AnimalBehavior:
	return inactive_behavior
