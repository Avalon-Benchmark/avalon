extends Animal
class_name Frog

export var inactive_speed := Vector2(3, 2.5)
export var inactive_movement_frequency: float = 1.0 / 16
export var inactive_movement_hops := 1


func _ready():
	set_inactive(
		HopRandomly.new().init(
			_rng_key("inactive"),
			inactive_movement_frequency,
			inactive_speed,
			inactive_movement_hops
		)
	)

	set_avoid_ocean(AvoidOcean.new().init(_rng_key("avoid_ocean"), 8, inactive_speed / 2))


func select_next_behavior() -> AnimalBehavior:
	return inactive_behavior
