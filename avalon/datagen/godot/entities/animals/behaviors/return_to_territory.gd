extends AnimalBehavior

class_name ReturnToTerritoryBehavior

export var territory_origin: Vector3
export var territory_radius: float
export var return_target_radius: float

export var is_returning: bool = false

var movement_behavior: AnimalBehavior


func get_logic_nodes() -> Array:
	return [movement_behavior]


func init(
	_movement_behavior,
	_territory_origin: Vector3,
	_territory_radius: float,
	_return_target_radius: float
) -> AnimalBehavior:
	movement_behavior = _movement_behavior
	validate_movement_behavior()
	territory_origin = _territory_origin
	territory_radius = _territory_radius
	return_target_radius = _return_target_radius
	return self


func _ready():
	movement_behavior = LogicNodes.prefer_persisted(self, "movement_behavior", movement_behavior)
	validate_movement_behavior()


func validate_movement_behavior():
	HARD.assert(
		movement_behavior is FlyInDirection or movement_behavior is HopInDirection,
		"can only ReturnToTerritory with directed behaviors"
	)
	HARD.assert(movement_behavior.direction == Animal.TOWARDS_TERRITORY)


func get_territory_origin_at_elevation(position: Vector3) -> Vector3:
	return Vector3(territory_origin.x, position.y, territory_origin.z)


func get_return_target(animal_position: Vector3) -> Vector3:
	var target_at_elevation = get_territory_origin_at_elevation(animal_position)
	if movement_behavior is FlyInDirection:
		target_at_elevation.y += 10
	return target_at_elevation


func _is_within(position: Vector3, radius: float) -> bool:
	var territory_center = get_territory_origin_at_elevation(position)
	return position.distance_to(territory_center) < radius


func is_returning_to_territory(animal: Animal) -> bool:
	var animal_position = animal.global_transform.origin
	if is_returning:
		is_returning = not _is_within(animal_position, return_target_radius)
	else:
		is_returning = not _is_within(animal_position, territory_radius)

	if not is_returning:
		movement_behavior.reset()

	return is_returning


func is_within_territory(position: Vector3) -> bool:
	return _is_within(position, territory_radius)


func do(animal: Animal, delta: float) -> Vector3:
	movement_behavior.fixed_point = get_return_target(animal.global_transform.origin)
	return movement_behavior.do(animal, delta)


func reset():
	.reset()
	movement_behavior.reset()


func describe():
	return movement_behavior.describe()
