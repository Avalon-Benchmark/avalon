extends AnimalBehavior

class_name ClimbWhenNecessary

const DOWN = Vector2(0, -1)
const UP = Vector2(0, 1)

export var climb_speed: float
export var escape_hop_speed: Vector2
export var is_horizontal_movement_enabled := false

export var is_making_escape_hop := false

var grounded_headway_criteria: AbleToMakeHeadway
var hop_behavior: AnimalBehavior


func get_logic_nodes() -> Array:
	var nodes = [hop_behavior]
	if grounded_headway_criteria != null:
		nodes.append(grounded_headway_criteria)
	return nodes


func init(
	_climb_speed: float,
	_escape_hop_speed: Vector2,
	_hop_behavior: AnimalBehavior,
	_grounded_headway_criteria = null
) -> AnimalBehavior:
	HARD.assert(
		"hop_speed" in _hop_behavior, "ClimbWhenNecessary.hop_behavior must implement .hop_speed"
	)
	climb_speed = _climb_speed
	escape_hop_speed = _escape_hop_speed
	hop_behavior = _hop_behavior
	if _grounded_headway_criteria != null:
		grounded_headway_criteria = _grounded_headway_criteria
	return self


func _ready():
	hop_behavior = LogicNodes.prefer_persisted(self, "hop_behavior", hop_behavior)

	var is_null_ok = true
	grounded_headway_criteria = LogicNodes.prefer_persisted(
		self, "grounded_headway_criteria", grounded_headway_criteria, is_null_ok
	)


func get_climb_direction(animal: Animal) -> Vector2:
	if not hop_behavior is HopInDirection:
		return DOWN

	var directed = hop_behavior as HopInDirection
	var target = directed.target_position(animal)

	var is_facing_target = animal.is_point_in_front_of(target)

	# climbing down only happens when "giving up," so if we've started climbing,
	# it is because we're trying to get over an obstacle between us and target
	# (player or territory)
	var dir: Vector2
	match directed.direction:
		animal.AWAY_FROM_PLAYER:
			dir = DOWN if is_facing_target else UP
		Animal.TOWARDS_PLAYER, Animal.TOWARDS_TERRITORY:
			dir = UP if is_facing_target else DOWN
		_:
			HARD.assert(false, "%s.direction is invalid" % directed)

	if is_horizontal_movement_enabled:
		dir.x = get_horizontal_movement(animal, target)

	return dir


func get_horizontal_movement(animal: Animal, target: Vector3):
	if not is_horizontal_movement_enabled or not hop_behavior is HopInDirection:
		return 0
	var is_target_to_right = animal.to_local(target).x > 0
	match hop_behavior.direction:
		animal.AWAY_FROM_PLAYER:
			return -1 if is_target_to_right else 1
		Animal.TOWARDS_PLAYER, Animal.TOWARDS_TERRITORY:
			return 1 if is_target_to_right else -1
		_:
			HARD.assert(false, "%s.direction is invalid" % hop_behavior)


func make_escape_hop_until_landed(animal: Animal, delta: float):
	if is_making_escape_hop and not animal.controller.is_on_floor():
		# persist escape hop velocity to avoid early collisions resulting in dud hops
		var continue_forward_velocity = animal.forward_hop(
			Vector2(escape_hop_speed.x, animal.get_ongoing_movement().y)
		)
		return animal.controller.move(continue_forward_velocity, delta)

	if not is_making_escape_hop and get_climb_direction(animal) == UP:
		if HARD.mode():
			print("%s is hopping out of a climb" % animal)
		is_making_escape_hop = true
		reset_criteria()
		var escape_hop = animal.forward_hop(escape_hop_speed)
		return animal.controller.move(escape_hop, delta)


func do(animal: Animal, delta: float) -> Vector3:
	var is_able_to_climb = animal.controller.is_able_to_climb()
	var is_climbing = animal.is_climbing()

	if (not is_able_to_climb) and is_climbing:
		var observed_velocity = make_escape_hop_until_landed(animal, delta)
		if observed_velocity:
			return observed_velocity

	is_making_escape_hop = false

	if (not is_able_to_climb) or is_making_headway_without_climbing(animal):
		animal.stop_climbing()
		return hop_behavior.do(animal, delta)

	reset_criteria()

	var direction = get_climb_direction(animal)

	if direction.y == DOWN.y:
		if not is_climbing:
			return hop_behavior.do(animal, delta)
		if is_close_to_bottom_of_climb(animal):
			if HARD.mode():
				print("%s dropping from climb" % animal)
			animal.stop_climbing()
			return hop_behavior.do(animal, delta)

	if not is_climbing and HARD.mode():
		print("%s started climbing" % animal)

	return animal.controller.climb(direction * climb_speed, delta)


func is_making_headway_without_climbing(animal: Animal) -> bool:
	if animal.is_climbing():
		return false
	return grounded_headway_criteria != null and grounded_headway_criteria.is_matched_by(animal)


func is_close_to_bottom_of_climb(animal: Animal) -> bool:
	return animal.is_climbing() and animal.controller.get_floor_ray().is_colliding()


func reset_criteria():
	if grounded_headway_criteria != null:
		grounded_headway_criteria.reset()


func reset():
	reset_criteria()
	hop_behavior.reset()


func describe():
	var hop_name = hop_behavior.describe()
	return "climb or %s" % hop_name
