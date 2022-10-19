extends AnimalBehavior

class_name HopRandomly

export var remaining_rotation: float = 0
export var rotation_target := NAN
export var is_hopping := false

export var hop_speed: Vector2
export var hop_count: int
export var next_hop: int = 0
export var hop_frequency: float

export var rng_key: String

export var turn_accuracy_threshold = deg2rad(7.5)

var rotation_dist: UniformDistribution
var hop_frequency_dist: ChoicesDistribution


func init(_rng_key: String, _hop_frequency: float, _hop: Vector2, _hop_count: int) -> AnimalBehavior:
	rng_key = _rng_key
	hop_frequency = _hop_frequency
	hop_speed = _hop
	hop_count = _hop_count
	_ready()
	return self


func _ready():
	hop_frequency_dist = ChoicesDistribution.new([false, true], [1 - hop_frequency, hop_frequency])
	rotation_dist = UniformDistribution.new(deg2rad(-179), deg2rad(180))


func turn_until_ready_to_hop(animal: Animal) -> bool:
	if is_nan(rotation_target) or remaining_rotation == 0:
		return true

	animal.rotate_y(clamp_rotation(remaining_rotation))

	var new_remaining_rotation = normalize_angle(rotation_target - animal.rotation.y)
	var is_stuck = remaining_rotation == new_remaining_rotation
	if is_stuck:
		remaining_rotation = 0
		rotation_target = animal.rotation.y
		return true

	remaining_rotation = new_remaining_rotation
	var is_turned_sufficiently = abs(remaining_rotation) < turn_accuracy_threshold
	return is_turned_sufficiently


func do(animal: Animal, delta: float) -> Vector3:
	var controller = animal.controller
	if animal.is_mid_hop():
		return controller.move(animal.get_ongoing_movement(), delta, Vector3.DOWN)

	var is_turned_sufficiently = turn_until_ready_to_hop(animal)
	if not is_turned_sufficiently:
		return controller.move(Vector3.ZERO, delta)

	if next_hop > 0:
		var hop = animal.forward_hop(hop_speed)
		next_hop = next_hop + 1 if next_hop < hop_count else 0
		return controller.hop(hop, delta)

	var is_starting_new_hop = hop_frequency_dist.new_value(rng_key)
	if is_starting_new_hop:
		next_hop = 1
		remaining_rotation = get_rotation_rad()
		rotation_target = normalize_angle(animal.rotation.y + remaining_rotation)
		animal.rotate_y(clamp_rotation(remaining_rotation))

	return controller.move(Vector3.ZERO, delta)


func get_rotation_rad() -> float:
	return rotation_dist.new_float(rng_key)


# https://stackoverflow.com/a/2323034/2234013
func normalize_angle(radians: float) -> float:
	var angle = int(rad2deg(radians)) % 360
	angle = (angle + 360) % 360
	if angle > 180:
		angle -= 360
	return deg2rad(angle)


func clamp_rotation(rotation: float, max_rotation: float = deg2rad(15)):
	if rotation < -max_rotation:
		rotation = -max_rotation
	elif rotation > max_rotation:
		rotation = max_rotation
	return rotation


func reset():
	rotation_target = NAN
	next_hop = 0
