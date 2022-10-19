extends AnimalBehavior

class_name FlyRandomly

const TURN_ACCURACY_THRESHOLD = deg2rad(7.5)

export var rng_key: String
export var speed: float
export var rotation_target := NAN
export var turn_frequency: float

var rotation_dist: UniformDistribution = UniformDistribution.new(deg2rad(-179), deg2rad(180))
var turn_frequency_dist: ChoicesDistribution


func init(_rng_key: String, _turn_frequency: float, _speed: float) -> AnimalBehavior:
	rng_key = _rng_key
	turn_frequency = _turn_frequency
	speed = _speed
	_ready()
	return self


func _ready():
	rotation_dist = UniformDistribution.new(deg2rad(-179), deg2rad(180))
	turn_frequency_dist = ChoicesDistribution.new(
		[false, true], [1 - turn_frequency, turn_frequency]
	)


func do(animal: Animal, delta: float) -> Vector3:
	var controller = animal.controller

	var is_ok_to_turn_again = true
	if not is_nan(rotation_target):
		var remaining_rotation = rotation_target - animal.rotation.y
		is_ok_to_turn_again = abs(remaining_rotation) < TURN_ACCURACY_THRESHOLD
		animal.rotate_y(clamp_rotation(remaining_rotation))
		if remaining_rotation == 0:
			rotation_target = NAN

	var should_turn = is_ok_to_turn_again and turn_frequency_dist.new_value(rng_key)
	if should_turn:
		var turn = get_rotation_rad()
		rotation_target = normalize_angle(animal.rotation.y + turn)

	animal.transform = seek_desired_altitude(animal, delta)

	return controller.move(animal.get_local_forward() * speed, delta)


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


static func seek_desired_altitude(animal: Animal, delta: float) -> Transform:
	var altitude_diff = animal.controller.distance_from_desired_altitude()
	var desired_pitch: float
	if abs(altitude_diff) <= 2:
		desired_pitch = 0
	elif altitude_diff > 0:
		desired_pitch = deg2rad(-30)
	elif altitude_diff < 0:
		desired_pitch = deg2rad(30)

	return animal.transform.interpolate_with(
		correct_course(animal.transform, animal.rotation, desired_pitch), delta
	)


static func correct_course(transform: Transform, current_rotation: Vector3, desired_pitch: float) -> Transform:
	var scale = transform.basis.get_scale()
	transform.basis = Basis(Vector3(desired_pitch, current_rotation.y, 0)).scaled(scale)
	return transform


func reset():
	rotation_target = NAN
