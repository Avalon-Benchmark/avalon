extends Item

class_name Stackable

export var rest_delay := 1.0
export var rest_linear_speed := 0.0
export var rest_angular_speed := 0.0
export var is_axis_aligned_when_held := false

var _timer = null
var _is_held := false


func _ready():
	_timer = Timer.new()
	add_child(_timer)
	_timer.connect("timeout", self, "_on_timer_timeout")
	_timer.start(rest_delay)


func is_grabbable() -> bool:
	return true


func grab(physical_hand: RigidBody) -> Node:
	mode = RigidBody.MODE_RIGID
	_is_held = true
	_timer.stop()

	if is_axis_aligned_when_held:
		global_transform.basis = Basis()
		axis_lock_angular_x = true
		axis_lock_angular_y = true
		axis_lock_angular_z = true

	return .grab(physical_hand)


func release() -> void:
	_is_held = false
	_timer.start(rest_delay)

	if is_axis_aligned_when_held:
		global_transform.basis = Basis()
		axis_lock_angular_x = false
		axis_lock_angular_y = false
		axis_lock_angular_z = false

	.release()


func _on_timer_timeout():
	if (
		(
			is_equal_approx(linear_velocity.length(), rest_linear_speed)
			or linear_velocity.length() < rest_linear_speed
		)
		and (
			is_equal_approx(angular_velocity.length(), rest_angular_speed)
			or angular_velocity.length() < rest_angular_speed
		)
		and mode == RigidBody.MODE_RIGID
	):
		print("freezing %s" % name)
		mode = RigidBody.MODE_STATIC
		_timer.stop()
	else:
		_timer.start(rest_delay)
