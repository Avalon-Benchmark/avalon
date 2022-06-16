extends Item

class_name Pushable

onready var player = get_tree().root.find_node("player", true, false)
export var is_climbing_allowed := false
export var rest_delay := 1.0
export var rest_linear_speed := 0.0
export var rest_angular_speed := 0.0
export var is_axis_aligned_when_held := false

var _timer = null


func _ready():
	_timer = Timer.new()
	add_child(_timer)
	_timer.connect("timeout", self, "_on_timer_timeout")
	_timer.start(rest_delay)


func is_pushable() -> bool:
	return true


func is_climbable() -> bool:
	return is_climbing_allowed


func push(push_impulse: Vector3, push_offset: Vector3) -> void:
	mode = RigidBody.MODE_RIGID
	if _timer.is_stopped():
		_timer.start(rest_delay)
	else:
		_timer.wait_time = rest_delay
		_timer.paused = true
	# TODO better ways to push up slopes
	push_impulse.y += push_impulse.length() * 0.1
	.push(push_impulse, push_offset)


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


func _physics_process(_delta):
	if mode == RigidBody.MODE_RIGID:
		_timer.paused = false
	var collision = player.get_floor_collision()
	var is_item_on_floor = ground_contact_count > 0
	if collision and collision.collider == self and is_item_on_floor:
		if HARD.mode():
			print("player collided with %s, making static" % self)
		mode = RigidBody.MODE_STATIC
