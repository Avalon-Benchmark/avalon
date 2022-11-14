extends ControlledNodeSpec

class_name PlayerSpec

# m/s
var max_head_linear_speed: float
# deg/s
var max_head_angular_speed: Vector3
# m/s
var max_hand_linear_speed: float
# deg/s
var max_hand_angular_speed: float
# m
var jump_height: float
var arm_length: float
var starting_hit_points: float
# kg
var mass: float
var arm_mass_ratio: float
var head_mass_ratio: float
# m/s
var standup_speed_after_climbing: float
# m
var min_head_position_off_of_floor: float
# N
var push_force_magnitude: float
# N
var throw_force_magnitude: float
var starting_left_hand_position_relative_to_head: Vector3
var starting_right_hand_position_relative_to_head: Vector3
# m/s
var minimum_fall_speed: float
var fall_damage_coefficient: float
var num_frames_alive_after_food_is_gone: int
var eat_area_radius: float
var is_displaying_debug_meshes: bool
var is_human_playback_enabled: bool
var is_slowed_from_crouching: bool


func get_scene_instance() -> Object:
	return HARD.assert(false, "Must be overridden")


func get_node() -> ControlledNode:
	var player = get_scene_instance()
	player.spec = self
	player.spawn_point_name = spawn_point_name
	# TODO: sigh ... this is very error prone how these get set
	player.max_head_linear_speed = max_head_linear_speed
	player.max_head_angular_speed = max_head_angular_speed
	player.max_hand_linear_speed = max_hand_linear_speed
	player.max_hand_angular_speed = max_hand_angular_speed
	player.jump_height = jump_height
	player.arm_length = arm_length
	player.starting_hit_points = starting_hit_points
	player.mass = mass
	player.arm_mass_ratio = arm_mass_ratio
	player.head_mass_ratio = head_mass_ratio
	player.standup_speed_after_climbing = standup_speed_after_climbing
	player.min_head_position_off_of_floor = min_head_position_off_of_floor
	player.push_force_magnitude = push_force_magnitude
	player.throw_force_magnitude = throw_force_magnitude
	player.minimum_fall_speed = minimum_fall_speed
	player.fall_damage_coefficient = fall_damage_coefficient
	player.num_frames_alive_after_food_is_gone = num_frames_alive_after_food_is_gone
	player.eat_area_radius = eat_area_radius
	player.is_displaying_debug_meshes = is_displaying_debug_meshes
	player.is_human_playback_enabled = is_human_playback_enabled
	player.is_slowed_from_crouching = is_slowed_from_crouching

	HARD.assert(player.max_head_linear_speed > 0, "`max_head_linear_speed` must be greater than 0")
	HARD.assert(
		player.max_head_angular_speed.x > 0, "`max_head_angular_speed.x` must be greater than 0"
	)
	HARD.assert(
		player.max_head_angular_speed.y > 0, "`max_head_angular_speed.y` must be greater than 0"
	)
	HARD.assert(
		player.max_head_angular_speed.z > 0, "`max_head_angular_speed.z` must be greater than 0"
	)
	HARD.assert(player.max_hand_linear_speed > 0, "`max_hand_linear_speed` must be greater than 0")
	HARD.assert(
		player.max_hand_angular_speed > 0, "`max_hand_angular_speed` must be greater than 0"
	)
	HARD.assert(player.jump_height > 0, "`jump_height` must be greater than 0")

	return player
