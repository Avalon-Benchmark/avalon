extends SpecBase

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
var total_energy_coefficient: float
var body_kinetic_energy_coefficient: float
var body_potential_energy_coefficient: float
var head_potential_energy_coefficient: float
var left_hand_kinetic_energy_coefficient: float
var left_hand_potential_energy_coefficient: float
var right_hand_kinetic_energy_coefficient: float
var right_hand_potential_energy_coefficient: float
var num_frames_alive_after_food_is_gone: int
var eat_area_radius: float
var is_displaying_debug_meshes: bool


func create_player() -> Object:
	return HARD.assert(false, "Must be overridden")


func build(_is_registered: bool = true) -> Object:
	var node = create_player()
	node.spec = self
	return node


static func create_action() -> Action:
	return HARD.assert(false, "Not implemented")


static func process_pipe_into_action(_action_pipe: StreamPeerBuffer) -> Action:
	return HARD.assert(false, "Not implemented")
