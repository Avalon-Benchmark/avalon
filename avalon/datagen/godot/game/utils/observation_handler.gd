extends Reference

class_name ObservationHandler

var root: Node
var camera_controller

var last_food_frame := INF
var is_food_gone := false
var is_done_episode := false

var feature_shape_type := {}

const SELECTED_FEATURES_FOR_HUMAN_RECORDING := {
	"episode_id": true,
	"frame_id": true,
	"reward": true,
	"is_done": true,
	"is_dead": true,
	"target_head_position": true,
	"target_left_hand_position": true,
	"target_right_hand_position": true,
	"target_head_rotation": true,
	"target_left_hand_rotation": true,
	"target_right_hand_rotation": true,
	"physical_body_position": true,
	"physical_head_position": true,
	"physical_left_hand_position": true,
	"physical_right_hand_position": true,
	"physical_body_rotation": true,
	"physical_head_rotation": true,
	"physical_left_hand_rotation": true,
	"physical_right_hand_rotation": true,
	"physical_body_delta_position": true,
	"physical_head_delta_position": true,
	"physical_left_hand_delta_position": true,
	"physical_right_hand_delta_position": true,
	"physical_body_delta_rotation": true,
	"physical_head_delta_rotation": true,
	"physical_left_hand_delta_rotation": true,
	"physical_right_hand_delta_rotation": true,
	"physical_head_relative_position": true,
	"physical_left_hand_relative_position": true,
	"physical_right_hand_relative_position": true,
	"physical_head_relative_rotation": true,
	"physical_left_hand_relative_rotation": true,
	"physical_right_hand_relative_rotation": true,
	"left_hand_thing_colliding_with_hand": true,
	"left_hand_held_thing": true,
	"right_hand_thing_colliding_with_hand": true,
	"right_hand_held_thing": true,
	"nearest_food_position": true,
	"nearest_food_id": true,
	"is_food_present_in_world": true,
	"physical_body_kinetic_energy_expenditure": true,
	"physical_body_potential_energy_expenditure": true,
	"physical_head_potential_energy_expenditure": true,
	"physical_left_hand_kinetic_energy_expenditure": true,
	"physical_left_hand_potential_energy_expenditure": true,
	"physical_right_hand_kinetic_energy_expenditure": true,
	"physical_right_hand_potential_energy_expenditure": true,
	"fall_damage": true,
	"hit_points_lost_from_enemies": true,
	"hit_points_gained_from_eating": true,
	"hit_points": true,
}

var PERF_SIMPLE_AGENT: bool = ProjectSettings.get_setting("avalon/simple_agent")
var PERF_FRAME_READ: bool = ProjectSettings.get_setting("avalon/frame_read")
var PERF_FRAME_WRITE: bool = ProjectSettings.get_setting("avalon/frame_write")


func _init(_root: Node, _camera_controller):
	root = _root
	camera_controller = _camera_controller


func reset() -> void:
	last_food_frame = INF
	is_food_gone = false
	is_done_episode = false


func get_current_observation(player: Player, frame: int) -> Dictionary:
	var observation = player.get_observation_and_reward()
	if PERF_SIMPLE_AGENT:
		return observation

	if is_food_present_in_world():
		last_food_frame = frame

	var nearest_food = get_nearest_food(player)
	if is_instance_valid(nearest_food):
		observation["nearest_food_position"] = nearest_food.global_transform.origin
		observation["nearest_food_id"] = RID(nearest_food).get_id()
	else:
		observation["nearest_food_position"] = Vector3(INF, INF, INF)
		observation["nearest_food_id"] = -1

	observation["is_food_present_in_world"] = int(is_food_present_in_world())

	is_done_episode = (
		(frame - last_food_frame) >= player.num_frames_alive_after_food_is_gone
		or observation["is_dead"]
	)
	observation["is_done"] = int(is_done_episode)

	return observation


func get_interactive_observation(
	player: Player,
	episode: int,
	frame: int,
	selected_features: Dictionary = {},
	is_limiting_to_selected_features: bool = true,
	is_recording_images: bool = true
) -> Dictionary:
	var current_observation = get_current_observation(player, frame)
	if PERF_SIMPLE_AGENT:
		var rgbd_data: PoolByteArray
		if PERF_FRAME_READ:
			rgbd_data = camera_controller.get_rgbd_data().data["data"]
		if PERF_FRAME_WRITE:
			current_observation[CONST.RGBD_FEATURE] = rgbd_data
		else:
			current_observation[CONST.RGBD_FEATURE] = null
		return current_observation
	return get_interactive_observation_from_current_observation(
		current_observation,
		episode,
		frame,
		selected_features,
		is_limiting_to_selected_features,
		is_recording_images
	)


func get_interactive_observation_from_current_observation(
	current_observation: Dictionary,
	episode: int,
	frame: int,
	selected_features: Dictionary = {},
	is_limiting_to_selected_features: bool = true,
	is_recording_images: bool = true
) -> Dictionary:
	var interactive_observation = _convert_to_interactive_observation(
		current_observation, episode, frame, is_recording_images
	)
	if is_limiting_to_selected_features and len(selected_features) > 0:
		return _limit_to_selected_features(interactive_observation, selected_features)
	else:
		return interactive_observation


func get_available_features(player: Player) -> Dictionary:
	# TODO ideally we could return `.keys` but env bridge does something fancy with this dictionary
	var interactive_observation = get_interactive_observation(player, 0, 0, {}, false, false)

	if camera_controller.are_debug_views_enabled:
		interactive_observation[CONST.TOP_DOWN_RGBD_FEATURE] = [
			null,
			CONST.FAKE_TYPE_IMAGE,
			[camera_controller.resolution.y, camera_controller.resolution.x, 4]
		]
		interactive_observation[CONST.ISOMETRIC_RGBD_FEATURE] = [
			null,
			CONST.FAKE_TYPE_IMAGE,
			[camera_controller.resolution.y, camera_controller.resolution.x, 4]
		]

	return interactive_observation


func _convert_to_interactive_observation(
	observation: Dictionary, episode: int, frame: int, is_recording_images: bool = true
) -> Dictionary:
	var resolution_x = camera_controller.resolution.x
	var resolution_y = camera_controller.resolution.y

	if feature_shape_type.empty():
		for feature in observation:
			var value = observation[feature]
			var data_type = typeof(value)
			var shape = _get_shape(value, data_type)
			feature_shape_type[feature] = [data_type, shape]
		feature_shape_type[CONST.RGBD_FEATURE] = [
			CONST.FAKE_TYPE_IMAGE, [resolution_y, resolution_x, 4]
		]
		if camera_controller.are_debug_views_enabled:
			feature_shape_type[CONST.TOP_DOWN_RGBD_FEATURE] = [
				CONST.FAKE_TYPE_IMAGE, [resolution_y, resolution_x, 4]
			]
			feature_shape_type[CONST.ISOMETRIC_RGBD_FEATURE] = [
				CONST.FAKE_TYPE_IMAGE, [resolution_y, resolution_x, 4]
			]

	var observed_data := {}
	for feature in observation:
		var value = observation[feature]
		var feature_shape = feature_shape_type[feature]
		var data_type: int = feature_shape[0]
		var shape = feature_shape[1]
		observed_data[feature] = [value, data_type, shape]

	observed_data[CONST.EPISODE_ID_FEATURE] = [episode, typeof(episode), [1]]
	observed_data[CONST.FRAME_ID_FEATURE] = [frame, typeof(frame), [1]]

	observed_data[CONST.RGBD_FEATURE] = [
		camera_controller.get_rgbd_data() if is_recording_images else null,
		CONST.FAKE_TYPE_IMAGE,
		[resolution_y, resolution_x, 4]
	]

	if camera_controller.are_debug_views_enabled and is_recording_images:
		observed_data[CONST.TOP_DOWN_RGBD_FEATURE] = [
			camera_controller.get_top_down_rgbd_data(),
			CONST.FAKE_TYPE_IMAGE,
			[resolution_y, resolution_x, 4]
		]
		observed_data[CONST.ISOMETRIC_RGBD_FEATURE] = [
			camera_controller.get_isometric_rgbd_data(),
			CONST.FAKE_TYPE_IMAGE,
			[resolution_y, resolution_x, 4]
		]

	return observed_data


func _limit_to_selected_features(observation: Dictionary, selected_features: Dictionary) -> Dictionary:
	if len(selected_features) == 0:
		return observation
	var result := {}
	for feature_name in selected_features:
		HARD.assert(
			observation.has(feature_name),
			"Feature name %s not present in on observation" % feature_name
		)
		result[feature_name] = observation[feature_name]
	return result


func is_food_present_in_world():
	return len(get_all_food_in_world()) > 0


func get_nearest_food(player: Player) -> Node:
	var current_position = player.get_current_position()
	var min_distance = INF
	var nearest_food = null
	for food in get_all_food_in_world():
		var food_distance = (food.global_transform.origin - current_position).length()
		if food_distance < min_distance:
			nearest_food = food
			min_distance = food_distance
	return nearest_food


func get_all_food_in_world() -> Array:
	var foods = []
	for food in root.get_tree().get_nodes_in_group("food"):
		if food.is_impossible_to_eat():
			continue
		foods.append(food)
	return foods


func _get_shape(value, data_type):
	match data_type:
		TYPE_VECTOR2:
			return [2]
		TYPE_VECTOR3:
			return [3]
		TYPE_REAL:
			return [1]
		TYPE_INT:
			return [1]
		TYPE_ARRAY:
			var shape = _get_shape(value[0], typeof(value[0]))
			shape.append(value.size())
			return shape
		_:
			HARD.stop("Unknown data type: %s", data_type)
