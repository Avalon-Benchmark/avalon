extends Resource

class_name VisualEffectHandler

var environment: Environment
var max_brightness := 1.0
var max_saturation := 1.0
var min_brightness := 0.2
var min_saturation := 0.2
var last_hit_points := 1.0


func _init():
	pass


func bind_world(world_node: WorldEnvironment):
	environment = world_node.environment
	max_brightness = environment.adjustment_brightness
	max_saturation = environment.adjustment_saturation
	last_hit_points = 1.0


func react(observation: Dictionary):
	if not environment:
		return
	var hit_points = observation["hit_points"]
	var is_success = (hit_points > 0) and not observation["is_food_present_in_world"]
	var was_bitten = observation["hit_points_lost_from_enemies"] > 0
	var new_brightness = environment.adjustment_brightness
	var new_saturation = environment.adjustment_saturation
	if is_success:
		new_brightness = 2.0
		new_saturation = 2.0
	elif was_bitten:
		new_brightness = 0.25
	elif last_hit_points != hit_points:
		last_hit_points = hit_points
		hit_points = clamp(hit_points, 0.0, 1.0)
		new_brightness = lerp(min_brightness, max_brightness, hit_points)
		new_saturation = lerp(min_saturation, max_saturation, hit_points)
	else:
		return
	environment.adjustment_brightness = new_brightness
	environment.adjustment_saturation = new_saturation
