extends Animal

class_name Predator

export var attack_damage := 0.5
export var rest_steps_between_attacks := 7

export var attack_rest_step := 0

var weapon: NaturalWeapon


func _ready():
	weapon = $natural_weapon


func is_attack_rested() -> bool:
	return attack_rest_step == 0


func rest_until_ready():
	if is_attack_rested():
		return

	if attack_rest_step <= rest_steps_between_attacks:
		attack_rest_step += 1
	else:
		attack_rest_step = 0


func _physics_process(_delta):
	if is_behaving_like_item or distance_to_player() > freeze_at_distance_from_player:
		return

	rest_until_ready()
	if (
		previous_behavior != active_behavior
		or not is_attack_rested()
		or not weapon.is_able_to_attack
	):
		return
	weapon.attack(attack_damage)
	attack_rest_step = 1


# chase players off the map, fine
func _safely_select_next_behavior():
	var behavior = select_next_behavior()
	var is_pursuing = behavior == active_behavior
	if is_pursuing or not is_avoiding_ocean():
		return behavior
	return avoid_ocean_behavior
