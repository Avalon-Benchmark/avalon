extends AnimalBehavior

class_name PursueAndAttackPlayer

export var knock_back: Vector2

var pursue_behavior: AnimalBehavior


func get_logic_nodes() -> Array:
	return [pursue_behavior]


func init(_pursue_behavior: AnimalBehavior, _knock_back: Vector2 = Vector2(-2, 2)) -> AnimalBehavior:
	pursue_behavior = _pursue_behavior
	knock_back = _knock_back
	return self


func _ready():
	pursue_behavior = LogicNodes.prefer_persisted(self, "pursue_behavior", pursue_behavior)


func do(animal, delta: float) -> Vector3:
	if is_knock_back_sensible(animal):
		# all animals get hoppish knockback
		return apply_knock_back(animal, delta)
	elif is_resting_after_attack(animal):
		return animal.controller.move(animal.get_ongoing_movement(), delta)

	return pursue_behavior.do(animal, delta)


func apply_knock_back(animal, delta: float) -> Vector3:
	# all animals get hoppish knockback
	return animal.controller.hop(animal.forward_hop(knock_back), delta)


func is_resting_after_attack(predator: Predator) -> bool:
	if not predator.is_attack_rested():
		pursue_behavior.reset()
		return true
	return false


func is_knock_back_sensible(predator: Predator) -> bool:
	return predator.attack_rest_step == 1


func reset():
	.reset()
	pursue_behavior.reset()
