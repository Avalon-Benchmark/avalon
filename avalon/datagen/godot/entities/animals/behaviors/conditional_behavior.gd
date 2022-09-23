# Only do the wrapped behavior if all criteria are met. Otherwise, stay still.
# Used by deer, hippos, and snakes to avoid the player when they get within their detection radius.
extends AnimalBehavior

class_name ConditionalBehavior

var criteria: Array

var if_behavior: AnimalBehavior

var is_matched = null


func _init(_criteria: Array, _if_behavior):
	criteria = _criteria
	if_behavior = _if_behavior


func do(animal: Animal, delta: float) -> Vector3:
	var is_previously_matched = is_matched
	is_matched = BehaviorCriteria.all_match(animal, criteria)
	var is_changed = is_previously_matched != is_matched

	if is_changed and HARD.mode():
		var b_name = if_behavior.get_name() if is_matched else "stay_still"
		print("behavior change in %s.conditional: -> %s" % [animal.name, b_name])

	if is_matched:
		return if_behavior.do(animal, delta)

	return animal.controller.move(animal.get_ongoing_movement(), delta)


func select_behavior(animal: Animal):
	is_matched = BehaviorCriteria.all_match(animal, criteria)
	if is_matched:
		return if_behavior


func reset():
	.reset()
	is_matched = null
	if_behavior.reset()


func get_name():
	var c_names = []
	for c in criteria:
		c_names.append(c.get_name())

	return "when(%s, %s)" % [c_names, if_behavior.get_name()]
