extends AnimalBehavior

class_name ConditionalBehavior

var criteria: Array

var if_behavior: AnimalBehavior
var else_behavior: AnimalBehavior

var is_matched = null


func _init(_criteria: Array, _if_behavior, _else_behavior = null):
	criteria = _criteria
	if_behavior = _if_behavior
	else_behavior = _else_behavior


func do(animal: Animal, delta: float) -> Vector3:
	var is_previously_matched = is_matched

	var behavior = select_behavior(animal)
	var is_changed = is_previously_matched != is_matched

	if is_changed and HARD.mode():
		var b_name = behavior.get_name() if behavior else "stay_still"
		print("behavior change in %s.conditional: -> %s" % [animal.name, b_name])

	if behavior:
		return behavior.do(animal, delta)

	return animal.controller.move(animal.get_ongoing_movement(), delta)


func select_behavior(animal: Animal):
	is_matched = BehaviorCriteria.all_match(animal, criteria)
	if is_matched:
		if else_behavior:
			else_behavior.reset()
		return if_behavior
	if else_behavior:
		if_behavior.reset()
		return else_behavior


func reset():
	.reset()
	is_matched = null
	if_behavior.reset()
	if else_behavior:
		else_behavior.reset()


func get_name():
	var c_names = []
	for c in criteria:
		c_names.append(c.get_name())

	var name = "when(%s, %s" % [c_names, if_behavior.get_name()]
	if else_behavior:
		name += ", else=%s" % else_behavior.get_name()

	return name + ")"
