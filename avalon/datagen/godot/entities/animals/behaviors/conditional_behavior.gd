# Only do the wrapped behavior if all criteria are met. Otherwise, stay still.
# Used by deer, hippos, and snakes to avoid the player when they get within their detection radius.
extends AnimalBehavior

class_name ConditionalBehavior

export var is_reset := true
export var is_matched := false

var criteria: Array

var if_behavior: AnimalBehavior


func init(_criteria: Array, _if_behavior) -> AnimalBehavior:
	criteria = _criteria
	if_behavior = _if_behavior
	return self


func _ready():
	criteria = LogicNodes.prefer_persisted_array(self, "criteria", criteria)
	if_behavior = LogicNodes.prefer_persisted(self, "if_behavior", if_behavior)


func get_logic_nodes() -> Array:
	var nodes = [if_behavior]
	nodes.append_array(criteria)
	return nodes


func do(animal: Animal, delta: float) -> Vector3:
	var is_previously_matched = is_matched
	is_matched = BehaviorCriteria.all_match(animal, criteria)
	var is_changed = is_reset or is_previously_matched != is_matched
	is_reset = false

	if is_changed and HARD.mode():
		var b_description = if_behavior.describe() if is_matched else "stay_still"
		print("behavior change in %s.conditional: -> %s" % [animal.name, b_description])

	if is_matched:
		return if_behavior.do(animal, delta)

	return animal.controller.move(animal.get_ongoing_movement(), delta)


func select_behavior(animal: Animal):
	is_matched = BehaviorCriteria.all_match(animal, criteria)
	if is_matched:
		return if_behavior


func reset():
	.reset()
	is_reset = true
	is_matched = false
	if_behavior.reset()


func describe():
	var c_names = []
	for c in criteria:
		c_names.append(c.describe())

	return "when(%s, %s)" % [c_names, if_behavior.describe()]
