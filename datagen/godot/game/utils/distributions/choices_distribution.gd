extends BaseDistribution

class_name ChoicesDistribution

var value_choice: Array
var value_chance: Array

var _weights_cdf: Array


func _init(_values = null, _weights = null):
	# early exit if we are deserializing
	if (_values == null) and (_weights == null):
		return
	value_choice = _values
	value_chance = _weights
	_post_init()


func _post_init():
	HARD.assert(
		len(value_choice) == len(value_chance), "Must have equal number of choices and chances."
	)
	_weights_cdf = _weights_to_cdf_array(value_chance)


func _is_constant() -> bool:
	return len(value_choice) == 1


func _new_value(rng: RandomNumberGenerator):
	var choice_index: int = _weights_cdf.bsearch(rng.randf())
	return value_choice[choice_index]


static func _weights_to_cdf_array(array: PoolRealArray) -> Array:
	var a_sum := 0.0
	for value in array:
		a_sum += abs(value)
	var clone := []
	var r_sum := 0.0
	for value in array:
		r_sum += abs(value)
		clone.append(r_sum / a_sum)
	return clone
