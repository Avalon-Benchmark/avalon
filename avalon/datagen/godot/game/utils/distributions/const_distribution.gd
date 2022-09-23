extends BaseDistribution

class_name ConstDistribution

var value


func _init(_value = null):
	# early exit if we are deserializing
	if _value == null:
		return
	value = _value
	_post_init()


func _is_constant() -> bool:
	return true


func _new_value(_rng: RandomNumberGenerator):
	return value
