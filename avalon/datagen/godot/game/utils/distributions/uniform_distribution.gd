extends BaseDistribution

class_name UniformDistribution

var value_min
var value_max


func _init(_value_min = null, _value_max = null):
	# early exit if we are deserializing
	if (_value_min == null) and (_value_max == null):
		return
	value_min = _value_min
	value_max = _value_max
	_post_init()


func _is_constant() -> bool:
	return value_min == value_max


func _new_value(rng: RandomNumberGenerator):
	match typeof(value_min):
		TYPE_INT:
			return _rng(rng)
		TYPE_REAL:
			return _rng(rng)
		TYPE_VECTOR2:
			return Vector2(_rng(rng, 0), _rng(rng, 1))
		TYPE_VECTOR3:
			return Vector3(_rng(rng, 0), _rng(rng, 1), _rng(rng, 2))
		_:
			HARD.stop("unknown distribution value type")


# TODO: rename
func _rng(rng: RandomNumberGenerator, index: int = -1) -> float:
	if index < 0:
		return rng.randf_range(self.value_min, self.value_max)
	else:
		return rng.randf_range(self.value_min[index], self.value_max[index])
