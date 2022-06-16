extends BaseDistribution

class_name GaussianDistribution

var value_mean
var value_std


func _init(_value_mean = null, _value_std = null):
	# early exit if we are deserializing
	if (_value_mean == null) and (_value_std == null):
		return
	value_mean = _value_mean
	value_std = _value_std
	_post_init()


func _is_constant() -> bool:
	return value_mean == 0 and value_std == 0


func _new_value(rng: RandomNumberGenerator):
	match typeof(value_mean):
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
		return rng.randfn(self.value_mean, self.value_std)
	else:
		return rng.randfn(self.value_mean[index], self.value_std[index])
