extends SpecBase

class_name BaseDistribution


func _is_constant() -> bool:
	return HARD.stop("need to override _is_constant")


func _new_value(_rng: RandomNumberGenerator):
	return HARD.stop("need to override _new_value")


func new_value(key: String):
	var rng = CleanRNG.get_global_rng()
	return _new_value(rng.get_rng(key))


func new_int(key: String) -> int:
	var value = new_value(key)
	match typeof(value):
		TYPE_INT:
			return value
		_:
			return HARD.stop("invalid distribution value: %s", value)


func new_float(key: String) -> float:
	var value = new_value(key)
	match typeof(value):
		TYPE_REAL:
			return value
		TYPE_INT:
			return float(value)
		_:
			return HARD.stop("invalid distribution value: %s", value)


func new_vec_2(key: String) -> Vector2:
	var value = new_value(key)
	match typeof(value):
		TYPE_VECTOR2:
			return value
		_:
			return HARD.stop("invalid distribution value: %s", value)


func new_vec_3(key: String) -> Vector3:
	var value = new_value(key)
	match typeof(value):
		TYPE_VECTOR3:
			return value
		_:
			return HARD.stop("invalid distribution value: %s", value)
