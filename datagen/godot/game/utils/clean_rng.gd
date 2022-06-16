extends Reference

class_name CleanRNG

var _seed_int := 0
var _video_id := 0
var _frame_id := 0

const _cached_rng = {}
const _cached_str = {}

const _cached_self = {}


# TODO this is incredibly cursed
func _init():
	_cached_self["self"] = self


static func get_global_rng():
	return _cached_self["self"]


static func get_key(node: Node, key: String):
	return str(node.get_path()) + "/" + key


func get_rng(key: String) -> RandomNumberGenerator:
	var rng: RandomNumberGenerator = _cached_rng.get(key)
	if rng == null:
		rng = RandomNumberGenerator.new()
		rng.seed = self._str_hash(key)
		rng.seed = (rng.randi() << 32) ^ _seed_int
		rng.seed = (rng.randi() << 32) ^ _video_id
		rng.seed = (rng.randi() << 32) ^ _frame_id
		_cached_rng[key] = rng
	return rng


func set_seed(video: int, frame: int, r_int: int, r_key: String) -> void:
	_cached_rng.clear()
	_seed_int = r_int ^ self._str_hash(r_key)
	_video_id = video
	_frame_id = frame


func _str_hash(key: String) -> int:
	var hash_int = _cached_str.get(key)
	if hash_int != null:
		return hash_int
	hash_int = Tools.string_hash(key)
	_cached_str[key] = hash_int
	return hash_int
