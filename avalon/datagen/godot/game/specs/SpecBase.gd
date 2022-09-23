extends Reference

class_name SpecBase

var _config_path: String


func _post_init():
	pass


func rng_key(key: String) -> String:
	return _config_path + "/" + key
