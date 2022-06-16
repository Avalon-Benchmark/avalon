extends Reference

class_name SpecBase

var _config_path: String


func build(_is_registered: bool = true) -> Object:
	HARD.stop("Specs can't be built, you want to override this in a Builder")
	return null


func _post_init():
	pass


func rng_key(key: String) -> String:
	return _config_path + "/" + key
