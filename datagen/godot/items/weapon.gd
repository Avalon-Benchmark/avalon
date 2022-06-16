extends Item

class_name Weapon

export var damage := 1.0


func is_throwable() -> bool:
	return true


func is_grabbable() -> bool:
	return true


func is_pushable() -> bool:
	return true


func get_inflicted_damage(_impact_velocity: Vector3) -> float:
	return damage
