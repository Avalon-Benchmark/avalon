extends Stackable

class_name Stone

onready var player = get_tree().root.find_node("player", true, false)
var extra_extents: StaticBody

export var extra_extent_factor := 1.3


func _ready():
	mass = 20.0
	linear_damp_factor = 0.5
	angular_damp_factor = 0.5

	# our body is super thin and it can be very easy to fall through stones
	#	so we add another collision shape that only the player colliders with
	extra_extents = StaticBody.new()
	extra_extents.set_collision_layer_bit(0, false)
	extra_extents.set_collision_mask_bit(0, false)
	extra_extents.set_collision_layer_bit(1, true)
	var collision = CollisionShape.new()
	collision.shape = BoxShape.new()
	collision.shape.extents = get_node("collision_shape").shape.extents * extra_extent_factor
	extra_extents.add_child(collision)
	add_child(extra_extents)


func _physics_process(_delta):
	if is_held:
		extra_extents.set_collision_layer_bit(1, false)
	else:
		extra_extents.set_collision_layer_bit(1, true)
	# this isn't that great because it breaks collisions but when players
	# 	are jumping on stones they've usually settled
	extra_extents.global_transform = global_transform
