extends MeshInstance

class_name SphereMarker

export var color = "#ff0808"


func _ready():
	var material = get_surface_material(0)
	material.albedo_color = Color(color)
	set_surface_material(0, material)
