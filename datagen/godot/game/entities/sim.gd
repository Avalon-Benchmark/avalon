extends Node

class_name Sim

var sim_spec: SimSpec


func _init(_sim_spec: SimSpec):
	sim_spec = _sim_spec
	var scene: PackedScene = ResourceLoader.load(sim_spec.scene_path)
	self.add_child(scene.instance())


func get_features() -> Dictionary:
	var data = {}
	data[CONST.DATASET_ID_FEATURE] = sim_spec.dataset_id
	data[CONST.LABEL_FEATURE] = sim_spec.label

	if "event" in self:
		data[CONST.EVENT_HAPPENED_FEATURE] = int(
			Globals.time_in_seconds >= self.event.time_in_seconds
		)
	else:
		data[CONST.EVENT_HAPPENED_FEATURE] = 0

	return data
