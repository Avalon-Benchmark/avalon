extends InputCollector

class_name CombinedInputCollector

var input_collectors: Array  # InputCollector


func _init(_input_collectors: Array):
	input_collectors = _input_collectors


func reset():
	for ic in input_collectors:
		ic.reset()


func get_actions(controlled_nodes: Array) -> Array:
	var actions = []
	for index in len(input_collectors):
		var ic: InputCollector = input_collectors[index]
		var controlled_node = controlled_nodes[index]
		actions.append(ic.get_action(controlled_node))
	return actions


func read_input_from_event(event: InputEvent) -> void:
	for _ic in input_collectors:
		var ic: InputCollector = _ic
		ic.read_input_from_event(event)


func read_input_before_physics() -> void:
	for _ic in input_collectors:
		var ic: InputCollector = _ic
		ic.read_input_before_physics()


func read_input_from_pipe(action_pipe: StreamPeerBuffer) -> void:
	for _ic in input_collectors:
		var ic: InputCollector = _ic
		ic.read_input_from_pipe(action_pipe)


func read_input_from_data(action: PoolRealArray) -> void:
	HARD.assert(len(input_collectors) == 1, "Combined read_input_from_data currently unsupported")
	var ic: InputCollector = input_collectors[0]
	ic.read_input_from_data(action)


func write_into_stream(stream: StreamPeerBuffer, controlled_nodes) -> void:
	for i in len(input_collectors):
		var ic: InputCollector = input_collectors[i]
		ic.write_into_stream(stream, controlled_nodes[i])
