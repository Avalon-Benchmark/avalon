# Helpers for the usage of Nodes as persistent logic & data containers.
#
# "logic nodes" are only added to the tree in the event of persistence (i.e. snapshotting),
# and always have a default configuration set in parent constructor or _ready.
#
# > NOTE: In the future we may replace in-constructor defaults with in-scene instances of these nodes.
# > A snapshot should be usable as an automated refactoring tool for this purpose,
# > but be careful about shared parent variables.
# >
# > Trade-offs to consider:
# > * All animal logic could be made into behaviors and their scripts deleted
# > * No more coupling of top-level export vars to default behaviors (more granular behavior customization)
# > * Even less type control for the behavior tree
# > * Basically a lispy code-as-data situation
extends Reference

class_name LogicNodes

const LOGIC_NODE_LIST_NODE_NAME = "persisted_logic_nodes"


static func is_container(container: Node) -> bool:
	# TODO could directly set logic nodes in prefer persisted
	return container.has_method("get_logic_nodes")  # and container.has_method("load_or_init")


static func is_container_already_persiting(container: Node) -> bool:
	return container.has_node(LOGIC_NODE_LIST_NODE_NAME)


static func persist(container: Node):
	HARD.assert(is_container(container), "%s does not implement get_logic_nodes")
	if is_container_already_persiting(container):
		return
	var list_node = get_persisted(container)
	for logic_node in container.get_logic_nodes():
		list_node.add_child(logic_node)
		if is_container(logic_node):
			persist(logic_node)
	return


static func get_persisted(container: Node) -> Node:
	var list_node_name = LOGIC_NODE_LIST_NODE_NAME
	var list = container.get_node_or_null(list_node_name)
	if list != null:
		return list

	list = Node.new()
	list.name = list_node_name
	container.add_child(list)
	return list


static func prefer_persisted(
	container: Node, name: String, newly_configured_logic_node: Node, is_null_ok: bool = false
) -> Node:
	# outside of a tree path is empty, so this returns after a single conditional
	var persisted = container.get_node_or_null("%s/%s" % [LOGIC_NODE_LIST_NODE_NAME, name])
	if persisted != null:
		if is_instance_valid(newly_configured_logic_node):
			_free(newly_configured_logic_node)
		return persisted
	if is_null_ok and newly_configured_logic_node == null:
		return null
	newly_configured_logic_node.name = name
	return newly_configured_logic_node


static func prefer_persisted_array(container: Node, prefix: String, newly_configured_logic_nodes: Array) -> Array:
	var node_prefix = "%s__" % prefix

	if not is_container_already_persiting(container):
		for node in newly_configured_logic_nodes:
			if not node.name.begins_with(node_prefix):
				var subname = node.name if node.name != "" else node.script_name()
				node.name = node_prefix + subname
		return newly_configured_logic_nodes

	var selected = []
	for persisted in get_persisted(container).get_children():
		if persisted.name.begins_with(node_prefix):
			selected.append(persisted)

	for discard in newly_configured_logic_nodes:
		_free(discard)

	return selected


static func handle_predelete(container: Node) -> void:
	if is_container_already_persiting(container):
		return
	for logic_node in container.get_logic_nodes():
		if not is_instance_valid(logic_node):
			return
		_free(logic_node)


static func _free(logic_node: Node):
	if is_container(logic_node):
		handle_predelete(logic_node)
	logic_node.free()
