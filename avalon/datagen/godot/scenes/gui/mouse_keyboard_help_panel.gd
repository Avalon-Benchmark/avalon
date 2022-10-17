extends PopupPanel
class_name MouseKeyboardHelpPanel

const TOGGLE_EVENT := "toggle_help"
const BACKGROUND := Color(0.745098, 0.745098, 0.745098, 0.5)

const non_wasd_bindings = [
	{"Climb Up": "move_forward"},
	{"Climb Down": "move_backward"},
	{"Grab/Release": "grab"},
	"eat",
	"throw",
	"jump",
	"crouch",
	{"Extend hand": "wheel_up"},
	{"Retract hand": "wheel_down"},
	"toggle_active_hand",
	{"Reset back to world selection": "reset"},
	{"Toggle this Popup": TOGGLE_EVENT},
	# TODO remove or make use of unbound bindings:
	# "move_toggle_rotation",
	# "toggle_mouse_capture",
	# "toggle_free_movement",
	# "save",
	# "mouse_left",
	# "mouse_right",
]


func _init():
	add_child(describe_controls())
	set_as_minsize()
	var center_x = get_rect().size.x / 2
	set_position(Vector2(center_x, 20))


func _ready():
	var styles = StyleBoxFlat.new()
	styles.bg_color = BACKGROUND
	add_stylebox_override("panel", styles)
	visible = true


func describe_controls():
	var vbox = VBoxContainer.new()
	var styles = StyleBoxFlat.new()
	styles.bg_color = Color.gray
	add_stylebox_override("panel", styles)
	vbox.add_child(new_label("Keyboard and Mouse Controls:"))
	vbox.add_child(control_desc("Movement", wasd()))
	for binding in non_wasd_bindings:
		var control: Control
		if binding is String:
			var label = snake_to_title_case(binding)
			control = control_desc(label, binding_repr(binding))
		elif binding is Dictionary:
			var label = binding.keys()[0]
			control = control_desc(label, binding_repr(binding[label]))
		else:
			HARD.stop("invalid binging %s" % binding)
		vbox.add_child(control)

	var container = MarginContainer.new()
	var margin_value = 10
	container.add_constant_override("margin_top", margin_value)
	container.add_constant_override("margin_left", margin_value)
	container.add_constant_override("margin_bottom", margin_value)
	container.add_constant_override("margin_right", margin_value)
	container.add_child(vbox)
	return container


static func control_desc(label_text: String, details: Control) -> Control:
	var row = HBoxContainer.new()
	row.add_child(new_label(label_text + ":"))

	details.size_flags_horizontal = Control.SIZE_EXPAND + Control.SIZE_SHRINK_END
	row.add_child(details)
	return row


static func new_label(label_text: String) -> Label:
	var label = Label.new()
	label.align = Label.ALIGN_FILL
	label.text = label_text
	label.add_color_override("font_color", Color.darkslategray)
	return label


static func wasd() -> VBoxContainer:
	var row_one = HBoxContainer.new()
	var row_two = HBoxContainer.new()
	row_one = row([binding_repr("move_forward")])
	row_two = row(
		[
			binding_repr("move_left"),
			binding_repr("move_backward"),
			binding_repr("move_right"),
		]
	)
	return column([row_one, row_two])


static func binding_repr(name: String) -> Control:
	var bindings = InputMap.get_action_list(name)
	match len(bindings):
		0:
			return new_button("Unbound!")
		1:
			return new_button(input_event_as_text(bindings[0]))
		_:
			var reprs = []
			for i in len(bindings):
				reprs.append(new_button(input_event_as_text(bindings[i])))
				if i < len(bindings) - 1:
					reprs.append(new_label(", "))
			return row(reprs)


static func input_event_as_text(event: InputEvent) -> String:
	if event is InputEventMouseButton:
		return mouse_button_name(event.button_index)

	var text = event.as_text()
	if text == "":
		text = OS.get_scancode_string(event.get_physical_scancode_with_modifiers())
	if "Shift+Slash" in text:
		text = "?"
	return text


static func mouse_button_name(button_index: int) -> String:
	match button_index:
		BUTTON_LEFT:
			return "Left Click"
		BUTTON_RIGHT:
			return "Right Click"
		BUTTON_WHEEL_DOWN:
			return "Mouse Wheel Down"
		BUTTON_WHEEL_UP:
			return "Mouse Wheel Up"
		_:
			HARD.stop("unexpected mouse binding")
	return "Impossible Click"


static func row(controls: Array, alignment := BoxContainer.ALIGN_CENTER) -> HBoxContainer:
	var row = HBoxContainer.new()
	row.alignment = alignment
	for control in controls:
		row.add_child(control)
	return row


static func column(controls: Array) -> VBoxContainer:
	var column = VBoxContainer.new()
	for control in controls:
		column.add_child(control)
	return column


static func new_button(text: String) -> Button:
	var button = Button.new()
	button.disabled = true
	button.add_color_override("font_color_disabled", Color.antiquewhite)
	button.add_color_override("disabled", Color.darkslategray)
	button.add_stylebox_override("focus", StyleBoxEmpty.new())
	button.add_stylebox_override("hover", StyleBoxEmpty.new())
	button.text = text
	return button


static func snake_to_title_case(text: String) -> String:
	var words := PoolStringArray()
	for word in text.split("_"):
		words.append(word.capitalize())
	return words.join(" ")


func _input(event: InputEvent):
	if event.is_action_pressed(TOGGLE_EVENT):
		visible = not visible
	elif visible and event.is_action_pressed("ui_cancel"):
		visible = false
