extends Reference

class_name HARD


static func mode() -> bool:
	return Array(OS.get_cmdline_args()).has("--dev")


static func assert(condition, text: String = "assertion failed", args = null):
	if condition:
		return condition
	else:
		return stop(text, args, 2)


static func stop(text: String, args = null, stack_skip := 1):
	if typeof(args) != TYPE_ARRAY:
		args = [args]
	if args.size() == text.count("%s"):
		text = text % args
	print()
	printerr()
	push_error(text)
	printerr()
	stackerr(stack_skip)
	printerr()
	if mode():
		breakpoint
	else:
		quit()


static func stackerr(stack_skip := 0) -> void:
	var stack := get_stack()
	for i in range(stack_skip + 1, stack.size()):
		var frame: Dictionary = stack[i]
		var s = frame.source
		var n = frame.line
		var f = frame.function
		var c = "  "
		if f.begins_with("_"):
			c = " "
		printerr("%-50s: %4d %s%s" % [s, n, c, f])


static func quit():
	var _dead := OS.kill(OS.get_process_id())
