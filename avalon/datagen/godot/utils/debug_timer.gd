extends Reference
class_name DebugTimer

const TIMERS := []

var section: String
var t_index: int

var t_start: int
var t_close: int
var t_print: int

var t_timed: int
var n_timed: int


func _init(name: String = "NO_NAME"):
	section = name
	t_index = TIMERS.size()
	TIMERS.append(name)
	t_timed = 0
	n_timed = 0


func ticks() -> int:
	return OS.get_ticks_usec()


func start():
	t_start = ticks()


func close():
	t_close = ticks()
	t_timed = t_timed + (t_close - t_start)
	n_timed = n_timed + 1

	if n_timed % 2000:
		return

	var s_timed := t_timed / 1_000.0
	var s_total := t_close / 1_000.0
	var f_total := s_timed / s_total

	if t_index == 0:
		print("DEBUG TIMER: runtime: %8.2f ms" % [s_total])
		print("DEBUG TIMER: section: %8.2f ms (%2.0f%%) @ %s" % [s_timed, 100 * f_total, section])
	else:
		print("DEBUG TIMER: section: %8.2f ms (%2.0f%%) @ %s" % [s_timed, 100 * f_total, section])
