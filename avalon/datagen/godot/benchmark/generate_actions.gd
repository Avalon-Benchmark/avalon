extends SceneTree


func _init():
	var _dir := Directory.new()
	var _err := _dir.make_dir("/dev/shm/godot")

	write_actions("/dev/shm/godot/perf_032.actions", "res://benchmark/perf_032/main.tscn")
	write_actions("/dev/shm/godot/perf_064.actions", "res://benchmark/perf_064/main.tscn")
	write_actions("/dev/shm/godot/perf_220.actions", "res://benchmark/perf_220/main.tscn")
	write_actions("/dev/shm/godot/perf_440.actions", "res://benchmark/perf_440/main.tscn")

	quit(0)


func write_actions(file_path: String, level_path: String, n_steps = 1_000_000):
	var file := File.new()
	var _err := file.open(file_path, File.WRITE)

	if _err != OK:
		quit(2)
		return

	var start_data := [0, null, 2137, level_path, 100.0]
	var steps_data := [3, PoolRealArray([0.0, 0.0, -0.001, 0.0, 0.0, 0.0])]
	var close_data := [6]

	file.store_var(start_data)
	for _i in range(n_steps):
		file.store_var(steps_data)
	file.store_var(close_data)
