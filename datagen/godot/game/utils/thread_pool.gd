extends Reference

class_name ThreadPool

var nthreads := 0
var nbuffers := 0

var _threads := []
var _buffers := []

var _shared := Mutex.new()

var _thread_sync := Semaphore.new()
var _buffer_sync := Semaphore.new()

var _pool_opened := false
var _pool_closed := false

var _semhits := PoolIntArray([0, 0, 0, 0])
var _bufhits := PoolIntArray()

enum STATS {
	THREAD_POST,
	THREAD_WAIT,
	BUFFER_POST,
	BUFFER_WAIT,
}


static func work_init_context(_tid: int):
	return 0


static func work(_task, _context):
	HARD.assert(false, "thread pool missing a .work() method override")
	return _context + 1


static func work_done(_context):
	return _context


func start(threads := 4, buffers := 8) -> void:
	_shared.lock()
	HARD.assert(_pool_opened == false, "Unhandled error")
	HARD.assert(threads > 0, "Threads must be greater than 0.")
	HARD.assert(buffers > threads, "Buffers must be greater than threads.")
	_pool_opened = true
	nthreads = threads
	nbuffers = buffers
	_threads.resize(threads)
	for i in range(nthreads):
		var it := Thread.new()
		var ok := it.start(self, "_thread_loop", i)
		HARD.assert(ok == OK, "Unhandled error")
		_threads[i] = it
	_buffers.resize(buffers)
	_bufhits.resize(buffers)
	for i in range(nbuffers):
		_writer_poke()
		_buffers[i] = null
		_bufhits[i] = 0
	_shared.unlock()


func post(task) -> void:
	_writer_wait()
	_shared.lock()
	_buffer_send(task)
	_shared.unlock()
	_reader_poke()


func wait():
	_shared.lock()
	print("done: %s %s" % [self._bufhits, self._semhits])
	HARD.assert(_pool_opened == true, "Unhandled error")
	HARD.assert(_pool_closed == false, "Unhandled error")
	_pool_closed = true
	for i in range(nthreads):
		var it: Thread = _threads[i]
		HARD.assert(it.is_active(), "Unhandled error")
		_reader_poke()
	_shared.unlock()
	for i in range(nthreads):
		var it: Thread = _threads[i]
		_threads[i] = it.wait_to_finish()
	for i in range(nbuffers):
		HARD.assert(_buffers[i] == null, "Unhandled error")


func _thread_loop(t_id: int):
	var context = self.work_init_context(t_id)
	while true:
		_reader_wait()
		_shared.lock()
		var task = _buffer_recv()
		var exit = _pool_closed
		_shared.unlock()
		if task != null:
			_writer_poke()
			context = self.work(task, context)
		else:
			HARD.assert(exit, "Unhandled error")
			break
	var context_done = self.work_done(context)
	return context_done


func _reader_poke() -> void:
	while _thread_sync.post():
		_semhits[STATS.THREAD_POST] += 1


func _reader_wait() -> void:
	while _thread_sync.wait():
		_semhits[STATS.THREAD_WAIT] += 1


func _writer_poke() -> void:
	while _buffer_sync.post():
		_semhits[STATS.BUFFER_POST] += 1


func _writer_wait() -> void:
	while _buffer_sync.wait():
		_semhits[STATS.BUFFER_WAIT] += 1


func _buffer_send(task):
	var i := 0
	while i < nbuffers:
		if _buffers[i] == null:
			break
		else:
			i += 1
	HARD.assert(i != nbuffers, "Unhandled error")
	_buffers[i] = task
	_bufhits[i] += 1


func _buffer_recv():
	var task = null
	var i := 0
	while i < nbuffers:
		if _buffers[i] != null:
			break
		else:
			i += 1
	if i != nbuffers:
		task = _buffers[i]
		_buffers[i] = null
	return task
