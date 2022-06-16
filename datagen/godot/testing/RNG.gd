extends MainLoop

# Somewhat unstructured testes for RNG reproducibility.
#
# TL;DR:
#
# This works:
# var seed := 42
# var rng1 := RandomNumberGenerator.new(); rng1.seed = seed
# var rng2 := RandomNumberGenerator.new(); rng2.seed = seed
# assert(rng1.randi() == rng2.randi())
#
# This won't:
# var rng1 := RandomNumberGenerator.new(); rng1.randi()
# var rng2 := RandomNumberGenerator.new(); rng2.seed = rng1.seed
# assert(rng1.randi() == rng2.randi())
#
# This would:
# var rng1 := RandomNumberGenerator.new(); rng1.randi()
# var rng2 := RandomNumberGenerator.new(); clone(rng1, rng2.seed)
#
# OK but why?
# Left as an exercise to the reader.
# More seriously, ask me to type it. :)
#
# The primary PCG implementation + details:
# https://www.pcg-random.org/download.html
#
# The actual code in Godot:
# https://github.com/godotengine/godot/blob/3.2/core/math/random_pcg.h
# https://github.com/godotengine/godot/blob/3.2/core/math/random_pcg.cpp
# https://github.com/godotengine/godot/blob/3.2/thirdparty/misc/pcg.h
# https://github.com/godotengine/godot/blob/3.2/thirdparty/misc/pcg.cpp

# The PCG constants.
const PCG_ADD_64 := +1442695040888963407 << 1 | 1
const PCG_MUL_64 := +6364136223846793005

# Godot doesn't set the PCG state directly, and .seed is the previous RNG state.
# Multiplicative inverse of PCG_MUL_64 mod 2**64, found with extended GCD.
const PCG_INV_64 := -4568919932995229531


func _initialize():
	pass


func _idle(_delta: float) -> bool:
	var i := 0
	var j := 10
	var k := 42
	var s := 0
	var r := RandomNumberGenerator.new()

	r.seed = k
	i = 0
	while i < j:
		printt(i, r.seed, r.randi())
		i += 1
		if i == 5:
			s = r.seed

	print()
	printt("?", s)
	print()

	self.clone(r, s)
	i = 0
	while i < j:
		printt(i + 5, r.seed, r.randi())
		i += 1

	return true


func _finalize():
	pass


func clone(rng: RandomNumberGenerator, clone_seed: int):
	rng.seed = PCG_INV_64 * (clone_seed - PCG_ADD_64) - PCG_ADD_64
	# warning-ignore:return_value_discarded
	rng.randi()


func rng_next(r_seed: int) -> int:
	return r_seed * PCG_MUL_64 + PCG_ADD_64


func rng_ffwd(r_seed: int, r_delta: int) -> int:
	var cur_mul := PCG_MUL_64
	var cur_inc := PCG_ADD_64
	var acc_mul := 1
	var acc_inc := 0

	while r_delta:
		if r_delta & 1:
			acc_mul = acc_mul * cur_mul
			acc_inc = acc_inc * cur_mul + cur_inc
		cur_inc = (cur_mul + 1) * cur_inc
		cur_mul = cur_mul * cur_mul
		r_delta /= 2
	return acc_mul * r_seed + acc_inc
