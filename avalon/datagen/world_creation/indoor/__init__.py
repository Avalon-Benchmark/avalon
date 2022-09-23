"""
Some notes on notation for indoor worlds:
Godot uses Y as the vertical axis, while X is east-west, and Z is south-north. However, most of the code here
operates in 2D top-down representation, opening room for confusion of which numpy dimension maps to which axis name.

To keep things consistent, we maintain the same axis notation for numpy arrays as well as Godot:
 - For 2D numpy arrays, we keep the row-major axis ordering, so the notation is (z,x)
 - For 3D numpy arrays, the extra axis is the vertical axis: (z,x,y),
 where z = length, x = width and y = height, as in Godot.
 - For points/vertices, we maintain the mathematical notation of x,y,z to stay consistent with existing datatypes
"""
