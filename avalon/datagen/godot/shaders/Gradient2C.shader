shader_type spatial;
render_mode blend_mix,depth_draw_opaque,cull_back,diffuse_burley,specular_schlick_ggx;
uniform float specular;
uniform float metallic;
uniform float roughness : hint_range(0, 1);

uniform vec4 color1 : hint_color = vec4(1.0, 1.0, 1.0, 1.0);
uniform vec4 color2 : hint_color = vec4(0.0, 0.0, 0.0, 1.0);

uniform float angle : hint_range(0, 90, 0.1) = 0.0;


void vertex() {
	COLOR = mix(color1, color2, mix(UV.x, UV.y, sin(radians(angle))));
}


//  A convenient property of vertex shaders - their outputs get interpolated!
//  COLOR is a builtin vec4 varying with no forced meaning, and we'll use it.


void fragment() {
	ALBEDO = COLOR.rgb;
	SPECULAR = specular;
	METALLIC = metallic;
	ROUGHNESS = roughness;
}
