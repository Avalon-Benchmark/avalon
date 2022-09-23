shader_type spatial;
render_mode blend_mix,depth_draw_opaque,cull_back,diffuse_burley,specular_schlick_ggx;
uniform float specular;
uniform float metallic;
uniform float roughness : hint_range(0, 1);

uniform vec4 color1 : hint_color = vec4(1.0, 1.0, 1.0, 1.0);
uniform vec4 color2 : hint_color = vec4(0.0, 0.0, 0.0, 1.0);


void fragment() {
	ALBEDO = color1.rgb;
	SPECULAR = specular;
	METALLIC = metallic;
	ROUGHNESS = roughness;
}
