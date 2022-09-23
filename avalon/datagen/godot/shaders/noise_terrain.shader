shader_type spatial;
render_mode blend_mix,depth_draw_opaque,cull_back,diffuse_burley,specular_schlick_ggx;

uniform vec4 color1 : hint_color = vec4(0.5, 0.5, 0.5, 1.0);
uniform vec4 color2 : hint_color = vec4(0.0, 0.0, 0.0, 1.0);
uniform float random_seed = 0.0;

uniform float magnitude :  hint_range(0, 1, 0.1) = 0.2;
uniform float spatial_scale :  hint_range(0, 1000, 0.1) = 0.5;

uniform float specular;
uniform float metallic;
uniform float roughness : hint_range(0, 1);

varying vec4 global_vertex;


vec3 hash(vec3 p) {
    p += random_seed;
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));

    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float noise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);

  return mix(mix(mix(dot(hash(i + vec3(0.0, 0.0, 0.0)), f - vec3(0.0, 0.0, 0.0)),
                     dot(hash(i + vec3(1.0, 0.0, 0.0)), f - vec3(1.0, 0.0, 0.0)), u.x),
                 mix(dot(hash(i + vec3(0.0, 1.0, 0.0)), f - vec3(0.0, 1.0, 0.0)),
                     dot(hash(i + vec3(1.0, 1.0, 0.0)), f - vec3(1.0, 1.0, 0.0)), u.x), u.y),
             mix(mix(dot(hash(i + vec3(0.0, 0.0, 1.0)), f - vec3(0.0, 0.0, 1.0)),
                     dot(hash(i + vec3(1.0, 0.0, 1.0)), f - vec3(1.0, 0.0, 1.0)), u.x),
                 mix(dot(hash(i + vec3(0.0, 1.0, 1.0)), f - vec3(0.0, 1.0, 1.0)),
                     dot(hash(i + vec3(1.0, 1.0, 1.0)), f - vec3(1.0, 1.0, 1.0)), u.x), u.y), u.z );
}

void vertex() {
	global_vertex = (WORLD_MATRIX * vec4(VERTEX, 1.0));
}

void fragment() {
	vec3 unit;
	
	//vec4 vertex = (CAMERA_MATRIX * vec4(VERTEX, 1.0));
	unit = (global_vertex.xyz / global_vertex.w) * vec3(spatial_scale);
	
    float n = noise(unit * 5.0) * 1.0;
	n += noise(unit * 10.0) * 0.5;
	n += noise(unit * 20.0) * 0.25;
	n += noise(unit * 40.0) * 0.125;
	
	// because the previous value is 0-mean, shift up to 0.5
	// need to clamp because the range is -1.0 to 1.0 (but most values are closer to the center, so this looks nicer)
	n = clamp(n + 0.5, 0.0, 1.0);
	vec3 color3 = mix(color1.rgb, color2.rgb, n);
	ALBEDO = mix(COLOR.rgb, color3, magnitude);

	SPECULAR = specular;
	METALLIC = metallic;
	ROUGHNESS = roughness;
}
