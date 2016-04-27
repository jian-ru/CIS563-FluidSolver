#version 450 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 velocity;

out vec3 frag_color;

uniform mat4 MVP;

void main()
{
	float speed = length(velocity) / 10.0;
	
	frag_color = vec3(speed, speed, 1.0);
	gl_Position = MVP * vec4(position, 1.0);
}