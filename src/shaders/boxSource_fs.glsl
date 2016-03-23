#version 450 core

out vec4 color;

uniform vec3 uniColor;

void main()
{
	color = vec4(uniColor, 1.0);
}