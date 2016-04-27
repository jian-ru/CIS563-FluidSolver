#version 450 core

in vec3 frag_color;

out vec4 color;

//uniform vec3 uniColor;

void main()
{
	color = vec4(frag_color, 1.0);
}