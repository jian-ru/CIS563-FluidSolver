#version 450 core


layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) writeonly buffer zerobuffer_block0
{
	int data[];
};


layout(location = 0) uniform int bufferSize;


void main()
{
	uint tID = gl_GlobalInvocationID.x;
	
	if (tID < bufferSize)
	{
		data[tID] = 0;
	}
}