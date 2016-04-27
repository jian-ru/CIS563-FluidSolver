#version 450 core


layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 1) buffer seg_add_block0
{
	int data[];
};


layout(location = 0) uniform int whichSegment;


void main()
{
	const int segmentSize = 2048;
	int preSum = data[whichSegment * segmentSize - 1];
	
	data[whichSegment * segmentSize + gl_LocalInvocationID.x * 2] += preSum;
	data[whichSegment * segmentSize + gl_LocalInvocationID.x * 2 + 1] += preSum;
}