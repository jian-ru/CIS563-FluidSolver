#version 450 core


layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer vmul_block0
{
	float in_a[];
};

layout(std430, binding = 1) readonly buffer vmul_block1
{
	float in_b[];
};

layout(std430, binding = 2) writeonly buffer vmul_block2
{
	float out_v[];
};


layout(std140, binding = 0) uniform cbMeta
{
	int size1D;
	float c1;
	float c2;
};


void main()
{
	int tID = int(gl_GlobalInvocationID.x);
	
	if (tID < size1D)
	{
		out_v[tID] = in_a[tID] * in_b[tID];
	}
}