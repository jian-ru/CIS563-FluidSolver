#version 450 core


layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 7) writeonly buffer sv_block7
{
	float uVels2[];
};

layout(std430, binding = 8) writeonly buffer sv_block8
{
	float vVels2[];
};

layout(std430, binding = 9) writeonly buffer sv_block9
{
	float wVels2[];
};

layout(std430, binding = 10) readonly buffer sv_block10
{
	float uVels1[];
};

layout(std430, binding = 11) readonly buffer sv_block11
{
	float vVels1[];
};

layout(std430, binding = 12) readonly buffer sv_block12
{
	float wVels1[];
};


layout(std140, binding = 1) uniform cbMeta
{
	int xcount;
	int ycount;
	int zcount;
	float cellSize;
	int numParticles;
	int numCells;
	float deltaTIme;
};


void main()
{
	int id = int(gl_GlobalInvocationID.x);
	
	if (id < numCells)
	{
		uVels2[id] = uVels1[id];
		vVels2[id] = vVels1[id];
		wVels2[id] = wVels1[id];
	}
}