#version 450 core

#define FLUID_CELL 1
#define GRAVITY_ACCELERATION -9.8


layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer abg_block0
{
	int cellTypes[];
};

layout(std430, binding = 1) buffer abg_block1
{
	float vVels[];
};


layout(std140, binding = 1) uniform cbMeta
{
	int xcount;
	int ycount;
	int zcount;
	float cellSize;
	int numParticles;
	int numCells;
	float deltaTime;
};


void get_ijk(in int flatIdx, out ivec3 ijk)
{
	int pageSize = ycount * zcount;
	
	ijk.x = flatIdx / pageSize;
	ijk.y = flatIdx % pageSize / zcount;
	ijk.z = flatIdx % zcount;
}


void main()
{
	int id = int(gl_GlobalInvocationID.x);
	
	if (id < numCells)
	{
		int ct = cellTypes[id];
		ivec3 ijk;
		
		get_ijk(id, ijk);
		
		float notOnBoundary = float(ijk.y != 0 && ijk.y != 1 && ijk.y != ycount - 1);
		
		vVels[id] += float(ct == FLUID_CELL) * notOnBoundary *
		             GRAVITY_ACCELERATION * deltaTime;
	}
}