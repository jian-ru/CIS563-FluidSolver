#version 450 core


layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) buffer uvupg_block0
{
	float uVels[];
};

layout(std430, binding = 1) buffer uvupg_block1
{
	float vVels[];
};

layout(std430, binding = 2) buffer uvupg_block2
{
	float wVels[];
};

layout(std430, binding = 3) readonly buffer uvupg_block3
{
	float pressures[];
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

int flatIdx(in int i, in int j, in int k)
{
	return i * (ycount * zcount) + j * zcount + k;
}


void main()
{
	int id = int(gl_GlobalInvocationID.x);
	
	if (id < numCells)
	{
		ivec3 ijk;
		int nidx;
		float dp;
		const float dt_dx = deltaTime / cellSize;
		
		get_ijk(id, ijk);
		
		// update u velocity
		if (ijk.x == 0 || ijk.x == 1 || ijk.x == xcount - 1)
		{
			uVels[id] = 0.0;
		}
		else
		{
			nidx = flatIdx(ijk.x - 1, ijk.y, ijk.z);
			dp = pressures[id] - pressures[nidx];
			uVels[id] -= dt_dx * dp;
		}
		
		// update v velocity
		if (ijk.y == 0 || ijk.y == 1 || ijk.y == ycount - 1)
		{
			vVels[id] = 0.0;
		}
		else
		{
			nidx = flatIdx(ijk.x, ijk.y - 1, ijk.z);
			dp = pressures[id] - pressures[nidx];
			vVels[id] -= dt_dx * dp;
		}
		
		// update w velocity
		if (ijk.z == 0 || ijk.z == 1 || ijk.z == zcount - 1)
		{
			wVels[id] = 0.0;
		}
		else
		{
			nidx = flatIdx(ijk.x, ijk.y, ijk.z - 1);
			dp = pressures[id] - pressures[nidx];
			wVels[id] -= dt_dx * dp;
		}
	}
}