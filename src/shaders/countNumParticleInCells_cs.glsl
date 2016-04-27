#version 450 core

//#define FLT_EPSILON 0.0
#define FLT_EPSILON 1e-7


layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer cnpic_block0
{
	float particles[];
};

layout(std430, binding = 1) buffer cnpic_block1
{
	int uParticleCounts[];
};

layout(std430, binding = 2) buffer cnpic_block2
{
	int vParticleCounts[];
};

layout(std430, binding = 3) buffer cnpic_block3
{
	int wParticleCounts[];
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


void getu_ijk(in vec3 pos, out ivec3 ijk)
{
	float delta = 0.5 * cellSize;
	
	ijk.x = int((pos.x - FLT_EPSILON) / cellSize);
	ijk.y = int((pos.y - delta - FLT_EPSILON) / cellSize);
	ijk.z = int((pos.z - delta - FLT_EPSILON) / cellSize);
}

void getv_ijk(in vec3 pos, out ivec3 ijk)
{
	float delta = 0.5 * cellSize;
	
	ijk.x = int((pos.x - delta - FLT_EPSILON) / cellSize);
	ijk.y = int((pos.y - FLT_EPSILON) / cellSize);
	ijk.z = int((pos.z - delta - FLT_EPSILON) / cellSize);
}

void getw_ijk(in vec3 pos, out ivec3 ijk)
{
	float delta = 0.5 * cellSize;
	
	ijk.x = int((pos.x - delta - FLT_EPSILON) / cellSize);
	ijk.y = int((pos.y - delta - FLT_EPSILON) / cellSize);
	ijk.z = int((pos.z - FLT_EPSILON) / cellSize);
}

int flatIdx(in int i, in int j, in int k)
{
	return i * (ycount * zcount) + j * zcount + k;
}


void main()
{
	int id = int(gl_GlobalInvocationID.x); // which particle
	
	if (id < numParticles)
	{
		vec3 pos = vec3(particles[6 * id], particles[6 * id + 1], particles[6 * id + 2]);
		ivec3 ijk;
		int idx; // flat index
		
		// u grid
		getu_ijk(pos, ijk);
		idx = flatIdx(ijk.x, ijk.y, ijk.z);
		atomicAdd(uParticleCounts[idx], 1);
		idx = flatIdx(ijk.x, ijk.y, ijk.z + 1);
		atomicAdd(uParticleCounts[idx], 1);
		idx = flatIdx(ijk.x, ijk.y + 1, ijk.z);
		atomicAdd(uParticleCounts[idx], 1);
		idx = flatIdx(ijk.x, ijk.y + 1, ijk.z + 1);
		atomicAdd(uParticleCounts[idx], 1);
		idx = flatIdx(ijk.x + 1, ijk.y, ijk.z);
		atomicAdd(uParticleCounts[idx], 1);
		idx = flatIdx(ijk.x + 1, ijk.y, ijk.z + 1);
		atomicAdd(uParticleCounts[idx], 1);
		idx = flatIdx(ijk.x + 1, ijk.y + 1, ijk.z);
		atomicAdd(uParticleCounts[idx], 1);
		idx = flatIdx(ijk.x + 1, ijk.y + 1, ijk.z + 1);
		atomicAdd(uParticleCounts[idx], 1);
		
		// v grid
		getv_ijk(pos, ijk);
		idx = flatIdx(ijk.x, ijk.y, ijk.z);
		atomicAdd(vParticleCounts[idx], 1);
		idx = flatIdx(ijk.x, ijk.y, ijk.z + 1);
		atomicAdd(vParticleCounts[idx], 1);
		idx = flatIdx(ijk.x, ijk.y + 1, ijk.z);
		atomicAdd(vParticleCounts[idx], 1);
		idx = flatIdx(ijk.x, ijk.y + 1, ijk.z + 1);
		atomicAdd(vParticleCounts[idx], 1);
		idx = flatIdx(ijk.x + 1, ijk.y, ijk.z);
		atomicAdd(vParticleCounts[idx], 1);
		idx = flatIdx(ijk.x + 1, ijk.y, ijk.z + 1);
		atomicAdd(vParticleCounts[idx], 1);
		idx = flatIdx(ijk.x + 1, ijk.y + 1, ijk.z);
		atomicAdd(vParticleCounts[idx], 1);
		idx = flatIdx(ijk.x + 1, ijk.y + 1, ijk.z + 1);
		atomicAdd(vParticleCounts[idx], 1);
		
		// w grid
		getw_ijk(pos, ijk);
		idx = flatIdx(ijk.x, ijk.y, ijk.z);
		atomicAdd(wParticleCounts[idx], 1);
		idx = flatIdx(ijk.x, ijk.y, ijk.z + 1);
		atomicAdd(wParticleCounts[idx], 1);
		idx = flatIdx(ijk.x, ijk.y + 1, ijk.z);
		atomicAdd(wParticleCounts[idx], 1);
		idx = flatIdx(ijk.x, ijk.y + 1, ijk.z + 1);
		atomicAdd(wParticleCounts[idx], 1);
		idx = flatIdx(ijk.x + 1, ijk.y, ijk.z);
		atomicAdd(wParticleCounts[idx], 1);
		idx = flatIdx(ijk.x + 1, ijk.y, ijk.z + 1);
		atomicAdd(wParticleCounts[idx], 1);
		idx = flatIdx(ijk.x + 1, ijk.y + 1, ijk.z);
		atomicAdd(wParticleCounts[idx], 1);
		idx = flatIdx(ijk.x + 1, ijk.y + 1, ijk.z + 1);
		atomicAdd(wParticleCounts[idx], 1);
	}
}