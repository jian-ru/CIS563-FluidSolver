#version 450 core

//#define FLT_EPSILON 0.0
#define FLT_EPSILON 1e-7
#define FLUID_CELL 1


layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer bpib_block0
{
	float particles[];
};

layout(std430, binding = 1) readonly buffer bpib_block1
{
	int uParticleIndexOffsets[];
};

layout(std430, binding = 2) readonly buffer bpib_block2
{
	int vParticleIndexOffsets[];
};

layout(std430, binding = 3) readonly buffer bpib_block3
{
	int wParticleIndexOffsets[];
};

layout(std430, binding = 4) readonly buffer bpib_block4
{
	int uParticleCounts[];
};

layout(std430, binding = 5) readonly buffer bpib_block5
{
	int vParticleCounts[];
};

layout(std430, binding = 6) readonly buffer bpib_block6
{
	int wParticleCounts[];
};

layout(std430, binding = 7) buffer bpib_block7
{
	int uParticleCurCounts[];
};

layout(std430, binding = 8) buffer bpib_block8
{
	int vParticleCurCounts[];
};

layout(std430, binding = 9) buffer bpib_block9
{
	int wParticleCurCounts[];
};

layout(std430, binding = 10) writeonly buffer bpib_block10
{
	int uParticleIndexBuffer[];
};

layout(std430, binding = 11) writeonly buffer bpib_block11
{
	int vParticleIndexBuffer[];
};

layout(std430, binding = 12) writeonly buffer bpib_block12
{
	int wParticleIndexBuffer[];
};

layout(std430, binding = 13) writeonly buffer bpib_block13
{
	int cellTypes[];
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


void getCellType_ijk(in vec3 pos, out ivec3 ijk)
{
	ijk.x = int((pos.x - FLT_EPSILON) / cellSize);
	ijk.y = int((pos.y - FLT_EPSILON) / cellSize);
	ijk.z = int((pos.z - FLT_EPSILON) / cellSize);
}

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

void putIndexIntoU(in int i, in int j, in int k, in int pIdx)
{
	int idx; // flat index
	int ofs; // offset relative to cellOffset to put index into
	int cellOffset;
	
	idx = flatIdx(i, j, k);
	cellOffset = uParticleIndexOffsets[idx];
	ofs = atomicAdd(uParticleCurCounts[idx], 1);
	uParticleIndexBuffer[cellOffset + ofs] = pIdx;
}

void putIndexIntoV(in int i, in int j, in int k, in int pIdx)
{
	int idx; // flat index
	int ofs; // offset relative to cellOffset to put index into
	int cellOffset;
	
	idx = flatIdx(i, j, k);
	cellOffset = vParticleIndexOffsets[idx];
	ofs = atomicAdd(vParticleCurCounts[idx], 1);
	vParticleIndexBuffer[cellOffset + ofs] = pIdx;
}

void putIndexIntoW(in int i, in int j, in int k, in int pIdx)
{
	int idx; // flat index
	int ofs; // offset relative to cellOffset to put index into
	int cellOffset;
	
	idx = flatIdx(i, j, k);
	cellOffset = wParticleIndexOffsets[idx];
	ofs = atomicAdd(wParticleCurCounts[idx], 1);
	wParticleIndexBuffer[cellOffset + ofs] = pIdx;
}


void main()
{
	int id = int(gl_GlobalInvocationID.x);
	
	if (id < numParticles)
	{
		vec3 pos = vec3(particles[6 * id], particles[6 * id + 1], particles[6 * id + 2]);
		ivec3 ijk;
		
		// mark fluid cell
		getCellType_ijk(pos, ijk);
		cellTypes[flatIdx(ijk.x, ijk.y, ijk.z)] = FLUID_CELL;
		
		// u grid
		getu_ijk(pos, ijk);
		putIndexIntoU(ijk.x    , ijk.y    , ijk.z    , id);
		putIndexIntoU(ijk.x    , ijk.y    , ijk.z + 1, id);
		putIndexIntoU(ijk.x    , ijk.y + 1, ijk.z    , id);
		putIndexIntoU(ijk.x    , ijk.y + 1, ijk.z + 1, id);
		putIndexIntoU(ijk.x + 1, ijk.y    , ijk.z    , id);
		putIndexIntoU(ijk.x + 1, ijk.y    , ijk.z + 1, id);
		putIndexIntoU(ijk.x + 1, ijk.y + 1, ijk.z    , id);
		putIndexIntoU(ijk.x + 1, ijk.y + 1, ijk.z + 1, id);
		
		// v grid
		getv_ijk(pos, ijk);
		putIndexIntoV(ijk.x    , ijk.y    , ijk.z    , id);
		putIndexIntoV(ijk.x    , ijk.y    , ijk.z + 1, id);
		putIndexIntoV(ijk.x    , ijk.y + 1, ijk.z    , id);
		putIndexIntoV(ijk.x    , ijk.y + 1, ijk.z + 1, id);
		putIndexIntoV(ijk.x + 1, ijk.y    , ijk.z    , id);
		putIndexIntoV(ijk.x + 1, ijk.y    , ijk.z + 1, id);
		putIndexIntoV(ijk.x + 1, ijk.y + 1, ijk.z    , id);
		putIndexIntoV(ijk.x + 1, ijk.y + 1, ijk.z + 1, id);
		
		// w grid
		getw_ijk(pos, ijk);
		putIndexIntoW(ijk.x    , ijk.y    , ijk.z    , id);
		putIndexIntoW(ijk.x    , ijk.y    , ijk.z + 1, id);
		putIndexIntoW(ijk.x    , ijk.y + 1, ijk.z    , id);
		putIndexIntoW(ijk.x    , ijk.y + 1, ijk.z + 1, id);
		putIndexIntoW(ijk.x + 1, ijk.y    , ijk.z    , id);
		putIndexIntoW(ijk.x + 1, ijk.y    , ijk.z + 1, id);
		putIndexIntoW(ijk.x + 1, ijk.y + 1, ijk.z    , id);
		putIndexIntoW(ijk.x + 1, ijk.y + 1, ijk.z + 1, id);
	}
}