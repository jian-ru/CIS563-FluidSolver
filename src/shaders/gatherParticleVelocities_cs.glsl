#version 450 core

//#define FLT_EPSILON 0.0
#define FLT_EPSILON 1e-7
#define SOLID_CELL 2


layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer gpv_block0
{
	float particles[];
};

layout(std430, binding = 1) readonly buffer gpv_block1
{
	int uParticleIndexOffsets[];
};

layout(std430, binding = 2) readonly buffer gpv_block2
{
	int vParticleIndexOffsets[];
};

layout(std430, binding = 3) readonly buffer gpv_block3
{
	int wParticleIndexOffsets[];
};

layout(std430, binding = 4) readonly buffer gpv_block4
{
	int uParticleCounts[];
};

layout(std430, binding = 5) readonly buffer gpv_block5
{
	int vParticleCounts[];
};

layout(std430, binding = 6) readonly buffer gpv_block6
{
	int wParticleCounts[];
};

layout(std430, binding = 7) readonly buffer gpv_block7
{
	int uParticleIndexBuffer[];
};

layout(std430, binding = 8) readonly buffer gpv_block8
{
	int vParticleIndexBuffer[];
};

layout(std430, binding = 9) readonly buffer gpv_block9
{
	int wParticleIndexBuffer[];
};

layout(std430, binding = 10) buffer gpv_block10
{
	float uVels[];
};

layout(std430, binding = 11) buffer gpv_block11
{
	float vVels[];
};

layout(std430, binding = 12) buffer gpv_block12
{
	float wVels[];
};

layout(std430, binding = 13) buffer gpv_block13
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


void get_ijk(in int flatIdx, out ivec3 ijk)
{
	int pageSize = ycount * zcount;
	
	ijk.x = flatIdx / pageSize;
	ijk.y = flatIdx % pageSize / zcount;
	ijk.z = flatIdx % zcount;
}

float computeWeightU(in vec3 pos, in vec3 cxyz)
{
	float delta = 0.5 * cellSize;
	float ux = pos.x;
	float uy = pos.y - delta;
	float uz = pos.z - delta;
	float wu = 1.0 - abs(ux - cxyz.x) / cellSize;
	float wv = 1.0 - abs(uy - cxyz.y) / cellSize;
	float ww = 1.0 - abs(uz - cxyz.z) / cellSize;
	
	return wu * wv * ww;
}

float computeWeightV(in vec3 pos, in vec3 cxyz)
{
	float delta = 0.5 * cellSize;
	float vx = pos.x - delta;
	float vy = pos.y;
	float vz = pos.z - delta;
	float wu = 1.0 - abs(vx - cxyz.x) / cellSize;
	float wv = 1.0 - abs(vy - cxyz.y) / cellSize;
	float ww = 1.0 - abs(vz - cxyz.z) / cellSize;
	
	return wu * wv * ww;
}

float computeWeightW(in vec3 pos, in vec3 cxyz)
{
	float delta = 0.5 * cellSize;
	float wx = pos.x - delta;
	float wy = pos.y - delta;
	float wz = pos.z;
	float wu = 1.0 - abs(wx - cxyz.x) / cellSize;
	float wv = 1.0 - abs(wy - cxyz.y) / cellSize;
	float ww = 1.0 - abs(wz - cxyz.z) / cellSize;
	
	return wu * wv * ww;
}


void main()
{
	int id = int(gl_GlobalInvocationID.x);
	
	if (id < numCells)
	{
		ivec3 ijk;
		vec3 cxyz;
		
		get_ijk(id, ijk);
		cxyz = vec3(ijk.x * cellSize,
				    ijk.y * cellSize,
					ijk.z * cellSize); // beginning of the cell
		
		int cellOffset;
		int numIndices;
		int pIdx;
		vec3 pos, vel;
		float weight;
		float totalWeight;
		
		// mark boundaries as solid
		if (ijk.x == 0 || ijk.y == 0 || ijk.z == 0 ||
			ijk.x == xcount - 1 || ijk.y == ycount - 1 || ijk.z == zcount - 1)
		{
			cellTypes[id] = SOLID_CELL;
		}
		
		// contribute to u grid
		if (ijk.x == 0 || ijk.x == 1 || ijk.x == xcount - 1)
		{
			uVels[id] = 0.0;
		}
		else
		{
			cellOffset = uParticleIndexOffsets[id]; // offset into index buffer
			numIndices = uParticleCounts[id];
			totalWeight = 0.0;
			
			for (int i = 0; i < numIndices; ++i)
			{
				pIdx = uParticleIndexBuffer[cellOffset + i];
				pos = vec3(particles[pIdx * 6],
				           particles[pIdx * 6 + 1],
						   particles[pIdx * 6 + 2]);
				vel = vec3(particles[pIdx * 6 + 3],
				           particles[pIdx * 6 + 4],
						   particles[pIdx * 6 + 5]);
				weight = computeWeightU(pos, cxyz);
				
				uVels[id] += weight * vel.x;
				totalWeight += weight;
			}
			
			totalWeight += float(totalWeight == 0.0) * FLT_EPSILON;
			uVels[id] /= totalWeight;
		}
		
		// contribute to v grid
		if (ijk.y == 0 || ijk.y == 1 || ijk.y == ycount - 1)
		{
			vVels[id] = 0.0;
		}
		else
		{
			cellOffset = vParticleIndexOffsets[id]; // offset into index buffer
			numIndices = vParticleCounts[id];
			totalWeight = 0.0;
			
			for (int i = 0; i < numIndices; ++i)
			{
				pIdx = vParticleIndexBuffer[cellOffset + i];
				pos = vec3(particles[pIdx * 6],
				           particles[pIdx * 6 + 1],
						   particles[pIdx * 6 + 2]);
				vel = vec3(particles[pIdx * 6 + 3],
				           particles[pIdx * 6 + 4],
						   particles[pIdx * 6 + 5]);
				weight = computeWeightV(pos, cxyz);
				
				vVels[id] += weight * vel.y;
				totalWeight += weight;
			}
			
			totalWeight += float(totalWeight == 0.0) * FLT_EPSILON;
			vVels[id] /= totalWeight;
		}
		
		// contribute to w grid
		if (ijk.z == 0 || ijk.z == 1 || ijk.z == zcount - 1)
		{
			wVels[id] = 0.0;
		}
		else
		{
			cellOffset = wParticleIndexOffsets[id]; // offset into index buffer
			numIndices = wParticleCounts[id];
			totalWeight = 0.0;
			
			for (int i = 0; i < numIndices; ++i)
			{
				pIdx = wParticleIndexBuffer[cellOffset + i];
				pos = vec3(particles[pIdx * 6],
				           particles[pIdx * 6 + 1],
						   particles[pIdx * 6 + 2]);
				vel = vec3(particles[pIdx * 6 + 3],
				           particles[pIdx * 6 + 4],
						   particles[pIdx * 6 + 5]);
				weight = computeWeightW(pos, cxyz);
				
				wVels[id] += weight * vel.z;
				totalWeight += weight;
			}
			
			totalWeight += float(totalWeight == 0.0) * FLT_EPSILON;
			wVels[id] /= totalWeight;
		}
	}
}