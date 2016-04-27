#version 450 core

#define FLT_EPSILON 1e-7
#define AIR_CELL 0
#define FLUID_CELL 1
#define SOLID_CELL 2


layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer egv_block0
{
	int cellTypes[];
};

layout(std430, binding = 1) buffer egv_block1
{
	float uVels[];
};

layout(std430, binding = 2) buffer egv_block2
{
	float vVels[];
};

layout(std430, binding = 3) buffer egv_block3
{
	float wVels[];
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
		const ivec3 dx[16] =
		{
			{-1, -1, -1}, {-1, -1,  0}, {-1, -1,  1},
			{-1,  0, -1}, {-1,  0,  1},
			{-1,  1, -1}, {-1,  1,  0}, {-1,  1,  1},
			{ 0, -1, -1}, { 0, -1,  0}, { 0, -1,  1},
			{ 0,  0, -1}, { 0,  0,  1},
			{ 0,  1, -1}, { 0,  1,  0}, { 0,  1,  1}
		};
		const ivec3 dy[16] =
		{
			{-1, -1, -1}, {-1, -1,  0}, {-1, -1,  1},
			{ 0, -1, -1}, { 0, -1,  1},
			{ 1, -1, -1}, { 1, -1,  0}, { 1, -1,  1},
			{-1,  0, -1}, {-1,  0,  0}, {-1,  0,  1},
			{ 0,  0, -1}, { 0,  0,  1},
			{ 1,  0, -1}, { 1,  0,  0}, { 1,  0,  1}
		};
		const ivec3 dz[16] =
		{
			{-1, -1, -1}, { 0, -1, -1}, { 1, -1, -1},
			{-1,  0, -1}, { 1,  0, -1},
			{-1,  1, -1}, { 0,  1, -1}, { 1,  1, -1},
			{-1, -1,  0}, { 0, -1,  0}, { 1, -1,  0},
			{-1,  0,  0}, { 1,  0,  0},
			{-1,  1,  0}, { 0,  1,  0}, { 1,  1,  0}
			
		};
		ivec3 ijk;
		int ct;
		
		get_ijk(id, ijk);
		ct = cellTypes[id];
		
		// extrapolate uVels
		if (ct != FLUID_CELL &&
			ijk.x != 0 && ijk.x != 1 && ijk.x != xcount - 1)
		{
			int numFluidCells = 0;
			int nidx;
			int nct;
			float newUVel = 0.0;
			ivec3 dijk;
			
			// left neighbour
			nidx = flatIdx(ijk.x - 1, ijk.y, ijk.z);
			nct = cellTypes[nidx];
			
			if (nct != FLUID_CELL)
			{
				for (int h = 0; h < 16; ++h)
				{
					dijk = ijk + dx[h];
					nidx = flatIdx(dijk.x, dijk.y, dijk.z);
					nct = cellTypes[nidx];
					
					newUVel += float(nct == FLUID_CELL) * uVels[nidx];
					numFluidCells += int(nct == FLUID_CELL);
				}
				
				float fnfc = float(numFluidCells) + float(numFluidCells == 0) * FLT_EPSILON;
				uVels[id] = newUVel / fnfc;
			}
		}
		
		// extrapolate vVels
		if (ct != FLUID_CELL &&
			ijk.y != 0 && ijk.y != 1 && ijk.y != ycount - 1)
		{
			int numFluidCells = 0;
			int nidx;
			int nct;
			float newVVel = 0.0;
			ivec3 dijk;
			
			// down neighbour
			nidx = flatIdx(ijk.x, ijk.y - 1, ijk.z);
			nct = cellTypes[nidx];
			
			if (nct != FLUID_CELL)
			{
				for (int h = 0; h < 16; ++h)
				{
					dijk = ijk + dy[h];
					nidx = flatIdx(dijk.x, dijk.y, dijk.z);
					nct = cellTypes[nidx];
					
					newVVel += float(nct == FLUID_CELL) * vVels[nidx];
					numFluidCells += int(nct == FLUID_CELL);
				}
				
				float fnfc = float(numFluidCells) + float(numFluidCells == 0) * FLT_EPSILON;
				vVels[id] = newVVel / fnfc;
			}
		}
		
		// extrapolate wVels
		if (ct != FLUID_CELL &&
			ijk.z != 0 && ijk.z != 1 && ijk.z != zcount - 1)
		{
			int numFluidCells = 0;
			int nidx;
			int nct;
			float newWVel = 0.0;
			ivec3 dijk;
			
			// front neighbour
			nidx = flatIdx(ijk.x, ijk.y, ijk.z - 1);
			nct = cellTypes[nidx];
			
			if (nct != FLUID_CELL)
			{
				for (int h = 0; h < 16; ++h)
				{
					dijk = ijk + dz[h];
					nidx = flatIdx(dijk.x, dijk.y, dijk.z);
					nct = cellTypes[nidx];
					
					newWVel += float(nct == FLUID_CELL) * wVels[nidx];
					numFluidCells += int(nct == FLUID_CELL);
				}
				
				float fnfc = float(numFluidCells) + float(numFluidCells == 0) * FLT_EPSILON;
				wVels[id] = newWVel / fnfc;
			}
		}
	}
}
