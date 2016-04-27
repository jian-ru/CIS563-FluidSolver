#version 450 core

#define AIR_CELL 0
#define FLUID_CELL 1
#define SOLID_CELL 2


layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer tpbtg_block0
{
	float x[];
};

layout(std430, binding = 1) readonly buffer tpbtg_block1
{
	int cellTypes[];
};

layout(std430, binding = 3) writeonly buffer tpbtg_block3
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


// ------------------ Hash map implementation
#define P1 73856093u
#define P2 19349669u
#define P3 83492791u


struct HashMapEntry
{
	int isOccupied;
	int cellFlatIdx;
	int cellRowNum; // == colNum
	int pNext;
};

layout(std430, binding = 2) readonly buffer tpbtg_block2
{
	HashMapEntry hashMapBuffer[];
};


layout(std140, binding = 2) uniform cbHashMap
{
	int numBuckets;
	int bucketSize;
};


int getBucketNum(in ivec3 ijk)
{
	uint i = uint(ijk.x);
	uint j = uint(ijk.y);
	uint k = uint(ijk.z);
	
	return int(((i * P1) ^ (j * P2) ^ (k * P3)) % uint(numBuckets));
}

int queryEntry(in int flatIdx, out int rowNum)
{
	ivec3 ijk;
	int bucketNum;
	int bucketStart;
	
	get_ijk(flatIdx, ijk);
	bucketNum = getBucketNum(ijk);
	bucketStart = bucketNum * bucketSize;
	
	for (int i = 0; i < bucketSize; ++i)
	{
		HashMapEntry entry = hashMapBuffer[bucketStart + i];
		
		if (entry.isOccupied == 1 && entry.cellFlatIdx == flatIdx)
		{
			rowNum = entry.cellRowNum;
			return 1; // found
		}
	}
	
	int lastEntryIdx = bucketStart + bucketSize - 1;
	HashMapEntry entry = hashMapBuffer[lastEntryIdx];
	
	if (entry.isOccupied == 1)
	{
		while (entry.pNext != 0)
		{
			// get next entry in the linked list
			lastEntryIdx = entry.pNext;
			entry = hashMapBuffer[lastEntryIdx];
			
			if (entry.cellFlatIdx == flatIdx)
			{
				rowNum = entry.cellRowNum;
				return 1; // found
			}
		}
	}
	
	return 0; // not found
}

// ------------------ Hash map implementation


void main()
{
	int id = int(gl_GlobalInvocationID.x);
	
	if (id < numCells)
	{
		int ct = cellTypes[id];
		int rowNum;
		
		if (ct == FLUID_CELL)
		{
			queryEntry(id, rowNum);
			pressures[id] = x[rowNum];
		}
	}
}