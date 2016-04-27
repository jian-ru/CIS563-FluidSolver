#version 450 core

#define AIR_CELL 0
#define FLUID_CELL 1
#define SOLID_CELL 2


layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer bpsls_block0
{
	int cellTypes[];
};

layout(std430, binding = 4) writeonly buffer bpsls_block4
{
	float diagA[];
};

layout(std430, binding = 5) writeonly buffer bpsls_block5
{
	float one_diagA[];
};

layout(std430, binding = 6) writeonly buffer bpsls_block6
{
	float offDiagA[];
};

layout(std430, binding = 7) writeonly buffer bpsls_block7
{
	int offsetsSizesBuffer[];
};

layout(std430, binding = 8) writeonly buffer bpsls_block8
{
	int colNums[];
};

layout(std430, binding = 9) writeonly buffer bpsls_block9
{
	float b[];
};

layout(std430, binding = 10) readonly buffer bpsls_block10
{
	float uVels[];
};

layout(std430, binding = 11) readonly buffer bpsls_block11
{
	float vVels[];
};

layout(std430, binding = 12) readonly buffer bpsls_block12
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

layout(std430, binding = 3) readonly buffer bpsls_block3
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
	
	if (id < numCells && cellTypes[id] == FLUID_CELL)
	{
		int numNonsolidNeighbours = 0;
		int coeffCount = 0;
		int offset; // offset into offDiagA and colNums
		int rowNum, colNum;
		ivec3 ijk;
		int nidx;
		int ct;
		float uPlus1, vPlus1, wPlus1;
		
		get_ijk(id, ijk);
		queryEntry(id, rowNum);
		offset = rowNum * 6; // each row can have up to 6 -1's
		
		// right neighbour
		nidx = flatIdx(ijk.x + 1, ijk.y, ijk.z);
		uPlus1 = uVels[nidx];
		ct = cellTypes[nidx];
		numNonsolidNeighbours += int(ct != SOLID_CELL);
		
		if (ct == FLUID_CELL)
		{
			offDiagA[offset + coeffCount] = -1.0;
			queryEntry(nidx, colNum);
			colNums[offset + coeffCount] = colNum;
			++coeffCount;
		}
		
		// left neighbour
		nidx = flatIdx(ijk.x - 1, ijk.y, ijk.z);
		ct = cellTypes[nidx];
		numNonsolidNeighbours += int(ct != SOLID_CELL);
		
		if (ct == FLUID_CELL)
		{
			offDiagA[offset + coeffCount] = -1.0;
			queryEntry(nidx, colNum);
			colNums[offset + coeffCount] = colNum;
			++coeffCount;
		}
		
		// up neighbour
		nidx = flatIdx(ijk.x, ijk.y + 1, ijk.z);
		vPlus1 = vVels[nidx];
		ct = cellTypes[nidx];
		numNonsolidNeighbours += int(ct != SOLID_CELL);
		
		if (ct == FLUID_CELL)
		{
			offDiagA[offset + coeffCount] = -1.0;
			queryEntry(nidx, colNum);
			colNums[offset + coeffCount] = colNum;
			++coeffCount;
		}
		
		// down neighbour
		nidx = flatIdx(ijk.x, ijk.y - 1, ijk.z);
		ct = cellTypes[nidx];
		numNonsolidNeighbours += int(ct != SOLID_CELL);
		
		if (ct == FLUID_CELL)
		{
			offDiagA[offset + coeffCount] = -1.0;
			queryEntry(nidx, colNum);
			colNums[offset + coeffCount] = colNum;
			++coeffCount;
		}
		
		// back neighbour
		nidx = flatIdx(ijk.x, ijk.y, ijk.z + 1);
		wPlus1 = wVels[nidx];
		ct = cellTypes[nidx];
		numNonsolidNeighbours += int(ct != SOLID_CELL);
		
		if (ct == FLUID_CELL)
		{
			offDiagA[offset + coeffCount] = -1.0;
			queryEntry(nidx, colNum);
			colNums[offset + coeffCount] = colNum;
			++coeffCount;
		}
		
		// front neighbour
		nidx = flatIdx(ijk.x, ijk.y, ijk.z - 1);
		ct = cellTypes[nidx];
		numNonsolidNeighbours += int(ct != SOLID_CELL);
		
		if (ct == FLUID_CELL)
		{
			offDiagA[offset + coeffCount] = -1.0;
			queryEntry(nidx, colNum);
			colNums[offset + coeffCount] = colNum;
			++coeffCount;
		}
		
		diagA[rowNum] = float(numNonsolidNeighbours);
		one_diagA[rowNum] = 1.0 / float(numNonsolidNeighbours);
		offsetsSizesBuffer[rowNum * 2] = offset;
		offsetsSizesBuffer[rowNum * 2 + 1] = coeffCount;
		
		// calculate divergence
		float negativeDivergence = (uPlus1 - uVels[id]) +
								   (vPlus1 - vVels[id]) +
								   (wPlus1 - wVels[id]);
		negativeDivergence *= -cellSize / deltaTime;
		b[rowNum] = negativeDivergence;
	}
}