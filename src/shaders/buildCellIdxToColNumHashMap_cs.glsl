#version 450 core

#define FLUID_CELL 1


layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer bcitcnhm_block0
{
	int cellTypes[];
};

layout(std430, binding = 1) writeonly buffer bcitcnhm_block1
{
	int numFluidCells;
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


layout(std430, binding = 2) buffer bcitcnhm_block2
{
	int bucketLocks[];
};

layout(std430, binding = 3) buffer bcitcnhm_block3
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

int insertEntry(in int flatIdx, in int rowNum)
{
	ivec3 ijk;
	int bucketNum;
	int lock;
	
	get_ijk(flatIdx, ijk);
	bucketNum = getBucketNum(ijk);
	lock = atomicCompSwap(bucketLocks[bucketNum], 0, 1);
	
	if (lock == 0) // bucket wasn't locked
	{
		// try to find a free entry in this bucket
		int offset = bucketNum * bucketSize;
		
		for (int i = 0; i < bucketSize; ++i)
		{
			if (hashMapBuffer[offset + i].isOccupied == 0)
			{
				HashMapEntry entry;
				
				entry.isOccupied = 1;
				entry.cellFlatIdx = flatIdx;
				entry.cellRowNum = rowNum;
				entry.pNext = 0;
				
				hashMapBuffer[offset + i] = entry;
				
				bucketLocks[bucketNum] = 0; // unlock
				return 1;
			}
		}
		
		// bucket is full. Append to linked list
		const int numEntries = numBuckets * bucketSize;
		int lastEntryIdx = offset + bucketSize - 1;
		int tailIdx;
		int newEntryIdx;
		
		while(hashMapBuffer[lastEntryIdx].pNext != 0)
		{
			lastEntryIdx = hashMapBuffer[lastEntryIdx].pNext;
		}
		
		tailIdx = lastEntryIdx;
		// avoid last entry in the bucket
		newEntryIdx = lastEntryIdx + 1;
		newEntryIdx = (newEntryIdx +
					   int(newEntryIdx % bucketSize == bucketSize - 1)) % numEntries;
		
		while (true)
		{
			if (hashMapBuffer[newEntryIdx].isOccupied == 0)
			{
				int bucketNum2 = newEntryIdx / bucketSize;
				int lock2 = atomicCompSwap(bucketLocks[bucketNum2], 0, 1);
				
				if (lock2 == 0)
				{
					HashMapEntry entry;
				
					entry.isOccupied = 1;
					entry.cellFlatIdx = flatIdx;
					entry.cellRowNum = rowNum;
					entry.pNext = 0;
					hashMapBuffer[tailIdx].pNext = newEntryIdx;
				
					hashMapBuffer[newEntryIdx] = entry;
					
					bucketLocks[bucketNum2] = 0;
					bucketLocks[bucketNum] = 0;
					return 1;
				}
				
				// skip this bucket
				newEntryIdx = (bucketNum2 * bucketSize + bucketSize - 1) % numEntries;
			}
			
			// try to find next free entry
			++newEntryIdx;
			newEntryIdx = (newEntryIdx +
					       int(newEntryIdx % bucketSize == bucketSize - 1)) % numEntries;
		}
	}
	
	return 0;
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
		int rowNum = atomicAdd(numFluidCells, 1);
		
		while (1 != insertEntry(id, rowNum));
	}
}