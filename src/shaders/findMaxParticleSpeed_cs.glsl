#version 450 core


layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;


struct Particle
{
	vec3 position;
	vec3 velocity;
};

layout(std430, binding = 0) readonly buffer fmps_block0
{
	Particle particles[];
};

layout(std430, binding = 1) readonly buffer fmps_block1
{
	float inSpeeds[];
};

layout(std430, binding = 2) writeonly buffer fmps_block2
{
	float outSpeeds[];
};

layout(std430, binding = 3) buffer fmps_block3
{
	int numGlobalIterations;
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


shared float shared_data[gl_WorkGroupSize.x * 2];


void main()
{
	uint gid = gl_GlobalInvocationID.x;
	uint id = gl_LocalInvocationID.x;
	uint rd_id, wr_id;
	uint mask;
	int curNumGlobalIterations = numGlobalIterations;
	
	const int steps = int(log2(gl_WorkGroupSize.x * 2));
	
	if (numGlobalIterations == 0)
	{
		shared_data[id * 2] = 0.0;
		shared_data[id * 2 + 1] = 0.0;
		
		if (gid * 2 < numParticles)
		{
			shared_data[id * 2] = length(particles[gid * 2].velocity);
		}
		if (gid * 2 + 1 < numParticles)
		{
			shared_data[id * 2 + 1] = length(particles[gid * 2 + 1].velocity);
		}
	}
	else
	{
		shared_data[id * 2] = inSpeeds[gid * 2];
		shared_data[id * 2 + 1] = inSpeeds[gid * 2 + 1];
	}
	barrier();
	memoryBarrierShared();
	
	uint numThreadsInSubGroup = 1;
	
	for (uint i = 0; i < steps; ++i)
	{
		if ((id + 1) % numThreadsInSubGroup == 0) // last thread in subgroup
		{
			mask = (1 << i) - 1;
			rd_id = ((id >> i) << (i + 1)) + mask;
			wr_id = rd_id + 1 + (id & mask);
		
			shared_data[wr_id] = max(shared_data[rd_id], shared_data[wr_id]);
		}
		
		numThreadsInSubGroup << 1;
		
		barrier();
		memoryBarrierShared();
	}
	
	if (id == gl_WorkGroupSize.x - 1)
	{
		outSpeeds[gl_WorkGroupID.x] = shared_data[id * 2 + 1];
		numGlobalIterations = curNumGlobalIterations + 1;
	}
}