#version 450 core


layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer reduce_block0
{
	float in_v[];
};

layout(std430, binding = 1) buffer reduce_block1
{
	float out_v[];
};


shared float shared_data[gl_WorkGroupSize.x * 2];


void main()
{
	uint gid = gl_GlobalInvocationID.x;
	uint id = gl_LocalInvocationID.x;
	uint rd_id, wr_id;
	uint mask;
	
	const int steps = int(log2(gl_WorkGroupSize.x * 2));
	
	shared_data[id * 2] = in_v[gid * 2];
	shared_data[id * 2 + 1] = in_v[gid * 2 + 1];
	
	barrier();
	memoryBarrierShared();
	
	for (uint i = 0; i < steps; ++i)
	{
		mask = (1 << i) - 1;
		rd_id = ((id >> i) << (i + 1)) + mask;
		wr_id = rd_id + 1 + (id & mask);
		
		shared_data[wr_id] += shared_data[rd_id];
		
		barrier();
		memoryBarrierShared();
	}
	
	if (id == gl_WorkGroupSize.x - 1)
	{
		out_v[gl_WorkGroupID.x] = shared_data[id * 2 + 1];
	}
}