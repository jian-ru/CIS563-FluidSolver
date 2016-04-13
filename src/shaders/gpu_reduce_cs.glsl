#version 450 core


layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer reduce_block0
{
	float in_v[2048];
};

layout(std430, binding = 1) buffer reduce_block1
{
	float out_v[2048];
};


shared float shared_data[2048];


void main()
{
	int id = int(gl_LocalInvocationID.x);
	int rd_id, wr_id;
	int mask;
	
	const int steps = int(log2(2048));
	
	shared_data[id * 2] = in_v[id * 2];
	shared_data[id * 2 + 1] = in_v[id * 2 + 1];
	
	barrier();
	memoryBarrierShared();
	
	for (int i = 0; i < steps; ++i)
	{
		mask = (1 << i) - 1;
		rd_id = ((id >> i) << (i + 1)) + mask;
		wr_id = rd_id + 1 + (id & mask);
		
		shared_data[wr_id] += shared_data[rd_id];
		
		barrier();
		memoryBarrierShared();
	}
	
	out_v[id * 2] = shared_data[id * 2];
	out_v[id * 2 + 1] = shared_data[id * 2 + 1];
}