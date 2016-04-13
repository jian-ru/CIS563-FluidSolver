#version 450 core


layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer sparse_mv_block0
{
	float diagA[];
};

layout(std430, binding = 1) readonly buffer sparse_mv_block1
{
	float offDiagA[];
};

layout(std430, binding = 2) readonly buffer sparse_mv_block2
{
	int offsetSizeBuffer[];
};

layout(std430, binding = 3) readonly buffer sparse_mv_block3
{
	int colNums[];
};

layout(std430, binding = 4) readonly buffer sparse_mv_block4
{
	float in_x[];
};

layout(std430, binding = 5) buffer sparse_mv_block5
{
	float out_x[];
};


layout(std140, binding = 0) uniform cbMeta
{
	int size1D;
	float c1;
	float c2;
};


void main()
{
	int tID = int(gl_GlobalInvocationID.x);
	
	if (tID < size1D)
	{
		int offset = offsetSizeBuffer[tID * 2];
		int numNonzeroElems = offsetSizeBuffer[tID * 2 + 1];
	
		float aii = diagA[tID];
		float tmp_A[6];
		int tmp_C[6];
		for (int i = 0; i < numNonzeroElems; ++i)
		{
			tmp_A[i] = offDiagA[offset + i];
			tmp_C[i] = colNums[offset + i];
		}
	
		float sum = aii * in_x[tID];
		for (int i = 0; i < numNonzeroElems; ++i)
		{
			sum += tmp_A[i] * in_x[tmp_C[i]];
		}
		
		out_x[tID] = sum;
	}
}