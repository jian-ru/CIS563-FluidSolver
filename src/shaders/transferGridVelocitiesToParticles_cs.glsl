#version 450 core

#define FLT_EPSILON 1e-7
//#define FLT_EPSILON 0.0


layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) readonly buffer tgvtp_block0
{
	float uVels1[];
};

layout(std430, binding = 1) readonly buffer tgvtp_block1
{
	float vVels1[];
};

layout(std430, binding = 2) readonly buffer tgvtp_block2
{
	float wVels1[];
};

layout(std430, binding = 3) readonly buffer tgvtp_block3
{
	float uVels2[];
};

layout(std430, binding = 4) readonly buffer tgvtp_block4
{
	float vVels2[];
};

layout(std430, binding = 5) readonly buffer tgvtp_block5
{
	float wVels2[];
};

layout(std430, binding = 6) buffer tgvtp_block6
{
	float particles[];
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

void getu_uvw(in vec3 pos, out ivec3 ijk, out vec3 uvw)
{
	const float delta = 0.5 * cellSize;
	vec3 cxyz;
	vec3 offPos;
	
	getu_ijk(pos, ijk);
	cxyz = vec3(ijk) * cellSize;
	offPos = vec3(pos.x, pos.y - delta, pos.z - delta);
	
	uvw = clamp((offPos - cxyz) / cellSize, 0.0, 1.0);
}

void getv_uvw(in vec3 pos, out ivec3 ijk, out vec3 uvw)
{
	const float delta = 0.5 * cellSize;
	vec3 cxyz;
	vec3 offPos;
	
	getv_ijk(pos, ijk);
	cxyz = vec3(ijk) * cellSize;
	offPos = vec3(pos.x - delta, pos.y, pos.z - delta);
	
	uvw = clamp((offPos - cxyz) / cellSize, 0.0, 1.0);
}

void getw_uvw(in vec3 pos, out ivec3 ijk, out vec3 uvw)
{
	const float delta = 0.5 * cellSize;
	vec3 cxyz;
	vec3 offPos;
	
	getw_ijk(pos, ijk);
	cxyz = vec3(ijk) * cellSize;
	offPos = vec3(pos.x - delta, pos.y - delta, pos.z);
	
	uvw = clamp((offPos - cxyz) / cellSize, 0.0, 1.0);
}

float triLerpU(in vec3 pos, in vec3 oldVel)
{
	vec3 uvw;
	ivec3 ijk;
	int cellIdx;
	
	getu_uvw(pos, ijk, uvw);
	cellIdx = flatIdx(ijk.x, ijk.y, ijk.z);
	float u000_1 = uVels1[cellIdx];
	float u000_2 = uVels2[cellIdx];
	cellIdx = flatIdx(ijk.x + 1, ijk.y, ijk.z);
	float u100_1 = uVels1[cellIdx];
	float u100_2 = uVels2[cellIdx];
	cellIdx = flatIdx(ijk.x, ijk.y + 1, ijk.z);
	float u010_1 = uVels1[cellIdx];
	float u010_2 = uVels2[cellIdx];
	cellIdx = flatIdx(ijk.x + 1, ijk.y + 1, ijk.z);
	float u110_1 = uVels1[cellIdx];
	float u110_2 = uVels2[cellIdx];
	cellIdx = flatIdx(ijk.x, ijk.y, ijk.z + 1);
	float u001_1 = uVels1[cellIdx];
	float u001_2 = uVels2[cellIdx];
	cellIdx = flatIdx(ijk.x + 1, ijk.y, ijk.z + 1);
	float u101_1 = uVels1[cellIdx];
	float u101_2 = uVels2[cellIdx];
	cellIdx = flatIdx(ijk.x, ijk.y + 1, ijk.z + 1);
	float u011_1 = uVels1[cellIdx];
	float u011_2 = uVels2[cellIdx];
	cellIdx = flatIdx(ijk.x + 1, ijk.y + 1, ijk.z + 1);
	float u111_1 = uVels1[cellIdx];
	float u111_2 = uVels2[cellIdx];
		
	float pic_u00 = (1.0 - uvw.x) * u000_1 + uvw.x * u100_1;
	float pic_u01 = (1.0 - uvw.x) * u010_1 + uvw.x * u110_1;
	float pic_u02 = (1.0 - uvw.x) * u001_1 + uvw.x * u101_1;
	float pic_u03 = (1.0 - uvw.x) * u011_1 + uvw.x * u111_1;
	float pic_u10 = (1.0 - uvw.y) * pic_u00 + uvw.y * pic_u01;
	float pic_u11 = (1.0 - uvw.y) * pic_u02 + uvw.y * pic_u03;
	float pic_u20 = (1.0 - uvw.z) * pic_u10 + uvw.z * pic_u11;
		
	float flip_u00 = (1.0 - uvw.x) * (u000_1 - u000_2) + uvw.x * (u100_1 - u100_2);
	float flip_u01 = (1.0 - uvw.x) * (u010_1 - u010_2) + uvw.x * (u110_1 - u110_2);
	float flip_u02 = (1.0 - uvw.x) * (u001_1 - u001_2) + uvw.x * (u101_1 - u101_2);
	float flip_u03 = (1.0 - uvw.x) * (u011_1 - u011_2) + uvw.x * (u111_1 - u111_2);
	float flip_u10 = (1.0 - uvw.y) * flip_u00 + uvw.y * flip_u01;
	float flip_u11 = (1.0 - uvw.y) * flip_u02 + uvw.y * flip_u03;
	float flip_u20 = (1.0 - uvw.z) * flip_u10 + uvw.z * flip_u11;
	flip_u20 = oldVel.x + flip_u20;
		
	return 0.95 * flip_u20 + 0.05 * pic_u20;
}

float triLerpV(in vec3 pos, in vec3 oldVel)
{
	vec3 uvw;
	ivec3 ijk;
	int cellIdx;
	
	getv_uvw(pos, ijk, uvw);
	cellIdx = flatIdx(ijk.x, ijk.y, ijk.z);
	float v000_1 = vVels1[cellIdx];
	float v000_2 = vVels2[cellIdx];
	cellIdx = flatIdx(ijk.x + 1, ijk.y, ijk.z);
	float v100_1 = vVels1[cellIdx];
	float v100_2 = vVels2[cellIdx];
	cellIdx = flatIdx(ijk.x, ijk.y + 1, ijk.z);
	float v010_1 = vVels1[cellIdx];
	float v010_2 = vVels2[cellIdx];
	cellIdx = flatIdx(ijk.x + 1, ijk.y + 1, ijk.z);
	float v110_1 = vVels1[cellIdx];
	float v110_2 = vVels2[cellIdx];
	cellIdx = flatIdx(ijk.x, ijk.y, ijk.z + 1);
	float v001_1 = vVels1[cellIdx];
	float v001_2 = vVels2[cellIdx];
	cellIdx = flatIdx(ijk.x + 1, ijk.y, ijk.z + 1);
	float v101_1 = vVels1[cellIdx];
	float v101_2 = vVels2[cellIdx];
	cellIdx = flatIdx(ijk.x, ijk.y + 1, ijk.z + 1);
	float v011_1 = vVels1[cellIdx];
	float v011_2 = vVels2[cellIdx];
	cellIdx = flatIdx(ijk.x + 1, ijk.y + 1, ijk.z + 1);
	float v111_1 = vVels1[cellIdx];
	float v111_2 = vVels2[cellIdx];
		
	float pic_v00 = (1.0 - uvw.x) * v000_1 + uvw.x * v100_1;
	float pic_v01 = (1.0 - uvw.x) * v010_1 + uvw.x * v110_1;
	float pic_v02 = (1.0 - uvw.x) * v001_1 + uvw.x * v101_1;
	float pic_v03 = (1.0 - uvw.x) * v011_1 + uvw.x * v111_1;
	float pic_v10 = (1.0 - uvw.y) * pic_v00 + uvw.y * pic_v01;
	float pic_v11 = (1.0 - uvw.y) * pic_v02 + uvw.y * pic_v03;
	float pic_v20 = (1.0 - uvw.z) * pic_v10 + uvw.z * pic_v11;
		
	float flip_v00 = (1.0 - uvw.x) * (v000_1 - v000_2) + uvw.x * (v100_1 - v100_2);
	float flip_v01 = (1.0 - uvw.x) * (v010_1 - v010_2) + uvw.x * (v110_1 - v110_2);
	float flip_v02 = (1.0 - uvw.x) * (v001_1 - v001_2) + uvw.x * (v101_1 - v101_2);
	float flip_v03 = (1.0 - uvw.x) * (v011_1 - v011_2) + uvw.x * (v111_1 - v111_2);
	float flip_v10 = (1.0 - uvw.y) * flip_v00 + uvw.y * flip_v01;
	float flip_v11 = (1.0 - uvw.y) * flip_v02 + uvw.y * flip_v03;
	float flip_v20 = (1.0 - uvw.z) * flip_v10 + uvw.z * flip_v11;
	flip_v20 = oldVel.y + flip_v20;
		
	return 0.95 * flip_v20 + 0.05 * pic_v20;
}

float triLerpW(in vec3 pos, in vec3 oldVel)
{
	vec3 uvw;
	ivec3 ijk;
	int cellIdx;
	
	getw_uvw(pos, ijk, uvw);
	cellIdx = flatIdx(ijk.x, ijk.y, ijk.z);
	float w000_1 = wVels1[cellIdx];
	float w000_2 = wVels2[cellIdx];
	cellIdx = flatIdx(ijk.x + 1, ijk.y, ijk.z);
	float w100_1 = wVels1[cellIdx];
	float w100_2 = wVels2[cellIdx];
	cellIdx = flatIdx(ijk.x, ijk.y + 1, ijk.z);
	float w010_1 = wVels1[cellIdx];
	float w010_2 = wVels2[cellIdx];
	cellIdx = flatIdx(ijk.x + 1, ijk.y + 1, ijk.z);
	float w110_1 = wVels1[cellIdx];
	float w110_2 = wVels2[cellIdx];
	cellIdx = flatIdx(ijk.x, ijk.y, ijk.z + 1);
	float w001_1 = wVels1[cellIdx];
	float w001_2 = wVels2[cellIdx];
	cellIdx = flatIdx(ijk.x + 1, ijk.y, ijk.z + 1);
	float w101_1 = wVels1[cellIdx];
	float w101_2 = wVels2[cellIdx];
	cellIdx = flatIdx(ijk.x, ijk.y + 1, ijk.z + 1);
	float w011_1 = wVels1[cellIdx];
	float w011_2 = wVels2[cellIdx];
	cellIdx = flatIdx(ijk.x + 1, ijk.y + 1, ijk.z + 1);
	float w111_1 = wVels1[cellIdx];
	float w111_2 = wVels2[cellIdx];
		
	float pic_w00 = (1.0 - uvw.x) * w000_1 + uvw.x * w100_1;
	float pic_w01 = (1.0 - uvw.x) * w010_1 + uvw.x * w110_1;
	float pic_w02 = (1.0 - uvw.x) * w001_1 + uvw.x * w101_1;
	float pic_w03 = (1.0 - uvw.x) * w011_1 + uvw.x * w111_1;
	float pic_w10 = (1.0 - uvw.y) * pic_w00 + uvw.y * pic_w01;
	float pic_w11 = (1.0 - uvw.y) * pic_w02 + uvw.y * pic_w03;
	float pic_w20 = (1.0 - uvw.z) * pic_w10 + uvw.z * pic_w11;
		
	float flip_w00 = (1.0 - uvw.x) * (w000_1 - w000_2) + uvw.x * (w100_1 - w100_2);
	float flip_w01 = (1.0 - uvw.x) * (w010_1 - w010_2) + uvw.x * (w110_1 - w110_2);
	float flip_w02 = (1.0 - uvw.x) * (w001_1 - w001_2) + uvw.x * (w101_1 - w101_2);
	float flip_w03 = (1.0 - uvw.x) * (w011_1 - w011_2) + uvw.x * (w111_1 - w111_2);
	float flip_w10 = (1.0 - uvw.y) * flip_w00 + uvw.y * flip_w01;
	float flip_w11 = (1.0 - uvw.y) * flip_w02 + uvw.y * flip_w03;
	float flip_w20 = (1.0 - uvw.z) * flip_w10 + uvw.z * flip_w11;
	flip_w20 = oldVel.z + flip_w20;
		
	return 0.95 * flip_w20 + 0.05 * pic_w20;
}


void main()
{
	int id = int(gl_GlobalInvocationID.x);
	
	if (id < numParticles)
	{
		vec3 pos = vec3(particles[id * 6],
		                particles[id * 6 + 1],
						particles[id * 6 + 2]);
		vec3 oldVel = vec3(particles[id * 6 + 3],
		                   particles[id * 6 + 4],
						   particles[id * 6 + 5]);
		vec3 newVel;
		
		newVel.x = triLerpU(pos, oldVel);
		newVel.y = triLerpV(pos, oldVel);
		newVel.z = triLerpW(pos, oldVel);
		
		particles[id * 6 + 3] = newVel.x;
		particles[id * 6 + 4] = newVel.y;
		particles[id * 6 + 5] = newVel.z;
	}
}