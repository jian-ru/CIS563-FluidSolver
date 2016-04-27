#version 450 core


// Work group size
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


layout(std430, binding = 0) buffer ap_block0
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


void main()
{
	int gid = int(gl_GlobalInvocationID.x);
	
	if (gid < numParticles)
	{
		vec3 position = vec3(particles[gid*6], particles[gid*6+1], particles[gid*6+2]);
		vec3 velocity = vec3(particles[gid*6+3], particles[gid*6+4], particles[gid*6+5]);
	
		vec3 next_pos = position + deltaTime * velocity;
	
		float bounds[6];
		bounds[0] = cellSize; bounds[1] = (xcount - 1) * cellSize;
		bounds[2] = cellSize; bounds[3] = (ycount - 1) * cellSize;
		bounds[4] = cellSize; bounds[5] = (zcount - 1) * cellSize;
	
		bool ncnx = next_pos.x > bounds[0];
		bool ncpx = next_pos.x < bounds[1];
		bool ncny = next_pos.y > bounds[2];
		bool ncpy = next_pos.y < bounds[3];
		bool ncnz = next_pos.z > bounds[4];
		bool ncpz = next_pos.z < bounds[5];
	
		position.x = next_pos.x * float(ncnx && ncpx) + bounds[0] * float(!ncnx) + bounds[1] * float(!ncpx);
		position.y = next_pos.y * float(ncny && ncpy) + bounds[2] * float(!ncny) + bounds[3] * float(!ncpy);
		position.z = next_pos.z * float(ncnz && ncpz) + bounds[4] * float(!ncnz) + bounds[5] * float(!ncpz);
	
		float df = 1.0; // damping factor
	
		velocity.x = velocity.x * float(ncnx && ncpx) - df * velocity.x * float(!ncnx || !ncpx);
		velocity.y = velocity.y * float(ncny && ncpy) - df * velocity.y * float(!ncny || !ncpy);
		velocity.z = velocity.z * float(ncnz && ncpz) - df * velocity.z * float(!ncnz || !ncpz);
	
		particles[gid*6] = position.x;
		particles[gid*6+1] = position.y;
		particles[gid*6+2] = position.z;
		particles[gid*6+3] = velocity.x;
		particles[gid*6+4] = velocity.y;
		particles[gid*6+5] = velocity.z;
	}
}