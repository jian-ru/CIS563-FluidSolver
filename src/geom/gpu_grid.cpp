#include "gpu_gridd.hpp"


void FS_GPU_MACGRID::seedParticles()
{
	const int numSamples2D = numSamples1D * numSamples1D;
	const int numSamples3D = numSamples2D * numSamples1D;
	const int xExtent = (xcount - 2) / 3;
	const int yExtent = (ycount - 2);
	const int zExtent = (zcount - 2);
	const int xStart = 1; // 0 is boundary. Do not spawn into boundary cells
	const int yStart = 1;
	const int zStart = 1;

	numParticles = xExtent * yExtent * zExtent * numSamples3D;
	std::vector<FS_Particle> cpuParticles(numParticles);

	memset(&cpuParticles[0], 0, numParticles * sizeof(FS_Particle));

	tbb::atomic<int> counter = 0;
	tbb::parallel_for(
		tbb::blocked_range3d<int>(xStart, xStart + xExtent,
			yStart, yStart + yExtent,
			zStart, zStart + zExtent),
		[&](const tbb::blocked_range3d<int> &r)
	{
		for (int i = r.pages().begin(); i < r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j < r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k < r.cols().end(); ++k)
				{
					int offset = counter.fetch_and_add(numSamples3D);

					for (int h = 0; h < numSamples3D; ++h)
					{
						FS_Particle p;
						int x = h / numSamples2D;
						int y = h / numSamples1D % numSamples1D;
						int z = h % numSamples1D;
						float jitterSize = cellSize / static_cast<float>(numSamples1D);
						float xpos = (x + rng(generator)) * jitterSize;
						float ypos = (y + rng(generator)) * jitterSize;
						float zpos = (z + rng(generator)) * jitterSize;

						xpos += i * cellSize;
						ypos += j * cellSize;
						zpos += k * cellSize;

						p.position = glm::vec3(xpos, ypos, zpos);
						p.velocity = glm::vec3(0.f);
						cpuParticles[offset + h] = p;
					}
				}
			}
		}
	} // lambda end
	); // parallel_for end

	glGenBuffers(1, &particles);
	glBindBuffer(GL_ARRAY_BUFFER, particles);
	glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(FS_Particle), cpuParticles.data(), GL_DYNAMIC_COPY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void FS_GPU_MACGRID::genGPUBuffers()
{
	glGenBuffers(1, &uVels1);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, uVels1);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	glGenBuffers(1, &uVels2);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, uVels2);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &vVels1);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, vVels1);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	glGenBuffers(1, &vVels2);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, vVels2);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &wVels1);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, wVels1);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	glGenBuffers(1, &wVels2);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, wVels2);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &pressures);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, pressures);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &cellTypes);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellTypes);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
}


void FS_GPU_MACGRID::transferParticleVelocitiesToGrid()
{

}