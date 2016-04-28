#include "gpu_gridd.hpp"
#include <iostream>
#include <unordered_map>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/IterativeLinearSolvers"


class GPUOPERATIONS
{
public:
	GLuint sparseMvProgram;
	GLuint vaddProgram;
	GLuint vmulProgram;
	GLuint vdivProgram;
	GLuint reduceProgram;
	GLuint exclusivePrefixSumProgram;
	GLuint segmentAddProgram;
	GLuint zeroBufferProgram;

	int size1D; // number of fluid cells
	int numCells;
	const int segmentSize = 2048;

	GLuint tmp, tmp2;
	GLuint v1v2;

	GLuint cbMetaName;
	char cbMeta[12];

	GPUOPERATIONS(int nc)
		: numCells(nc)
	{
		auto createCSProgram = [](const char *fn) -> GLuint
		{
			std::string csFileName(SHADERS_DIR);
			csFileName += "/";
			csFileName += fn;
			GLuint csn = loadShader(csFileName.c_str(), GL_COMPUTE_SHADER);

			GLuint program = glCreateProgram();
			glAttachShader(program, csn);
			glLinkProgram(program);

			glDeleteShader(csn);

			// Check link status
			GLint result;
			int infoLogLength;
			glGetProgramiv(program, GL_LINK_STATUS, &result);
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
			if (infoLogLength > 0) {
				std::vector<char> msg(infoLogLength + 1);
				glGetProgramInfoLog(program, infoLogLength, NULL, &msg[0]);
				std::cout << msg.data() << '\n';
			}
			if (!result)
			{
				exit(EXIT_FAILURE);
			}

			return program;
		};

		sparseMvProgram = createCSProgram("gpu_sparse_mv_cs.glsl");
		vaddProgram = createCSProgram("gpu_vadd_cs.glsl");
		vmulProgram = createCSProgram("gpu_vmul_cs.glsl");
		vdivProgram = createCSProgram("gpu_vdiv_cs.glsl");
		reduceProgram = createCSProgram("gpu_reduce_cs_v02.glsl");
		zeroBufferProgram = createCSProgram("gpu_zerobuffer_cs.glsl");
		exclusivePrefixSumProgram = createCSProgram("gpu_exprefixsum_cs.glsl");
		segmentAddProgram = createCSProgram("gpu_segment_add_cs.glsl");

		memset(cbMeta, 0, 12);
		glGenBuffers(1, &cbMetaName);
		glBindBuffer(GL_UNIFORM_BUFFER, cbMetaName);
		glBufferData(GL_UNIFORM_BUFFER, 12, cbMeta, GL_STREAM_DRAW);
		glBindBuffer(GL_UNIFORM_BUFFER, 0); // unbind

		int buffSize = (numCells + segmentSize - 1) / segmentSize * segmentSize * sizeof(float);

		glGenBuffers(1, &tmp);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, tmp);
		glBufferData(GL_SHADER_STORAGE_BUFFER, buffSize, NULL, GL_DYNAMIC_COPY);

		glGenBuffers(1, &tmp2);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, tmp2);
		glBufferData(GL_SHADER_STORAGE_BUFFER, buffSize, NULL, GL_DYNAMIC_COPY);

		glGenBuffers(1, &v1v2);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, v1v2);
		glBufferData(GL_SHADER_STORAGE_BUFFER, buffSize, NULL, GL_DYNAMIC_COPY);
	}

	~GPUOPERATIONS()
	{
		glDeleteProgram(sparseMvProgram);
		glDeleteProgram(vaddProgram);
		glDeleteProgram(vmulProgram);
		glDeleteProgram(vdivProgram);
		glDeleteProgram(reduceProgram);
		glDeleteProgram(exclusivePrefixSumProgram);
		glDeleteProgram(zeroBufferProgram);

		glDeleteBuffers(1, &cbMetaName);
		glDeleteBuffers(1, &tmp);
		glDeleteBuffers(1, &tmp2);
		glDeleteBuffers(1, &v1v2);
	}

	void setSize1D(int s1D)
	{
		size1D = s1D;
		glBindBuffer(GL_UNIFORM_BUFFER, cbMetaName);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, 4, &s1D);
	}

	void bindConstantBuffer()
	{
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, cbMetaName);
	}

	void zeroBuffer(GLuint buff, int numElements)
	{
		glUseProgram(zeroBufferProgram);
		glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buff, 0, numElements * 4); // 4 bytes per element
		glUniform1i(0, numElements);

		const int localGroupSize = 256;
		int numGroups = (numElements + localGroupSize - 1) / localGroupSize;

		glDispatchCompute(numGroups, 1, 1);

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		glFinish();
	}

	void sparseMv(GLuint diagA, GLuint offDiagA, GLuint offsetSizeBuffer, GLuint colNums, GLuint in_x, GLuint out_x)
	{
		glUseProgram(sparseMvProgram);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, diagA);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, offDiagA);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, offsetSizeBuffer);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, colNums);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, in_x);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, out_x);

		const int localGroupSize = 256;
		int numGroups = (size1D + localGroupSize - 1) / localGroupSize;

		glDispatchCompute(numGroups, 1, 1);

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		glFinish();
	}

	void vadd(float c1, GLuint v1, float c2, GLuint v2, GLuint out_v)
	{
		glUseProgram(vaddProgram);
		glBindBuffer(GL_UNIFORM_BUFFER, cbMetaName);
		float *c1c2 = (float *)glMapBufferRange(GL_UNIFORM_BUFFER, 4, 8, GL_MAP_WRITE_BIT);
		c1c2[0] = c1;
		c1c2[1] = c2;
		glUnmapBuffer(GL_UNIFORM_BUFFER);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, v1);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, v2);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, out_v);

		const int localGroupSize = 256;
		int numGroups = (size1D + localGroupSize - 1) / localGroupSize;

		glDispatchCompute(numGroups, 1, 1);

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		glFinish();
	}

	void vmul(GLuint v1, GLuint v2, GLuint out_v)
	{
		glUseProgram(vmulProgram);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, v1);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, v2);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, out_v);

		const int localGroupSize = 256;
		int numGroups = (size1D + localGroupSize - 1) / localGroupSize;

		glDispatchCompute(numGroups, 1, 1);

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		glFinish();
	}

	void vdiv(GLuint v1, GLuint v2, GLuint out_v)
	{
		glUseProgram(vdivProgram);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, v1);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, v2);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, out_v);

		const int localGroupSize = 256;
		int numGroups = (size1D + localGroupSize - 1) / localGroupSize;

		glDispatchCompute(numGroups, 1, 1);

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		glFinish();
	}

	void exclusivePrefixSum(GLuint in_v, GLuint out_v, int numElements)
	{
		int segments = (numElements + segmentSize - 1) / segmentSize;
		int tmpSize = segments * segmentSize;

		zeroBuffer(out_v, tmpSize);

		glUseProgram(exclusivePrefixSumProgram);

		glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, in_v, 0, tmpSize * sizeof(int));
		glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, out_v, 0, tmpSize * sizeof(int));

		glDispatchCompute(segments, 1, 1);

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		glFinish();

		// up to this point, we have the prefix sum of each segment
		glUseProgram(segmentAddProgram);

		for (int i = 1; i < segments; ++i)
		{
			glUniform1i(0, i);
			glDispatchCompute(1, 1, 1);

			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			glFinish();
		}
		// now we have prefix sum of in_v
	}

	void reduce(GLuint in_v, float *p_result)
	{
		int segments = (size1D + segmentSize - 1) / segmentSize;
		int pass = 0;

		while (true)
		{
			int tmpSize = (segments + segmentSize - 1) / segmentSize * segmentSize;
			
			zeroBuffer(tmp, tmpSize);
			glUseProgram(reduceProgram);

			if (pass == 0)
			{
				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, in_v, 0, segments * segmentSize * sizeof(float));
			}
			else
			{
				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, tmp2, 0, segments * segmentSize * sizeof(float));
			}

			glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, tmp, 0, tmpSize * sizeof(float));

			glDispatchCompute(segments, 1, 1);

			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			glFinish();

			if (segments > 1)
			{
				int tmptmp = tmp2;
				tmp2 = tmp;
				tmp = tmptmp;
				++pass;
				segments = (segments + segmentSize - 1) / segmentSize;
			}
			else
			{
				glBindBuffer(GL_COPY_READ_BUFFER, tmp);
				float *result = (float *)glMapBufferRange(GL_COPY_READ_BUFFER, 0, sizeof(float), GL_MAP_READ_BIT);
				*p_result = *result;
				glUnmapBuffer(GL_COPY_READ_BUFFER);
				break;
			}
		}
	}

	void dot(GLuint v1, GLuint v2, float *result)
	{
		int size = (size1D + segmentSize - 1) / segmentSize * segmentSize;
		zeroBuffer(v1v2, size);

		vmul(v1, v2, v1v2);
		reduce(v1v2, result);
	}
};


FS_GPU_MACGRID::FS_GPU_MACGRID(int xc, int yc, int zc, float cs, int ns1d, float maxDT, float minDT, float kcfl)
	: numSamples1D(ns1d), maxDeltaTime(maxDT), minDeltaTime(minDT), kCFL(kcfl)
{
	xcount = xc + 2;
	ycount = yc + 2;
	zcount = zc + 2;
	cellSize = cs;
	numCells = xcount * ycount * zcount;
	numBuckets = numCells * 8;
	cpuMaxParticleSpeed = -1.f;

	rng = std::uniform_real_distribution<float>(0.f, 1.f);
	generator = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	gpuops = std::make_shared<GPUOPERATIONS>(numCells);

	seedParticles();
	genGPUBuffers();
	genShaderPrograms();

	setCbMeta();
	setCbHashMap();
}


FS_GPU_MACGRID::~FS_GPU_MACGRID()
{
	glDeleteBuffers(1, &cbMeta);

	glDeleteBuffers(1, &particles);
	glDeleteBuffers(1, &uVels1);
	glDeleteBuffers(1, &uVels2);
	glDeleteBuffers(1, &vVels1);
	glDeleteBuffers(1, &vVels2);
	glDeleteBuffers(1, &wVels1);
	glDeleteBuffers(1, &wVels2);
	glDeleteBuffers(1, &pressures);
	glDeleteBuffers(1, &cellTypes);

	glDeleteBuffers(1, &uParticleCounts);
	glDeleteBuffers(1, &vParticleCounts);
	glDeleteBuffers(1, &wParticleCounts);
	glDeleteBuffers(1, &uParticleIndexOffsets);
	glDeleteBuffers(1, &vParticleIndexOffsets);
	glDeleteBuffers(1, &wParticleIndexOffsets);
	glDeleteBuffers(1, &uParticleIndexBuffer);
	glDeleteBuffers(1, &vParticleIndexBuffer);
	glDeleteBuffers(1, &wParticleIndexBuffer);
	glDeleteBuffers(1, &uParticleCurCounts);
	glDeleteBuffers(1, &vParticleCurCounts);
	glDeleteBuffers(1, &wParticleCurCounts);

	glDeleteBuffers(1, &diagA);
	glDeleteBuffers(1, &one_diagA);
	glDeleteBuffers(1, &offDiagA);
	glDeleteBuffers(1, &offsetsSizesBuffer);
	glDeleteBuffers(1, &colNums);
	glDeleteBuffers(1, &x);
	glDeleteBuffers(1, &b);
	glDeleteBuffers(1, &ax);
	glDeleteBuffers(1, &r);
	glDeleteBuffers(1, &d);
	glDeleteBuffers(1, &q);
	glDeleteBuffers(1, &s);
	glDeleteBuffers(1, &numFluidCells);
	glDeleteBuffers(1, &cbHashMap);
	glDeleteBuffers(1, &cellIdxToColNumMap);
	glDeleteBuffers(1, &bucketLocks);

	glDeleteBuffers(1, &inSpeedBuffer);
	glDeleteBuffers(1, &outSpeedBuffer);
	glDeleteBuffers(1, &numGlobalIterationsBuffer);

	glDeleteProgram(countNumParticleInCellsProgram);
	glDeleteProgram(buildParticleIndexBufferProgram);
	glDeleteProgram(gatherParticleVelocitiesProgram);
	glDeleteProgram(saveVelocitiesProgram);

	glDeleteProgram(accByGravityProgram);

	glDeleteProgram(buildCellIdxToColNumHashMapProgram);
	glDeleteProgram(buildPressureSolveLinearSystemProgram);
	glDeleteProgram(transferPressureBackToGridProgram);
	glDeleteProgram(updateVelocitiesUsingPressureGradientsProgram);

	glDeleteProgram(transferGridVelocitiesToParticlesProgram);
	
	glDeleteProgram(advectParticlesProgram);

	glDeleteProgram(extrapolateGridVelocitiesProgram);

	glDeleteProgram(renderProgram);

	glDeleteProgram(findMaxParticleSpeedProgram);

	glDeleteVertexArrays(1, &vao);
}


void FS_GPU_MACGRID::seedParticles()
{
	const int numSamples2D = numSamples1D * numSamples1D;
	const int numSamples3D = numSamples2D * numSamples1D;
	const int xExtent = (xcount - 2) / 4;
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
	int numCells = (this->numCells + segmentSize - 1) / segmentSize * segmentSize;
	int numParticles = (this->numParticles + segmentSize - 1) / segmentSize * segmentSize;

	glGenBuffers(1, &cbMeta);
	glBindBuffer(GL_UNIFORM_BUFFER, cbMeta);
	glBufferData(GL_UNIFORM_BUFFER, 7 * 4, NULL, GL_STREAM_DRAW);

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

	glGenBuffers(1, &uParticleCounts);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, uParticleCounts);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	glGenBuffers(1, &vParticleCounts);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, vParticleCounts);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	glGenBuffers(1, &wParticleCounts);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, wParticleCounts);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &uParticleIndexOffsets);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, uParticleIndexOffsets);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	glGenBuffers(1, &vParticleIndexOffsets);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, vParticleIndexOffsets);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	glGenBuffers(1, &wParticleIndexOffsets);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, wParticleIndexOffsets);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &uParticleCurCounts);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, uParticleCurCounts);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	glGenBuffers(1, &vParticleCurCounts);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, vParticleCurCounts);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	glGenBuffers(1, &wParticleCurCounts);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, wParticleCurCounts);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &uParticleIndexBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, uParticleIndexBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles * 8 * 4, NULL, GL_DYNAMIC_COPY);
	glGenBuffers(1, &vParticleIndexBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, vParticleIndexBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles * 8 * 4, NULL, GL_DYNAMIC_COPY);
	glGenBuffers(1, &wParticleIndexBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, wParticleIndexBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles * 8 * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &diagA);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, diagA);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	//glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_STATIC_DRAW);

	glGenBuffers(1, &one_diagA);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, one_diagA);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	//glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_STATIC_DRAW);

	glGenBuffers(1, &offDiagA);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, offDiagA);
	// each cell can have upto 6 fluid neighbours (-1.f)
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 6 * 4, NULL, GL_DYNAMIC_COPY);
	//glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 6 * 4, NULL, GL_STATIC_DRAW);

	glGenBuffers(1, &offsetsSizesBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, offsetsSizesBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 2 * 4, NULL, GL_DYNAMIC_COPY);
	//glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 2 * 4, NULL, GL_STATIC_DRAW);

	glGenBuffers(1, &colNums);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, colNums);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 6 * 4, NULL, GL_DYNAMIC_COPY);
	//glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 6 * 4, NULL, GL_STATIC_DRAW);

	glGenBuffers(1, &x);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, x);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	//glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_READ);

	glGenBuffers(1, &b);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);
	//glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_STATIC_DRAW);

	glGenBuffers(1, &ax);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ax);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &r);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, r);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &d);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, d);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &q);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, q);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &s);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, s);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numCells * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &numFluidCells);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, numFluidCells);
	glBufferData(GL_SHADER_STORAGE_BUFFER, 4, NULL, GL_DYNAMIC_READ); // just an integer

	glGenBuffers(1, &cellIdxToColNumMap);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellIdxToColNumMap);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numBuckets * bucketSize * entrySize, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &cbHashMap);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbHashMap);
	glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * 4, NULL, GL_STREAM_DRAW);

	glGenBuffers(1, &bucketLocks);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bucketLocks);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numBuckets * 4, NULL, GL_DYNAMIC_COPY);

	int speedBufferSizeUpperBound = (numParticles + segmentSize - 1) / segmentSize;
	speedBufferSizeUpperBound = (speedBufferSizeUpperBound + segmentSize - 1) / segmentSize * segmentSize;
	glGenBuffers(1, &inSpeedBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, inSpeedBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, speedBufferSizeUpperBound * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &outSpeedBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, outSpeedBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, speedBufferSizeUpperBound * 4, NULL, GL_DYNAMIC_COPY);

	glGenBuffers(1, &numGlobalIterationsBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, numGlobalIterationsBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, 4, NULL, GL_DYNAMIC_COPY);

	glGenVertexArrays(1, &vao);
}


void FS_GPU_MACGRID::genShaderPrograms()
{
	auto createGPUKernel = [](const char *fn, GLuint *p_program)
	{
		std::string fileName;
		std::vector<GLenum> shaderTypes;
		std::vector<const char *> shaderFileNames;

		fileName = SHADERS_DIR;
		fileName += "/";
		fileName += fn;
		shaderTypes.push_back(GL_COMPUTE_SHADER);
		shaderFileNames.push_back(fileName.c_str());
		createProgramWithShaders(shaderTypes, shaderFileNames, *p_program);
	};

	createGPUKernel("countNumParticleInCells_cs.glsl", &countNumParticleInCellsProgram);
	createGPUKernel("buildParticleIndexBuffer_cs.glsl", &buildParticleIndexBufferProgram);
	createGPUKernel("gatherParticleVelocities_cs.glsl", &gatherParticleVelocitiesProgram);
	createGPUKernel("accByGravity_cs.glsl", &accByGravityProgram);
	createGPUKernel("saveVelocities_cs.glsl", &saveVelocitiesProgram);
	createGPUKernel("buildCellIdxToColNumHashMap_cs.glsl", &buildCellIdxToColNumHashMapProgram);
	createGPUKernel("buildPressureSolveLinearSystem_cs.glsl", &buildPressureSolveLinearSystemProgram);
	createGPUKernel("transferPressureBackToGrid_cs.glsl", &transferPressureBackToGridProgram);
	createGPUKernel("updateVelocitiesUsingPressureGradients_cs.glsl", &updateVelocitiesUsingPressureGradientsProgram);
	createGPUKernel("transferGridVelocitiesToParticles_cs.glsl", &transferGridVelocitiesToParticlesProgram);
	createGPUKernel("advectParticles_cs.glsl", &advectParticlesProgram);
	createGPUKernel("extrapolateGridVelocities_cs.glsl", &extrapolateGridVelocitiesProgram);
	createGPUKernel("findMaxParticleSpeed_cs.glsl", &findMaxParticleSpeedProgram);

	std::vector<GLenum> shaderTypes;
	std::vector<const char *> shaderFileNames;
	std::string vsFileName(SHADERS_DIR);
	std::string fsFileName(SHADERS_DIR);

	vsFileName += "/boxSource_vs.glsl";
	fsFileName += "/boxSource_fs.glsl";
	shaderTypes.push_back(GL_VERTEX_SHADER);
	shaderTypes.push_back(GL_FRAGMENT_SHADER);
	shaderFileNames.push_back(vsFileName.c_str());
	shaderFileNames.push_back(fsFileName.c_str());
	createProgramWithShaders(shaderTypes, shaderFileNames, renderProgram);
}

struct Test
{
	int xcount;
	int ycount;
	int zcount;
	float cellSize;
	int numParticles;
	int numCells;
	float deltaTime;
};

void FS_GPU_MACGRID::transferParticleVelocitiesToGrid()
{
	glBindBufferBase(GL_UNIFORM_BUFFER, 1, cbMeta);

	// DEBUG
	//Test test;
	//glBindBuffer(GL_COPY_READ_BUFFER, cbMeta);
	//char *mapped = (char *)glMapBufferRange(GL_COPY_READ_BUFFER, 0, 7 * 4, GL_MAP_READ_BIT);
	//memcpy(&test, mapped, 7 * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	// count how many particles are influencing each cell
	int numCellsUpperBound = (numCells + segmentSize - 1) / segmentSize * segmentSize;

	// DEBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, uParticleCounts);
	//void *mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp(numCellsUpperBound);
	//memcpy(&tmp[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	gpuops->zeroBuffer(uParticleCounts, numCellsUpperBound);
	gpuops->zeroBuffer(vParticleCounts, numCellsUpperBound);
	gpuops->zeroBuffer(wParticleCounts, numCellsUpperBound);

	// DBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, uParticleCounts);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//memcpy(&tmp[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particles);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, uParticleCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vParticleCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, wParticleCounts);

	glUseProgram(countNumParticleInCellsProgram);
	
	int numGroups = (numParticles + 255) / 256;
	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	// DEBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, wParticleCounts);
	//void *mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp(numCellsUpperBound);
	//memcpy(&tmp[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	// compute the offset for each cell
	//gpuops->zeroBuffer(uParticleIndexOffsets, numCellsUpperBound);
	//gpuops->zeroBuffer(vParticleIndexOffsets, numCellsUpperBound);
	//gpuops->zeroBuffer(wParticleIndexOffsets, numCellsUpperBound);

	gpuops->exclusivePrefixSum(uParticleCounts, uParticleIndexOffsets, numCells);
	gpuops->exclusivePrefixSum(vParticleCounts, vParticleIndexOffsets, numCells);
	gpuops->exclusivePrefixSum(wParticleCounts, wParticleIndexOffsets, numCells);

	// DEBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, wParticleIndexOffsets);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp2(numCellsUpperBound);
	//memcpy(&tmp2[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	// build particle index buffer. Also mark fluid cells
	gpuops->zeroBuffer(uParticleCurCounts, numCells);
	gpuops->zeroBuffer(vParticleCurCounts, numCells);
	gpuops->zeroBuffer(wParticleCurCounts, numCells);
	gpuops->zeroBuffer(cellTypes, numCells);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particles);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, uParticleIndexOffsets);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vParticleIndexOffsets);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, wParticleIndexOffsets);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, uParticleCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, vParticleCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, wParticleCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, uParticleCurCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, vParticleCurCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, wParticleCurCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, uParticleIndexBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, vParticleIndexBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, wParticleIndexBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, cellTypes);

	glUseProgram(buildParticleIndexBufferProgram);

	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	//DEBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, uParticleCounts);
	//void *mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp(numCellsUpperBound);
	//memcpy(&tmp[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, uParticleIndexOffsets);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp2(numCellsUpperBound);
	//memcpy(&tmp2[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, uParticleCurCounts);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp3(numCellsUpperBound);
	//memcpy(&tmp3[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, uParticleIndexBuffer);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp4(numCellsUpperBound);
	//memcpy(&tmp4[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, cellTypes);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp5(numCellsUpperBound);
	//memcpy(&tmp5[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	// gather particle velocities into uVels, vVels, and wVels
	gpuops->zeroBuffer(uVels1, numCells);
	gpuops->zeroBuffer(vVels1, numCells);
	gpuops->zeroBuffer(wVels1, numCells);

	// DEBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, uVels1);
	//void *mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp1(numCellsUpperBound);
	//memcpy(&tmp1[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particles);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, uParticleIndexOffsets);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vParticleIndexOffsets);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, wParticleIndexOffsets);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, uParticleCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, vParticleCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, wParticleCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, uParticleIndexBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, vParticleIndexBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, wParticleIndexBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, uVels1);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, vVels1);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, wVels1);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, cellTypes);

	glUseProgram(gatherParticleVelocitiesProgram); // TODO

	numGroups = (numCells + 255) / 256;
	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	// DEBUG
	//Test test;
	//glBindBuffer(GL_COPY_READ_BUFFER, cbMeta);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, 7 * 4, GL_MAP_READ_BIT);
	//memcpy(&test, mapped, 7 * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, uVels1);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp3(numCellsUpperBound);
	//memcpy(&tmp3[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, cellTypes);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp5(numCellsUpperBound);
	//memcpy(&tmp5[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	// save copies of u, v, and w velocities for FLIP
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 7, uVels2, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 8, vVels2, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 9, wVels2, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 10, uVels1, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 11, vVels1, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 12, wVels1, 0, numCells * sizeof(float));

	glUseProgram(saveVelocitiesProgram);

	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	//DEBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, uVels1);
	//void *mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp1(numCellsUpperBound);
	//memcpy(&tmp1[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, uVels2);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCellsUpperBound * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp2(numCellsUpperBound);
	//memcpy(&tmp2[0], mapped, numCellsUpperBound * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);
}


void FS_GPU_MACGRID::accelerateByGravity()
{
	setDeltaTime();

	// DEBUG
	//Test test;
	//glBindBuffer(GL_COPY_READ_BUFFER, cbMeta);
	//void *mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, 7 * 4, GL_MAP_READ_BIT);
	//memcpy(&test, mapped, 7 * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, vVels1);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp2(numCells);
	//memcpy(&tmp2[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, cellTypes, 0, numCells * sizeof(int));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, vVels1, 0, numCells * sizeof(float));

	glUseProgram(accByGravityProgram);

	int numGroups = (numCells + 255) / 256;
	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	// DEBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, vVels1);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp3(numCells);
	//memcpy(&tmp3[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);
}


void FS_GPU_MACGRID::pressureSolve()
{
	// build diagA, one_diagA, offDiagA, offsetsSizesBuffer, colNums, and b
	buildPressureSolveLinearSystem();

	// CPU build linear system
	//auto flatIdx = [this](int i, int j, int k) -> int
	//{
	//	return i * (ycount * zcount) + j * zcount + k;
	//};

	//glBindBuffer(GL_COPY_READ_BUFFER, cellTypes);
	//void *mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<int> cpuCellTypes(numCells);
	//memcpy(&cpuCellTypes[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, uVels1);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> us(numCells);
	//memcpy(&us[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, vVels1);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> vs(numCells);
	//memcpy(&vs[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, wVels1);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> ws(numCells);
	//memcpy(&ws[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//int numFluidCells = 0;
	//std::unordered_map<int, int> mapping; // fluid cell flat index to row number

	//for (int i = 0; i < xcount; ++i)
	//{
	//	for (int j = 0; j < ycount; ++j)
	//	{
	//		for (int k = 0; k < zcount; ++k)
	//		{
	//			if (cpuCellTypes[flatIdx(i, j, k)] == 1)
	//			{
	//				int linIdx = i * ycount * zcount + j * zcount + k;
	//				mapping[linIdx] = numFluidCells;
	//				++numFluidCells;
	//			}
	//		}
	//	}
	//}

	//std::vector<float> diagACPU((numFluidCells + 2047) / 2048 * 2048, 0.f);
	//std::vector<float> one_diagACPU((numFluidCells + 2047) / 2048 * 2048, 0.f);
	//std::vector<float> bCPU((numFluidCells + 2047) / 2048 * 2048, 0.f);
	//std::vector<float> offDiagACPU;
	//std::vector<int> osbCPU(numFluidCells * 2);
	//std::vector<int> colNumsCPU;
	//std::vector<float> zeroVec((numFluidCells + 2047) / 2048 * 2048, 0.f);

	//auto updateRow = [&](int x, int y, int z)
	//{
	//	int numNonsolidNeighbours = 0;
	//	int linIdx = x * ycount * zcount + y * zcount + z;
	//	int rowNum = mapping[linIdx];
	//	int colNum;
	//	int ct;

	//	int offset = offDiagACPU.size();
	//	int size = 0;

	//	ct = cpuCellTypes[flatIdx(x - 1, y, z)];
	//	numNonsolidNeighbours += (ct == 2) ? 0 : 1;
	//	if (ct == 1)
	//	{
	//		colNum = mapping[(x - 1) * ycount * zcount + y * zcount + z];
	//		offDiagACPU.push_back(-1.f);
	//		colNumsCPU.push_back(colNum);
	//		++size;
	//	}

	//	ct = cpuCellTypes[flatIdx(x + 1, y, z)];
	//	numNonsolidNeighbours += (ct == 2) ? 0 : 1;
	//	if (ct == 1)
	//	{
	//		colNum = mapping[(x + 1) * ycount * zcount + y * zcount + z];
	//		offDiagACPU.push_back(-1.f);
	//		colNumsCPU.push_back(colNum);
	//		++size;
	//	}

	//	ct = cpuCellTypes[flatIdx(x, y - 1, z)];
	//	numNonsolidNeighbours += (ct == 2) ? 0 : 1;
	//	if (ct == 1)
	//	{
	//		colNum = mapping[x * ycount * zcount + (y - 1) * zcount + z];
	//		offDiagACPU.push_back(-1.f);
	//		colNumsCPU.push_back(colNum);
	//		++size;
	//	}

	//	ct = cpuCellTypes[flatIdx(x, y + 1, z)];
	//	numNonsolidNeighbours += (ct == 2) ? 0 : 1;
	//	if (ct == 1)
	//	{
	//		colNum = mapping[x * ycount * zcount + (y + 1) * zcount + z];
	//		offDiagACPU.push_back(-1.f);
	//		colNumsCPU.push_back(colNum);
	//		++size;
	//	}

	//	ct = cpuCellTypes[flatIdx(x, y, z - 1)];
	//	numNonsolidNeighbours += (ct == 2) ? 0 : 1;
	//	if (ct == 1)
	//	{
	//		colNum = mapping[x * ycount * zcount + y * zcount + (z - 1)];
	//		offDiagACPU.push_back(-1.f);
	//		colNumsCPU.push_back(colNum);
	//		++size;
	//	}

	//	ct = cpuCellTypes[flatIdx(x, y, z + 1)];
	//	numNonsolidNeighbours += (ct == 2) ? 0 : 1;
	//	if (ct == 1)
	//	{
	//		colNum = mapping[x * ycount * zcount + y * zcount + (z + 1)];
	//		offDiagACPU.push_back(-1.f);
	//		colNumsCPU.push_back(colNum);
	//		++size;
	//	}

	//	osbCPU[rowNum * 2] = offset;
	//	osbCPU[rowNum * 2 + 1] = size;
	//	diagACPU[rowNum] = static_cast<float>(numNonsolidNeighbours);
	//	one_diagACPU[rowNum] = 1.f / numNonsolidNeighbours;
	//};

	//float deltaTime = 1.f / 60.f;

	//auto computeDivergence = [&](int x, int y, int z) -> float
	//{
	//	float result;

	//	result = us[flatIdx(x + 1, y, z)] - us[flatIdx(x, y, z)] +
	//		vs[flatIdx(x, y + 1, z)] - vs[flatIdx(x, y, z)] +
	//		ws[flatIdx(x, y, z + 1)] - ws[flatIdx(x, y, z)];
	//	result *= cellSize / deltaTime;

	//	return result;
	//};

	//for (int i = 0; i < xcount; ++i)
	//{
	//	for (int j = 0; j < ycount; ++j)
	//	{
	//		for (int k = 0; k < zcount; ++k)
	//		{
	//			if (cpuCellTypes[flatIdx(i, j, k)] == 1)
	//			{
	//				updateRow(i, j, k); // update corresponding row in A

	//				float negDiv = -computeDivergence(i, j, k);
	//				int linIdx = i * ycount * zcount + j * zcount + k;
	//				int mappedIdx = mapping[linIdx];
	//				bCPU[mappedIdx] = negDiv;
	//			}
	//		}
	//	}
	//}

	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, diagA);
	//glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, diagACPU.size() * sizeof(float), diagACPU.data());

	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, one_diagA);
	//glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, one_diagACPU.size() * sizeof(float), one_diagACPU.data());

	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, offsetsSizesBuffer);
	//glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, osbCPU.size() * sizeof(float), osbCPU.data());

	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, colNums);
	//glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, colNumsCPU.size() * sizeof(float), colNumsCPU.data());

	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, offDiagA);
	//glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, offDiagACPU.size() * sizeof(float), offDiagACPU.data());

	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
	//glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bCPU.size() * sizeof(float), bCPU.data());

	//numFluidCells_CPU = numFluidCells;

	// clear temporary and result buffers x, ax, r, d, q, and s
	int numCellsUpperBound = (numCells + segmentSize - 1) / segmentSize * segmentSize;

	gpuops->zeroBuffer(x, numCellsUpperBound);
	gpuops->zeroBuffer(ax, numCellsUpperBound);
	gpuops->zeroBuffer(r, numCellsUpperBound);
	gpuops->zeroBuffer(d, numCellsUpperBound);
	gpuops->zeroBuffer(q, numCellsUpperBound);
	gpuops->zeroBuffer(s, numCellsUpperBound);

	// solve Ax = b using cg
	gpuops->setSize1D(numFluidCells_CPU);
	//gpuops->setSize1D(numFluidCells);
	gpuops->bindConstantBuffer();

	float delta_new, delta_old, delta0;

	gpuops->sparseMv(diagA, offDiagA, offsetsSizesBuffer, colNums, x, ax); // Ax
	gpuops->vadd(1.f, b, -1.f, ax, r); // r = b - Ax
	gpuops->vmul(one_diagA, r, d); // d = M-1 * r
	gpuops->dot(r, d, &delta_new);

	int i = 0;
	int iMax = numFluidCells_CPU;
	//int iMax = numFluidCells;
	delta0 = delta_new;

	while (i < iMax && delta_new > 1e-7 * delta0)
	//while (i < iMax)
	{
		gpuops->sparseMv(diagA, offDiagA, offsetsSizesBuffer, colNums, d, q); // q = Ad
		float tmp;
		gpuops->dot(d, q, &tmp);
		float alpha = delta_new / tmp;
		gpuops->vadd(1.f, x, alpha, d, x); // x_i+1 = x_i + alpha * d

		if (i % 50 == 0)
		{
			gpuops->sparseMv(diagA, offDiagA, offsetsSizesBuffer, colNums, x, ax);
			gpuops->vadd(1.f, b, -1.f, ax, r);
		}
		else
		{
			gpuops->vadd(1.f, r, -alpha, q, r); // r_i+1 = r_i - alpha * q
		}

		gpuops->vmul(one_diagA, r, s); // s = M-1 * r
		delta_old = delta_new;
		gpuops->dot(r, s, &delta_new);
		float beta = delta_new / delta_old;
		gpuops->vadd(1.f, s, beta, d, d); // d_i+1 = s + beta * d_i

		++i;
	}
	//std::cout << "";

	// CPU pressure solve
	//glBindBuffer(GL_COPY_READ_BUFFER, diagA);
	//void *mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp1(numCells);
	//memcpy(&tmp1[0], mapped1, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, one_diagA);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp2(numCells);
	//memcpy(&tmp2[0], mapped1, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, offDiagA);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 6 * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp3(numCells * 6);
	//memcpy(&tmp3[0], mapped1, numCells * 6 * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, offsetsSizesBuffer);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 2 * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp4(numCells * 2);
	//memcpy(&tmp4[0], mapped1, numCells * 2 * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, colNums);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 6 * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp5(numCells * 6);
	//memcpy(&tmp5[0], mapped1, numCells * 6 * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, b);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp6(numCells);
	//memcpy(&tmp6[0], mapped1, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//std::vector<Eigen::Triplet<double> > nonzeroEntries;
	//Eigen::VectorXd resPressure;
	//Eigen::VectorXd cpuB;

	//resPressure = Eigen::VectorXd::Zero(numFluidCells_CPU);
	//cpuB = Eigen::VectorXd::Zero(numFluidCells_CPU);

	//for (int i = 0; i < numFluidCells_CPU; ++i)
	//{
	//	nonzeroEntries.push_back(Eigen::Triplet<double>(i, i, tmp1[i])); // diagA

	//	int offset = tmp4[2 * i];
	//	int counts = tmp4[2 * i + 1];

	//	for (int j = 0; j < counts; ++j)
	//	{
	//		int col = tmp5[offset + j];
	//		nonzeroEntries.push_back(Eigen::Triplet<double>(i, col, -1.0));
	//	}

	//	cpuB(i) = tmp6[i];
	//}

	//Eigen::SparseMatrix<double> A(numFluidCells_CPU, numFluidCells_CPU);
	//A.setZero();
	//A.setFromTriplets(nonzeroEntries.begin(), nonzeroEntries.end());

	//Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::IncompleteCholesky<double> > cg(A);
	//resPressure = cg.solve(cpuB);

	//std::vector<float> xCPU;
	//for (int i = 0; i < numFluidCells_CPU; ++i)
	//{
	//	xCPU.push_back(resPressure(i));
	//}

	//std::vector<float> xCPU(numFluidCells_CPU);
	//glBindBuffer(GL_COPY_READ_BUFFER, x);
	//void *mapped2 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numFluidCells_CPU * sizeof(float), GL_MAP_READ_BIT);
	//memcpy(&xCPU[0], mapped2, numFluidCells_CPU * sizeof(float));
	//glUnmapBuffer(GL_COPY_READ_BUFFER);
	//std::cout << "";

	//std::vector<float> ps(numCells, 0.f);

	//tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount, 0, ycount, 0, zcount), [&](const tbb::blocked_range3d<int> &r) {
	//	for (int i = r.pages().begin(); i != r.pages().end(); ++i)
	//	{
	//		for (int j = r.rows().begin(); j != r.rows().end(); ++j)
	//		{
	//			for (int k = r.cols().begin(); k != r.cols().end(); ++k)
	//			{
	//				if (cpuCellTypes[flatIdx(i, j, k)] == 1)
	//				{
	//					int linIdx = i * ycount * zcount + j * zcount + k;
	//					int mappedIdx = mapping[linIdx];
	//					ps[flatIdx(i, j, k)] = xCPU[mappedIdx];
	//				}
	//			}
	//		}
	//	}
	//});

	//auto updateU = [&](int x, int y, int z)
	//{
	//	float dp;
	//	float dt_dx = deltaTime / cellSize;

	//	if (x != 0 && x != 1 && x != xcount)
	//	{
	//		dp = ps[flatIdx(x, y, z)] - ps[flatIdx(x - 1, y, z)];
	//		us[flatIdx(x, y, z)] -= dt_dx * dp;
	//	}
	//};

	//auto updateV = [&](int x, int y, int z)
	//{
	//	float dp;
	//	float dt_dx = deltaTime / cellSize;

	//	if (y != 0 && y != 1 && y != ycount)
	//	{
	//		dp = ps[flatIdx(x, y, z)] - ps[flatIdx(x, y - 1, z)];
	//		vs[flatIdx(x, y, z)] -= dt_dx * dp;
	//	}
	//};

	//auto updateW = [&](int x, int y, int z)
	//{
	//	float dp;
	//	float dt_dx = deltaTime / cellSize;

	//	if (z != 0 && z != 1 && z != zcount)
	//	{
	//		dp = ps[flatIdx(x, y, z)] - ps[flatIdx(x, y, z - 1)];
	//		ws[flatIdx(x, y, z)] -= dt_dx * dp;
	//	}
	//};

	//tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount, 0, ycount, 0, zcount), [&](const tbb::blocked_range3d<int> &r) {
	//	for (int i = r.pages().begin(); i != r.pages().end(); ++i)
	//	{
	//		for (int j = r.rows().begin(); j != r.rows().end(); ++j)
	//		{
	//			for (int k = r.cols().begin(); k != r.cols().end(); ++k)
	//			{
	//				updateU(i, j, k);
	//			}
	//		}
	//	}
	//});

	//tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount, 0, ycount, 0, zcount), [&](const tbb::blocked_range3d<int> &r) {
	//	for (int i = r.pages().begin(); i != r.pages().end(); ++i)
	//	{
	//		for (int j = r.rows().begin(); j != r.rows().end(); ++j)
	//		{
	//			for (int k = r.cols().begin(); k != r.cols().end(); ++k)
	//			{
	//				updateV(i, j, k);
	//			}
	//		}
	//	}
	//});

	//tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount, 0, ycount, 0, zcount), [&](const tbb::blocked_range3d<int> &r) {
	//	for (int i = r.pages().begin(); i != r.pages().end(); ++i)
	//	{
	//		for (int j = r.rows().begin(); j != r.rows().end(); ++j)
	//		{
	//			for (int k = r.cols().begin(); k != r.cols().end(); ++k)
	//			{
	//				updateW(i, j, k);
	//			}
	//		}
	//	}
	//});

	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, uVels1);
	//glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, us.size() * sizeof(float), us.data());

	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, vVels1);
	//glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vs.size() * sizeof(float), vs.data());

	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, wVels1);
	//glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, ws.size() * sizeof(float), ws.data());

	// DEBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, x);
	//void *mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp(numCells);
	//memcpy(&tmp[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	// update the pressure grid
	gpuops->zeroBuffer(pressures, numCells);

	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, x, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, cellTypes, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, cellIdxToColNumMap, 0, numBuckets * bucketSize * entrySize);
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, pressures, 0, numCells * sizeof(float));

	glUseProgram(transferPressureBackToGridProgram);

	int numGroups = (numCells + 255) / 256;
	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	// DEBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, cellTypes);
	//void *mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp1(numCells);
	//memcpy(&tmp1[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, pressures);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp2(numCells);
	//memcpy(&tmp2[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, uVels1);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp3(numCells);
	//memcpy(&tmp3[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, vVels1);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp4(numCells);
	//memcpy(&tmp4[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, wVels1);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp5(numCells);
	//memcpy(&tmp5[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	// update velocities using pressure gradients
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, uVels1, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, vVels1, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, wVels1, 0, numCells * sizeof(float));

	glUseProgram(updateVelocitiesUsingPressureGradientsProgram);

	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	// DEBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, uVels1);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp6(numCells);
	//memcpy(&tmp6[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, vVels1);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp7(numCells);
	//memcpy(&tmp7[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, wVels1);
	//mapped = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp8(numCells);
	//memcpy(&tmp8[0], mapped, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);
}


struct testEntry
{
	int isOccupied;
	int flatIdx;
	int rowNum;
	int pNext;
};

void FS_GPU_MACGRID::buildPressureSolveLinearSystem()
{
	// for each cell, if it is fluid cell, increment the counter and insert an entry into the hash map
	gpuops->zeroBuffer(cellIdxToColNumMap, numBuckets * bucketSize * 4);
	gpuops->zeroBuffer(bucketLocks, numBuckets);

	const int zero = 0;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, numFluidCells);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(int), &zero);

	glBindBufferBase(GL_UNIFORM_BUFFER, 2, cbHashMap);

	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, cellTypes, 0, numCells * sizeof(int));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, numFluidCells, 0, sizeof(int));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, bucketLocks, 0, numBuckets * sizeof(int));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, cellIdxToColNumMap, 0, numBuckets * bucketSize * entrySize);

	glUseProgram(buildCellIdxToColNumHashMapProgram);

	int numGroups = (numCells + 255) / 256;
	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	glBindBuffer(GL_COPY_READ_BUFFER, numFluidCells);
	int *mapped = (int *)glMapBufferRange(GL_COPY_READ_BUFFER, 0, 4, GL_MAP_READ_BIT);
	numFluidCells_CPU = *mapped;
	glUnmapBuffer(GL_COPY_READ_BUFFER);

	// DEBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, cellTypes);
	//void *mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp1(numCells);
	//memcpy(&tmp1[0], mapped1, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, bucketLocks);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numBuckets * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp2(numBuckets);
	//memcpy(&tmp2[0], mapped1, numBuckets * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, cellIdxToColNumMap);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numBuckets * bucketSize * entrySize, GL_MAP_READ_BIT);
	//std::vector<testEntry> tmp3(numBuckets * bucketSize);
	//memcpy(&tmp3[0], mapped1, numBuckets * bucketSize * entrySize);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	// build diagA, one_diagA, offDiagA, offsetsSizesBuffer, colNums, and b
	int numCellsUpperBound = (numCells + segmentSize - 1) / segmentSize * segmentSize;

	gpuops->zeroBuffer(diagA, numCellsUpperBound);
	gpuops->zeroBuffer(one_diagA, numCellsUpperBound);
	gpuops->zeroBuffer(offDiagA, numCells * 6);
	gpuops->zeroBuffer(offsetsSizesBuffer, numCells * 2);
	gpuops->zeroBuffer(colNums, numCells * 6);
	gpuops->zeroBuffer(b, numCellsUpperBound);

	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, cellTypes, 0, numCells * sizeof(int));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, cellIdxToColNumMap, 0, numBuckets * bucketSize * entrySize);
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 4, diagA, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 5, one_diagA, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 6, offDiagA, 0, numCells * 6 * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 7, offsetsSizesBuffer, 0, numCells * 2 * sizeof(int));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 8, colNums, 0, numCells * 6 * sizeof(int));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 9, b, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 10, uVels1, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 11, vVels1, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 12, wVels1, 0, numCells * sizeof(float));

	glUseProgram(buildPressureSolveLinearSystemProgram);

	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	// DEBUG
	//glBindBuffer(GL_COPY_READ_BUFFER, diagA);
	//void *mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp1(numCells);
	//memcpy(&tmp1[0], mapped1, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, one_diagA);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp2(numCells);
	//memcpy(&tmp2[0], mapped1, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, offDiagA);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 6 * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp3(numCells * 6);
	//memcpy(&tmp3[0], mapped1, numCells * 6 * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, offsetsSizesBuffer);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 2 * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp4(numCells * 2);
	//memcpy(&tmp4[0], mapped1, numCells * 2 * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, colNums);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 6 * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp5(numCells * 6);
	//memcpy(&tmp5[0], mapped1, numCells * 6 * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//for (int i = 0; i < numFluidCells_CPU; ++i)
	//{
	//	int offset = tmp4[i * 2];
	//	int counts = tmp4[i * 2 + 1];

	//	for (int j = 0; j < counts; ++j)
	//	{
	//		int col = tmp5[offset + j];
	//		int colOffset = tmp4[col * 2];
	//		int colCounts = tmp4[col * 2 + 1];
	//		bool rowFound = false;

	//		for (int k = 0; k < colCounts; ++k)
	//		{
	//			if (tmp5[colOffset + k] == i)
	//			{
	//				rowFound = true;
	//				break;
	//			}
	//		}

	//		if (!rowFound)
	//		{
	//			std::cout << "Error: A is not symmetric!\n";
	//			exit(EXIT_FAILURE);
	//		}
	//	}
	//}

	//glBindBuffer(GL_COPY_READ_BUFFER, b);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, numCells * 4, GL_MAP_READ_BIT);
	//std::vector<float> tmp6(numCells);
	//memcpy(&tmp6[0], mapped1, numCells * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	//glBindBuffer(GL_COPY_READ_BUFFER, cbHashMap);
	//mapped1 = glMapBufferRange(GL_COPY_READ_BUFFER, 0, 2 * 4, GL_MAP_READ_BIT);
	//std::vector<int> tmp7(2);
	//memcpy(&tmp7[0], mapped1, 2 * 4);
	//glUnmapBuffer(GL_COPY_READ_BUFFER);
}


void FS_GPU_MACGRID::transferGridVelocitiesToParticles()
{
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, uVels1, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, vVels1, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, wVels1, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, uVels2, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 4, vVels2, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 5, wVels2, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 6, particles, 0, numParticles * 6 * sizeof(float));

	glUseProgram(transferGridVelocitiesToParticlesProgram);

	int numGroups = (numParticles + 255) / 256;
	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();

	// DEBUG
	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, particles);
	//void *mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, numParticles * 6 * sizeof(float), GL_MAP_READ_BIT);
	//std::vector<FS_Particle> cpuParticles(numParticles);
	//memcpy(&cpuParticles[0], mapped, numParticles * 6 * sizeof(float));
	//glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}


void FS_GPU_MACGRID::advectParticles()
{
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, particles, 0, numParticles * 6 * sizeof(float));

	glUseProgram(advectParticlesProgram);

	int numGroups = (numParticles + 255) / 256;
	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();
}


void FS_GPU_MACGRID::extrapolateGridVelocities()
{
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, cellTypes, 0, numCells * sizeof(int));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, uVels1, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, vVels1, 0, numCells * sizeof(float));
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, wVels1, 0, numCells * sizeof(float));

	glUseProgram(extrapolateGridVelocitiesProgram);

	int numGroups = (numCells + 255) / 256;
	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();
}


void FS_GPU_MACGRID::findMaxParticleSpeed()
{
	int segments = (numParticles + segmentSize - 1) / segmentSize;
	int pass = 0;

	const int zero = 0;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, numGlobalIterationsBuffer);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(int), &zero);

	while (true)
	{
		int tmpSize = (segments + segmentSize - 1) / segmentSize * segmentSize;

		gpuops->zeroBuffer(outSpeedBuffer, tmpSize);
		glUseProgram(findMaxParticleSpeedProgram);

		if (pass == 0)
		{
			glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, particles, 0, numParticles * sizeof(FS_Particle));
		}
		else
		{
			glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, inSpeedBuffer, 0, segments * segmentSize * sizeof(float));
		}

		glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, outSpeedBuffer, 0, tmpSize * sizeof(float));

		glDispatchCompute(segments, 1, 1);

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		glFinish();

		if (segments > 1)
		{
			int tmp = outSpeedBuffer;
			outSpeedBuffer = inSpeedBuffer;
			inSpeedBuffer = tmp;
			++pass;
			segments = (segments + segmentSize - 1) / segmentSize;
		}
		else
		{
			glBindBuffer(GL_COPY_READ_BUFFER, outSpeedBuffer);
			float *result = (float *)glMapBufferRange(GL_COPY_READ_BUFFER, 0, sizeof(float), GL_MAP_READ_BIT);
			cpuMaxParticleSpeed = *result;
			glUnmapBuffer(GL_COPY_READ_BUFFER);
			break;
		}
	}
}


void FS_GPU_MACGRID::setCbMeta()
{
	glBindBuffer(GL_UNIFORM_BUFFER, cbMeta);
	char *mapped = (char *)glMapBufferRange(GL_UNIFORM_BUFFER, 0, 7 * 4, GL_MAP_WRITE_BIT);
	*((int *)mapped) = xcount;
	*((int *)(mapped + 4)) = ycount;
	*((int *)(mapped + 8)) = zcount;
	*((float *)(mapped + 12)) = cellSize;
	*((int *)(mapped + 16)) = numParticles;
	*((int *)(mapped + 20)) = numCells;
	glUnmapBuffer(GL_UNIFORM_BUFFER);
}


void FS_GPU_MACGRID::setDeltaTime()
{
	glBindBuffer(GL_UNIFORM_BUFFER, cbMeta);
	char *mapped = (char *)glMapBufferRange(GL_UNIFORM_BUFFER, 24, 4, GL_MAP_WRITE_BIT);
	*((float *)mapped) =
		(cpuMaxParticleSpeed < 0.f)? maxDeltaTime : fmax(minDeltaTime, fmin(maxDeltaTime, (kCFL * cellSize / cpuMaxParticleSpeed)));
	//*((float *)mapped) = 1.f / 60.f;
	glUnmapBuffer(GL_UNIFORM_BUFFER);
}


void FS_GPU_MACGRID::setCbHashMap()
{
	glBindBuffer(GL_UNIFORM_BUFFER, cbHashMap);
	char *mapped = (char *)glMapBufferRange(GL_UNIFORM_BUFFER, 0, 2 * 4, GL_MAP_WRITE_BIT);
	*((int *)mapped) = numBuckets;
	*((int *)(mapped + 4)) = bucketSize;
	glUnmapBuffer(GL_UNIFORM_BUFFER);
}


void FS_GPU_MACGRID::render(std::shared_ptr<FS_Camera> pCam)
{
	static bool onceThrough = false;

	glUseProgram(renderProgram);
	glBindVertexArray(vao);

	if (!onceThrough)
	{
		glBindBuffer(GL_ARRAY_BUFFER, particles);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(FS_Particle), 0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(FS_Particle), (const void *)(3 * sizeof(float)));
	}

	glm::mat4 MVP = pCam->proj * pCam->view; // model matrix is identity
	GLint unif_mvp = glGetUniformLocation(renderProgram, "MVP");
	glUniformMatrix4fv(unif_mvp, 1, GL_FALSE, &MVP[0][0]);

	//GLint uniColorLoc = glGetUniformLocation(renderProgram, "uniColor");
	//glUniform3f(uniColorLoc, 0.f, 0.f, 1.f);

	glPointSize(4);
	glDrawArrays(GL_POINTS, 0, numParticles);
	glPointSize(1);

	//buildSDF(); // uncomment to enable writing VDB files
}