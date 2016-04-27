#include "grid.hpp"
#include "../main.hpp"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/IterativeLinearSolvers"

#include <chrono>
#include <algorithm>
#include <iostream>
#include <unordered_map>


template <class T>
FS_Grid<T>::FS_Grid(int xcount, int ycount, int zcount)
	: xsize(xcount), ysize(ycount), zsize(zcount)
{
	resize(xcount, ycount, zcount);
}


template <class T>
void FS_Grid<T>::resize(int xcount, int ycount, int zcount)
{
	xsize = xcount + 2;
	ysize = ycount + 2;
	zsize = zcount + 2;
	values.resize(xsize * ysize * zsize);
}


template <class T>
void FS_Grid<T>::zeromem()
{
	memset(&values[0], 0, values.size() * sizeof(T));
}


//template <class T>
//T &FS_Grid<T>::operator()(int i, int j, int k)
//{
//	return values[(i + 1) * ysize * zsize + (j + 1) * zsize + (k + 1)];
//}


FS_MACGrid::FS_MACGrid(const FS_BBox &b, float csize)
	: maxVelocity(glm::vec3(0.f)), cellSize(csize), indicatorBufferGenerated(false)
{
	glm::vec3 center = glm::vec3(b.xmin + b.xmax, b.ymin + b.ymax, b.zmin + b.zmax) / 2.f;
	glm::vec3 size = glm::vec3(b.xmax - b.xmin, b.ymax - b.ymin, b.ymax - b.ymin);

	xcount = ceil(size.x / cellSize);
	ycount = ceil(size.y / cellSize);
	zcount = ceil(size.z / cellSize);

	size.x = xcount * cellSize;
	size.y = ycount * cellSize;
	size.z = zcount * cellSize;

	bounds.xmin = center.x - size.x * .5f;
	bounds.xmax = center.x + size.x * .5f;
	bounds.ymin = center.y - size.y * .5f;
	bounds.ymax = center.y + size.y * .5f;
	bounds.zmin = center.z - size.z * .5f;
	bounds.zmax = center.z + size.z * .5f;

	ps.resize(xcount, ycount, zcount);
	us.resize(xcount + 1, ycount, zcount);
	vs.resize(xcount, ycount + 1, zcount);
	ws.resize(xcount, ycount, zcount + 1);
	cellTypes.resize(xcount, ycount, zcount);

	rng = std::uniform_real_distribution<float>(0.f, 1.f);
	generator = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	glGenBuffers(12, buff_tmp);
}


void FS_MACGrid::init()
{
	int size1D = 2;
	int size2D = size1D * size1D;
	int size3D = size1D * size1D * size1D;

	int xsize = xcount / 3;
	int ysize = ycount;
	int zsize = zcount;

	particles1 = std::make_shared<std::vector<FS_Particle> >();
	particles2 = std::make_shared<std::vector<FS_Particle> >();
	particles1->clear();
	particles2->clear();
	particles1->reserve(xsize * ysize * zsize * size3D);
	particles2->resize(xsize * ysize * zsize * size3D);

	//int xstart = (xcount - xsize) / 2;
	//int ystart = (ycount - ysize) / 2;
	//int zstart = (zcount - zsize) / 2;
	int xstart = 0;
	int ystart = 0;
	int zstart = 0;

	for (int i = xstart; i < xstart + xsize; ++i)
	{
		for (int j = ystart; j < ystart + ysize; ++j)
		{
			for (int k = zstart; k < zstart + zsize; ++k)
			{
				for (int h = 0; h < size3D; ++h)
				{
					FS_Particle p;
					int x = h / size2D;
					int y = h / size1D % size1D;
					int z = h % size1D;
					float xpos = (x + rng(generator)) * (cellSize / static_cast<float>(size1D));
					float ypos = (y + rng(generator)) * (cellSize / static_cast<float>(size1D));
					float zpos = (z + rng(generator)) * (cellSize / static_cast<float>(size1D));

					xpos += bounds.xmin + i * cellSize;
					ypos += bounds.ymin + j * cellSize;
					zpos += bounds.zmin + k * cellSize;

					p.position = glm::vec3(xpos, ypos, zpos);
					p.velocity = glm::vec3(0.f);
					particles1->push_back(p);
				}
			}
		}
	}

	particles = particles1;
}


class GPUOPERATIONS
{
public:
	GLuint sparseMvProgram;
	GLuint vaddProgram;
	GLuint vmulProgram;
	GLuint vdivProgram;
	GLuint reduceProgram;

	int size1D;

	GLuint tmp, tmp2;
	GLuint v1v2;

	GLuint cbMetaName;
	char cbMeta[12];

	GPUOPERATIONS()
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
		//reduceProgram = createCSProgram("gpu_reduce_cs_v01.glsl");
		reduceProgram = createCSProgram("gpu_reduce_cs_v02.glsl");

		memset(cbMeta, 0, 12);
		glGenBuffers(1, &cbMetaName);
		glBindBuffer(GL_UNIFORM_BUFFER, cbMetaName);
		glBufferData(GL_UNIFORM_BUFFER, 12, cbMeta, GL_STREAM_DRAW);
		glBindBuffer(GL_UNIFORM_BUFFER, 0); // unbind
	}

	~GPUOPERATIONS()
	{
		glDeleteProgram(sparseMvProgram);
		glDeleteProgram(vaddProgram);
		glDeleteProgram(vmulProgram);
		glDeleteProgram(vdivProgram);
		glDeleteProgram(reduceProgram);
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

		// debug
		//glBindBuffer(GL_COPY_READ_BUFFER, out_x);
		//float *mapped = (float *)glMapBufferRange(GL_COPY_READ_BUFFER, 0, size1D * sizeof(float), GL_MAP_READ_BIT);
		//std::vector<float> tmp(size1D, 0.f);
		//memcpy(&tmp[0], mapped, size1D * sizeof(float));
		//glUnmapBuffer(GL_COPY_READ_BUFFER);
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

		// debug
		//glBindBuffer(GL_COPY_READ_BUFFER, out_v);
		//float *mapped = (float *)glMapBufferRange(GL_COPY_READ_BUFFER, 0, size1D * sizeof(float), GL_MAP_READ_BIT);
		//std::vector<float> tmp(size1D, 0.f);
		//memcpy(&tmp[0], mapped, size1D * sizeof(float));
		//glUnmapBuffer(GL_COPY_READ_BUFFER);

		//glBindBuffer(GL_COPY_READ_BUFFER, v1);
		//mapped = (float *)glMapBufferRange(GL_COPY_READ_BUFFER, 0, size1D * sizeof(float), GL_MAP_READ_BIT);
		//memcpy(&tmp[0], mapped, size1D * sizeof(float));
		//glUnmapBuffer(GL_COPY_READ_BUFFER);
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

		// debug
		//glBindBuffer(GL_COPY_READ_BUFFER, out_v);
		//float *mapped = (float *)glMapBufferRange(GL_COPY_READ_BUFFER, 0, size1D * sizeof(float), GL_MAP_READ_BIT);
		//std::vector<float> tmp(size1D, 0.f);
		//memcpy(&tmp[0], mapped, size1D * sizeof(float));
		//glUnmapBuffer(GL_COPY_READ_BUFFER);
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

	/* VERSION 01 */
	//void reduce(GLuint in_v, float *p_result)
	//{
	//	static bool onceThrough = false;

	//	if (!onceThrough)
	//	{
	//		float tmpCPU[2048] = { 0 };
	//		glGenBuffers(1, &tmp);
	//		glGenBuffers(1, &tmp2);
	//		glBindBuffer(GL_SHADER_STORAGE_BUFFER, tmp);
	//		glBufferData(GL_SHADER_STORAGE_BUFFER, 2048 * sizeof(float), tmpCPU, GL_DYNAMIC_READ);
	//		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	//		onceThrough = true;
	//	}

	//	glUseProgram(reduceProgram);

	//	int segments = (size1D + 2047) / 2048;
	//	int pass = 0;

	//	while (true)
	//	{
	//		std::vector<float> intermediateResults;
	//		intermediateResults.resize((segments + 2047) / 2048 * 2048, 0.f);

	//		for (int i = 0; i < segments; ++i)
	//		{
	//			if (pass == 0)
	//			{
	//				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, in_v, 2048 * i * sizeof(float), 2048 * sizeof(float));
	//			}
	//			else
	//			{
	//				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, tmp2, 2048 * i * sizeof(float), 2048 * sizeof(float));
	//			}

	//			glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, tmp, 0, 2048 * sizeof(float));

	//			glDispatchCompute(1, 1, 1);

	//			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	//			glFinish();

	//			float *result = (float *)glMapNamedBufferRange(tmp, 2047 * sizeof(float), sizeof(float), GL_MAP_READ_BIT);
	//			intermediateResults[i] = *result;
	//			glUnmapNamedBuffer(tmp);
	//		}

	//		if (segments > 1)
	//		{
	//			glBindBuffer(GL_SHADER_STORAGE_BUFFER, tmp2);
	//			glBufferData(GL_SHADER_STORAGE_BUFFER, intermediateResults.size() * sizeof(float), NULL, GL_STATIC_DRAW); // orphaning
	//			glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, intermediateResults.size() * sizeof(float), intermediateResults.data());
	//			++pass;
	//			segments = (segments + 2047) / 2048;
	//		}
	//		else
	//		{
	//			*p_result = intermediateResults[0];
	//			break;
	//		}
	//	}
	//}

	/* VERSION 02 */
	void reduce(GLuint in_v, float *p_result)
	{
		static bool onceThrough = false;
		static int lastSize = -1;
		static std::vector<float> zeroVec;
		
		if (!onceThrough)
		{
			glGenBuffers(1, &tmp);
			glGenBuffers(1, &tmp2);
			onceThrough = true;
		}

		glUseProgram(reduceProgram);
		
		const int localGroupSize = 1024;
		const int segmentSize = localGroupSize * 2;

		int segments = (size1D + segmentSize - 1) / segmentSize;
		int pass = 0;

		if (size1D != lastSize)
		{
			int tmpSize = (segments + segmentSize - 1) / segmentSize * segmentSize;

			zeroVec.resize(tmpSize, 0.f);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, tmp);
			glBufferData(GL_SHADER_STORAGE_BUFFER, zeroVec.size() * sizeof(float), NULL, GL_DYNAMIC_READ);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, tmp2);
			glBufferData(GL_SHADER_STORAGE_BUFFER, zeroVec.size() * sizeof(float), NULL, GL_DYNAMIC_READ);
			lastSize = size1D;
		}

		while (true)
		{
			int tmpSize = (segments + segmentSize - 1) / segmentSize * segmentSize;
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, tmp);
			glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, tmpSize * sizeof(float), zeroVec.data());

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
		static bool onceThrough = false;
		static int lastSize = -1;
		static std::vector<float> v1v2CPU;

		if (!onceThrough)
		{
			glGenBuffers(1, &v1v2);
			onceThrough = true;
		}

		int size = (size1D + 2047) / 2048 * 2048;

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, v1v2);
		
		if (size1D != lastSize)
		{
			v1v2CPU.resize(size, 0.f);
			glBufferData(GL_SHADER_STORAGE_BUFFER, size * sizeof(float), NULL, GL_DYNAMIC_COPY);
			lastSize = size1D;
		}
		
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size * sizeof(float), v1v2CPU.data());

		vmul(v1, v2, v1v2);
		reduce(v1v2, result);
	}
};


void FS_MACGrid::gpu_updatePressureAndVelocity(float deltaTime)
{
	int numFluidCells = 0;
	std::unordered_map<int, int> mapping; // fluid cell flat index to row number

	for (int i = 0; i < xcount; ++i)
	{
		for (int j = 0; j < ycount; ++j)
		{
			for (int k = 0; k < zcount; ++k)
			{
				if (cellTypes(i, j, k) == 1)
				{
					int linIdx = i * ycount * zcount + j * zcount + k;
					mapping[linIdx] = numFluidCells;
					++numFluidCells;
				}
			}
		}
	}

	// Build buffers used in conjugate gradient method
	std::vector<float> diagACPU((numFluidCells + 2047) / 2048 * 2048, 0.f);
	std::vector<float> one_diagACPU((numFluidCells + 2047) / 2048 * 2048, 0.f);
	std::vector<float> bCPU((numFluidCells + 2047) / 2048 * 2048, 0.f);
	std::vector<float> offDiagACPU;
	std::vector<int> osbCPU(numFluidCells * 2);
	std::vector<int> colNumsCPU;
	std::vector<float> zeroVec((numFluidCells + 2047) / 2048 * 2048, 0.f);

	auto updateRow = [&](int x, int y, int z)
	{
		int numNonsolidNeighbours = 0;
		int linIdx = x * ycount * zcount + y * zcount + z;
		int rowNum = mapping[linIdx];
		int colNum;
		int ct;

		int offset = offDiagACPU.size();
		int size = 0;

		ct = (x - 1 < 0) ? 2 : cellTypes(x - 1, y, z);
		numNonsolidNeighbours += (ct == 2) ? 0 : 1;
		if (ct == 1)
		{
			colNum = mapping[(x - 1) * ycount * zcount + y * zcount + z];
			offDiagACPU.push_back(-1.f);
			colNumsCPU.push_back(colNum);
			++size;
		}

		ct = (x + 1 >= xcount) ? 2 : cellTypes(x + 1, y, z);
		numNonsolidNeighbours += (ct == 2) ? 0 : 1;
		if (ct == 1)
		{
			colNum = mapping[(x + 1) * ycount * zcount + y * zcount + z];
			offDiagACPU.push_back(-1.f);
			colNumsCPU.push_back(colNum);
			++size;
		}

		ct = (y - 1 < 0) ? 2 : cellTypes(x, y - 1, z);
		numNonsolidNeighbours += (ct == 2) ? 0 : 1;
		if (ct == 1)
		{
			colNum = mapping[x * ycount * zcount + (y - 1) * zcount + z];
			offDiagACPU.push_back(-1.f);
			colNumsCPU.push_back(colNum);
			++size;
		}

		ct = (y + 1 >= ycount) ? 2 : cellTypes(x, y + 1, z);
		numNonsolidNeighbours += (ct == 2) ? 0 : 1;
		if (ct == 1)
		{
			colNum = mapping[x * ycount * zcount + (y + 1) * zcount + z];
			offDiagACPU.push_back(-1.f);
			colNumsCPU.push_back(colNum);
			++size;
		}

		ct = (z - 1 < 0) ? 2 : cellTypes(x, y, z - 1);
		numNonsolidNeighbours += (ct == 2) ? 0 : 1;
		if (ct == 1)
		{
			colNum = mapping[x * ycount * zcount + y * zcount + (z - 1)];
			offDiagACPU.push_back(-1.f);
			colNumsCPU.push_back(colNum);
			++size;
		}

		ct = (z + 1 >= zcount) ? 2 : cellTypes(x, y, z + 1);
		numNonsolidNeighbours += (ct == 2) ? 0 : 1;
		if (ct == 1)
		{
			colNum = mapping[x * ycount * zcount + y * zcount + (z + 1)];
			offDiagACPU.push_back(-1.f);
			colNumsCPU.push_back(colNum);
			++size;
		}

		osbCPU[rowNum * 2] = offset;
		osbCPU[rowNum * 2 + 1] = size;
		diagACPU[rowNum] = static_cast<float>(numNonsolidNeighbours);
		one_diagACPU[rowNum] = 1.f / numNonsolidNeighbours;
	};

	auto computeDivergence = [&](int x, int y, int z) -> float
	{
		float result;

		result = us(x + 1, y, z) - us(x, y, z) +
			vs(x, y + 1, z) - vs(x, y, z) +
			ws(x, y, z + 1) - ws(x, y, z);
		result *= cellSize / deltaTime;

		return result;
	};

	for (int i = 0; i < xcount; ++i)
	{
		for (int j = 0; j < ycount; ++j)
		{
			for (int k = 0; k < zcount; ++k)
			{
				if (cellTypes(i, j, k) == 1)
				{
					updateRow(i, j, k); // update corresponding row in A

					float negDiv = -computeDivergence(i, j, k);
					int linIdx = i * ycount * zcount + j * zcount + k;
					int mappedIdx = mapping[linIdx];
					bCPU[mappedIdx] = negDiv;
				}
			}
		}
	}

	GLuint diagA, offDiagA, osb, colNums, x, ax, b, one_diagA, r, d, q, s;

	diagA = buff_tmp[0];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, diagA);
	glBufferData(GL_SHADER_STORAGE_BUFFER, diagACPU.size() * sizeof(float), diagACPU.data(), GL_STATIC_DRAW);

	// debug
	//glBindBuffer(GL_COPY_READ_BUFFER, diagA);
	//float *mapped1 = (float *)glMapBufferRange(GL_COPY_READ_BUFFER, 0, diagACPU.size() * sizeof(float), GL_MAP_READ_BIT);
	//std::vector<float> tmptmp(diagACPU.size(), 0.f);
	//memcpy(&tmptmp[0], mapped1, diagACPU.size() * sizeof(float));
	//glUnmapBuffer(GL_COPY_READ_BUFFER);

	one_diagA = buff_tmp[1];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, one_diagA);
	glBufferData(GL_SHADER_STORAGE_BUFFER, one_diagACPU.size() * sizeof(float), one_diagACPU.data(), GL_STATIC_DRAW);

	osb = buff_tmp[2];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, osb);
	glBufferData(GL_SHADER_STORAGE_BUFFER, osbCPU.size() * sizeof(float), osbCPU.data(), GL_STATIC_DRAW);

	colNums = buff_tmp[3];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, colNums);
	glBufferData(GL_SHADER_STORAGE_BUFFER, colNumsCPU.size() * sizeof(float), colNumsCPU.data(), GL_STATIC_DRAW);

	x = buff_tmp[4];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, x);
	glBufferData(GL_SHADER_STORAGE_BUFFER, zeroVec.size() * sizeof(float), zeroVec.data(), GL_DYNAMIC_READ);

	ax = buff_tmp[5];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ax);
	glBufferData(GL_SHADER_STORAGE_BUFFER, zeroVec.size() * sizeof(float), zeroVec.data(), GL_DYNAMIC_COPY);

	b = buff_tmp[6];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
	glBufferData(GL_SHADER_STORAGE_BUFFER, bCPU.size() * sizeof(float), bCPU.data(), GL_STATIC_DRAW);

	r = buff_tmp[7];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, r);
	glBufferData(GL_SHADER_STORAGE_BUFFER, zeroVec.size() * sizeof(float), zeroVec.data(), GL_DYNAMIC_COPY);

	d = buff_tmp[8];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, d);
	glBufferData(GL_SHADER_STORAGE_BUFFER, zeroVec.size() * sizeof(float), zeroVec.data(), GL_DYNAMIC_COPY);

	q = buff_tmp[9];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, q);
	glBufferData(GL_SHADER_STORAGE_BUFFER, zeroVec.size() * sizeof(float), zeroVec.data(), GL_DYNAMIC_COPY);

	s = buff_tmp[10];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, s);
	glBufferData(GL_SHADER_STORAGE_BUFFER, zeroVec.size() * sizeof(float), zeroVec.data(), GL_DYNAMIC_COPY);

	offDiagA = buff_tmp[11];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, offDiagA);
	glBufferData(GL_SHADER_STORAGE_BUFFER, offDiagACPU.size() * sizeof(float), offDiagACPU.data(), GL_STATIC_DRAW);


	static GPUOPERATIONS gpu_ops;

	gpu_ops.setSize1D(numFluidCells);
	gpu_ops.bindConstantBuffer();

	float delta_new, delta_old, delta0;

	gpu_ops.sparseMv(diagA, offDiagA, osb, colNums, x, ax); // Ax
	gpu_ops.vadd(1.f, b, -1.f, ax, r);
	gpu_ops.vmul(one_diagA, r, d);
	gpu_ops.dot(r, d, &delta_new);

	int i = 0, iMax = numFluidCells;
	delta0 = delta_new;

	while (i < iMax && delta_new > 1e-7 * delta0)
	{
		gpu_ops.sparseMv(diagA, offDiagA, osb, colNums, d, q);
		float tmp;
		gpu_ops.dot(d, q, &tmp);
		float alpha = delta_new / tmp;
		gpu_ops.vadd(1.f, x, alpha, d, x);

		if (i % 50 == 0)
		{
			gpu_ops.sparseMv(diagA, offDiagA, osb, colNums, x, ax);
			gpu_ops.vadd(1.f, b, -1.f, ax, r);
		}
		else
		{
			gpu_ops.vadd(1.f, r, -alpha, q, r);
		}

		gpu_ops.vmul(one_diagA, r, s);
		delta_old = delta_new;
		gpu_ops.dot(r, s, &delta_new);
		float beta = delta_new / delta_old;
		gpu_ops.vadd(1.f, s, beta, d, d);

		++i;
	}
	
	std::vector<float> xCPU(numFluidCells);
	float *mapped = (float *)glMapNamedBufferRange(x, 0, numFluidCells * sizeof(float), GL_MAP_READ_BIT);
	memcpy(&xCPU[0], mapped, numFluidCells * sizeof(float));
	glUnmapNamedBuffer(x);

	tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount, 0, ycount, 0, zcount), [&](const tbb::blocked_range3d<int> &r) {
		for (int i = r.pages().begin(); i != r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j != r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k != r.cols().end(); ++k)
				{
					if (cellTypes(i, j, k) == 1)
					{
						int linIdx = i * ycount * zcount + j * zcount + k;
						int mappedIdx = mapping[linIdx];
						ps(i, j, k) = xCPU[mappedIdx];
					}
				}
			}
		}
	});

	// Compute pressure gradients and update velocities for the faces of all fluid cells
	// Boundary face is set to zero
	auto updateU = [&](int x, int y, int z)
	{
		float dp;
		float dt_dx = deltaTime / cellSize;

		if (x != 0 && x != xcount)
		{
			dp = ps(x, y, z) - ps(x - 1, y, z);
			us(x, y, z) -= dt_dx * dp;
		}
	};

	auto updateV = [&](int x, int y, int z)
	{
		float dp;
		float dt_dx = deltaTime / cellSize;

		if (y != 0 && y != ycount)
		{
			dp = ps(x, y, z) - ps(x, y - 1, z);
			vs(x, y, z) -= dt_dx * dp;
		}
	};

	auto updateW = [&](int x, int y, int z)
	{
		float dp;
		float dt_dx = deltaTime / cellSize;

		if (z != 0 && z != zcount)
		{
			dp = ps(x, y, z) - ps(x, y, z - 1);
			ws(x, y, z) -= dt_dx * dp;
		}
	};

	//for (int i = 0; i < xcount + 1; ++i)
	//{
	//	for (int j = 0; j < ycount; ++j)
	//	{
	//		for (int k = 0; k < zcount; ++k)
	//		{
	tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount + 1, 0, ycount, 0, zcount), [&](const tbb::blocked_range3d<int> &r) {
		for (int i = r.pages().begin(); i != r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j != r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k != r.cols().end(); ++k)
				{
					updateU(i, j, k);
				}
			}
		}
	});
	//		}
	//	}
	//}

	//for (int i = 0; i < xcount; ++i)
	//{
	//	for (int j = 0; j < ycount + 1; ++j)
	//	{
	//		for (int k = 0; k < zcount; ++k)
	//		{
	tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount, 0, ycount + 1, 0, zcount), [&](const tbb::blocked_range3d<int> &r) {
		for (int i = r.pages().begin(); i != r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j != r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k != r.cols().end(); ++k)
				{
					updateV(i, j, k);
				}
			}
		}
	});
	//		}
	//	}
	//}

	//for (int i = 0; i < xcount; ++i)
	//{
	//	for (int j = 0; j < ycount; ++j)
	//	{
	//		for (int k = 0; k < zcount + 1; ++k)
	//		{
	tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount, 0, ycount, 0, zcount + 1), [&](const tbb::blocked_range3d<int> &r) {
		for (int i = r.pages().begin(); i != r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j != r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k != r.cols().end(); ++k)
				{
					updateW(i, j, k);
				}
			}
		}
	});
	//		}
	//	}
	//}
}


void FS_MACGrid::updatePressureAndVelocity(float deltaTime)
{
	// Build coefficient matrix A and negative divergence vector b
	int n = xcount * ycount * zcount;
	std::vector<Eigen::Triplet<double> > nonzeroEntries;
	Eigen::VectorXd resPressure;
	Eigen::VectorXd b;
	int numFluidCells = 0;
	std::unordered_map<int, int> mapping;

	auto updateRow = [&](int x, int y, int z)
	{
		int numNonsolidNeighbours = 0;
		int linIdx = x * ycount * zcount + y * zcount + z;
		int rowNum = mapping[linIdx];
		int colNum;
		int ct;

		ct = (x - 1 < 0) ? 2 : cellTypes(x - 1, y, z);
		numNonsolidNeighbours += (ct == 2) ? 0 : 1;
		if (ct == 1)
		{
			colNum = mapping[(x - 1) * ycount * zcount + y * zcount + z];
			nonzeroEntries.push_back(Eigen::Triplet<double>(rowNum, colNum, -1.0));
		}

		ct = (x + 1 >= xcount) ? 2 : cellTypes(x + 1, y, z);
		numNonsolidNeighbours += (ct == 2) ? 0 : 1;
		if (ct == 1)
		{
			colNum = mapping[(x + 1) * ycount * zcount + y * zcount + z];
			nonzeroEntries.push_back(Eigen::Triplet<double>(rowNum, colNum, -1.0));
		}

		ct = (y - 1 < 0) ? 2 : cellTypes(x, y - 1, z);
		numNonsolidNeighbours += (ct == 2) ? 0 : 1;
		if (ct == 1)
		{
			colNum = mapping[x * ycount * zcount + (y - 1) * zcount + z];
			nonzeroEntries.push_back(Eigen::Triplet<double>(rowNum, colNum, -1.0));
		}

		ct = (y + 1 >= ycount) ? 2 : cellTypes(x, y + 1, z);
		numNonsolidNeighbours += (ct == 2) ? 0 : 1;
		if (ct == 1)
		{
			colNum = mapping[x * ycount * zcount + (y + 1) * zcount + z];
			nonzeroEntries.push_back(Eigen::Triplet<double>(rowNum, colNum, -1.0));
		}

		ct = (z - 1 < 0) ? 2 : cellTypes(x, y, z - 1);
		numNonsolidNeighbours += (ct == 2) ? 0 : 1;
		if (ct == 1)
		{
			colNum = mapping[x * ycount * zcount + y * zcount + (z - 1)];
			nonzeroEntries.push_back(Eigen::Triplet<double>(rowNum, colNum, -1.0));
		}

		ct = (z + 1 >= zcount) ? 2 : cellTypes(x, y, z + 1);
		numNonsolidNeighbours += (ct == 2) ? 0 : 1;
		if (ct == 1)
		{
			colNum = mapping[x * ycount * zcount + y * zcount + (z + 1)];
			nonzeroEntries.push_back(Eigen::Triplet<double>(rowNum, colNum, -1.0));
		}

		nonzeroEntries.push_back(Eigen::Triplet<double>(rowNum, rowNum, numNonsolidNeighbours));
	};

	auto computeDivergence = [&](int x, int y, int z) -> float
	{
		float result;

		result = us(x + 1, y, z) - us(x, y, z) +
			vs(x, y + 1, z) - vs(x, y, z) +
			ws(x, y, z + 1) - ws(x, y, z);
		result *= cellSize / deltaTime;

		return result;
	};

	for (int i = 0; i < xcount; ++i)
	{
		for (int j = 0; j < ycount; ++j)
		{
			for (int k = 0; k < zcount; ++k)
			{
				if (cellTypes(i, j, k) == 1)
				{
					int linIdx = i * ycount * zcount + j * zcount + k;
					mapping[linIdx] = numFluidCells;
					++numFluidCells;
				}
			}
		}
	}

	resPressure = Eigen::VectorXd::Zero(numFluidCells);
	b = Eigen::VectorXd::Zero(numFluidCells);

	for (int i = 0; i < xcount; ++i)
	{
		for (int j = 0; j < ycount; ++j)
		{
			for (int k = 0; k < zcount; ++k)
			{
				if (cellTypes(i, j, k) == 1)
				{
					updateRow(i, j, k); // update corresponding row in A

					float negDiv = -computeDivergence(i, j, k);
					int linIdx = i * ycount * zcount + j * zcount + k;
					int mappedIdx = mapping[linIdx];
					b(mappedIdx) = negDiv;
				}
			}
		}
	}

	Eigen::SparseMatrix<double> A(numFluidCells, numFluidCells);
	A.setZero();
	A.setFromTriplets(nonzeroEntries.begin(), nonzeroEntries.end());

	// Solve Ax = b put resulting pressure back to ps
	ps.zeromem();

	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::IncompleteCholesky<double> > cg(A);
	resPressure = cg.solve(b);

	//for (int i = 0; i < xcount; ++i)
	//{
	//	for (int j = 0; j < ycount; ++j)
	//	{
	//		for (int k = 0; k < zcount; ++k)
	//		{
	tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount, 0, ycount, 0, zcount), [&](const tbb::blocked_range3d<int> &r) {
		for (int i = r.pages().begin(); i != r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j != r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k != r.cols().end(); ++k)
				{
					if (cellTypes(i, j, k) == 1)
					{
						int linIdx = i * ycount * zcount + j * zcount + k;
						int mappedIdx = mapping[linIdx];
						ps(i, j, k) = resPressure(mappedIdx);
					}
				}
			}
		}
	});
	//		}
	//	}
	//}

	// Compute pressure gradients and update velocities for the faces of all fluid cells
	// Boundary face is set to zero
	auto updateU = [&](int x, int y, int z)
	{
		float dp;
		float dt_dx = deltaTime / cellSize;

		if (x != 0 && x != xcount)
		{
			dp = ps(x, y, z) - ps(x - 1, y, z);
			us(x, y, z) -= dt_dx * dp;
		}
	};

	auto updateV = [&](int x, int y, int z)
	{
		float dp;
		float dt_dx = deltaTime / cellSize;

		if (y != 0 && y != ycount)
		{
			dp = ps(x, y, z) - ps(x, y - 1, z);
			vs(x, y, z) -= dt_dx * dp;
		}
	};

	auto updateW = [&](int x, int y, int z)
	{
		float dp;
		float dt_dx = deltaTime / cellSize;

		if (z != 0 && z != zcount)
		{
			dp = ps(x, y, z) - ps(x, y, z - 1);
			ws(x, y, z) -= dt_dx * dp;
		}
	};

	//for (int i = 0; i < xcount + 1; ++i)
	//{
	//	for (int j = 0; j < ycount; ++j)
	//	{
	//		for (int k = 0; k < zcount; ++k)
	//		{
	tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount + 1, 0, ycount, 0, zcount), [&](const tbb::blocked_range3d<int> &r) {
		for (int i = r.pages().begin(); i != r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j != r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k != r.cols().end(); ++k)
				{
					updateU(i, j, k);
				}
			}
		}
	});
	//		}
	//	}
	//}

	//for (int i = 0; i < xcount; ++i)
	//{
	//	for (int j = 0; j < ycount + 1; ++j)
	//	{
	//		for (int k = 0; k < zcount; ++k)
	//		{
	tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount, 0, ycount + 1, 0, zcount), [&](const tbb::blocked_range3d<int> &r) {
		for (int i = r.pages().begin(); i != r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j != r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k != r.cols().end(); ++k)
				{
					updateV(i, j, k);
				}
			}
		}
	});
	//		}
	//	}
	//}

	//for (int i = 0; i < xcount; ++i)
	//{
	//	for (int j = 0; j < ycount; ++j)
	//	{
	//		for (int k = 0; k < zcount + 1; ++k)
	//		{
	tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount, 0, ycount, 0, zcount + 1), [&](const tbb::blocked_range3d<int> &r) {
		for (int i = r.pages().begin(); i != r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j != r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k != r.cols().end(); ++k)
				{
					updateW(i, j, k);
				}
			}
		}
	});
	//		}
	//	}
	//}
}


float FS_MACGrid::findMaxSpeed()
{
	float result = 0;

	for (int i = 0; i < xcount; ++i)
	{
		for (int j = 0; j < ycount; ++j)
		{
			for (int k = 0; k < zcount; ++k)
			{
				float u = us(i, j, k);
				float v = vs(i, j, k);
				float w = ws(i, j, k);
				float speed = sqrt(u * u + v * v + w * w);
				if (speed > result)
				{
					result = speed;
				}
			}
		}
	}

	return result;
}


void FS_MACGrid::swapActiveParticleArray()
{
	particles = (particles == particles1) ? particles2 : particles1;
}


void FS_MACGrid::saveVelocities()
{
	tmp_us = us;
	tmp_vs = vs;
	tmp_ws = ws;
}


void FS_MACGrid::accelerateByGravity(float deltaTime, float amount)
{
	for (int x = 0; x < xcount; ++x)
	{
		for (int y = 1; y < ycount; ++y)
		{
			for (int z = 0; z < zcount; ++z)
			{
				if (cellTypes(x, y, z) == 1)
				{
					vs(x, y, z) += amount * deltaTime;
				}
			}
		}
	}
}



//tbb::mutex tpv2gMutex;
tbb::queuing_mutex tpv2gMutex;

void FS_MACGrid::transferParticleVelocityToGrid()
{
	us.zeromem();
	vs.zeromem();
	ws.zeromem();
	cellTypes.zeromem();

	FS_Grid<float> wu(xcount + 1, ycount, zcount);
	FS_Grid<float> wv(xcount, ycount + 1, zcount);
	FS_Grid<float> ww(xcount, ycount, zcount + 1);
	wu.zeromem();
	wv.zeromem();
	ww.zeromem();

	const float ppc = 8.f; // expected number of particles per cell
	const float npn = 8.f * ppc; // number of particles in one-ring neighbourhood
	//const float one_npn = 1.f / npn;
	const float one_npn = 1.f;
	const float epsilon = 1e-5;

	//tbb::parallel_for(tbb::blocked_range<int>(0, particles->size()), [&](const tbb::blocked_range<int> &r) {
	//	for (int h = r.begin(); h != r.end(); ++h)
	//	{
	for (int h = 0; h < particles->size(); ++h)
	{
		FS_Particle &p = (*particles)[h];

		// update maxSpeed
		//if (p.velocity.length() > maxVelocity.length())
		//{
		//	maxVelocity = p.velocity;
		//}

		int x = fmax(0.f, floor((p.position.x - bounds.xmin - epsilon) / cellSize));
		int y = fmax(0.f, floor((p.position.y - bounds.ymin - epsilon) / cellSize));
		int z = fmax(0.f, floor((p.position.z - bounds.zmin - epsilon) / cellSize));
		int xx = x;
		int yy = y;
		int zz = z;

		cellTypes(x, y, z) = 1;

		// update u
		x = fmax(0.f, floor((p.position.x - bounds.xmin - epsilon) / cellSize));
		y = fmax(0.f, floor((p.position.y - .5f * cellSize - bounds.ymin - epsilon) / cellSize));
		z = fmax(0.f, floor((p.position.z - .5f * cellSize - bounds.zmin - epsilon) / cellSize));

		float cx = bounds.xmin + x * cellSize;
		float cy = bounds.ymin + y * cellSize;
		float cz = bounds.zmin + z * cellSize;
		float dx = (p.position.x - cx) / cellSize;
		float dy;
		if (y != yy)
			dy = (2.f * cellSize - p.position.y + cy) / cellSize;
		else
			dy = (p.position.y - cy) / cellSize;
		float dz;
		if (z != zz)
			dz = (2.f * cellSize - p.position.z + cz) / cellSize;
		else
			dz = (p.position.z - cz) / cellSize;
		float w000 = (1.f - dx) * (1.f - dy) * (1.f - dz) * one_npn;
		float w100 = dx * (1.f - dy) * (1.f - dz) * one_npn;
		float w010 = (1.f - dx) * dy * (1.f - dz) * one_npn;
		float w001 = (1.f - dx) * (1.f - dy) * dz * one_npn;
		float w110 = dx * dy * (1.f - dz) * one_npn;
		float w101 = dx * (1.f - dy) * dz * one_npn;
		float w011 = (1.f - dx) * dy * dz * one_npn;
		float w111 = dx * dy * dz * one_npn;

		{
			//tbb::mutex::scoped_lock lock(tpv2gMutex);
			//tbb::queuing_mutex::scoped_lock lock(tpv2gMutex);

			if (x > 0)
			{
				us(x, y, z) += p.velocity.x * w000;
				wu(x, y, z) += w000;
			}
			if (x + 1 < xcount)
			{
				us(x + 1, y, z) += p.velocity.x * w100;
				wu(x + 1, y, z) += w100;
			}
			if (x > 0 && y + 1 < ycount)
			{
				us(x, y + 1, z) += p.velocity.x * w010;
				wu(x, y + 1, z) += w010;
			}
			if (x > 0 && z + 1 < zcount)
			{
				us(x, y, z + 1) += p.velocity.x * w001;
				wu(x, y, z + 1) += w001;
			}
			if (x + 1 < xcount && y + 1 < ycount)
			{
				us(x + 1, y + 1, z) += p.velocity.x * w110;
				wu(x + 1, y + 1, z) += w110;
			}
			if (x + 1 < xcount && z + 1 < zcount)
			{
				us(x + 1, y, z + 1) += p.velocity.x * w101;
				wu(x + 1, y, z + 1) += w101;
			}
			if (x > 0 && y + 1 < ycount && z + 1 < zcount)
			{
				us(x, y + 1, z + 1) += p.velocity.x * w011;
				wu(x, y + 1, z + 1) += w011;
			}
			if (x + 1 < xcount && y + 1 < ycount && z + 1 < zcount)
			{
				us(x + 1, y + 1, z + 1) += p.velocity.x * w111;
				wu(x + 1, y + 1, z + 1) += w111;
			}
		}

		// update v
		x = fmax(0.f, floor((p.position.x - .5f * cellSize - bounds.xmin - epsilon) / cellSize));
		y = fmax(0.f, floor((p.position.y - bounds.ymin - epsilon) / cellSize));
		z = fmax(0.f, floor((p.position.z - .5f * cellSize - bounds.zmin - epsilon) / cellSize));

		cx = bounds.xmin + x * cellSize;
		cy = bounds.ymin + y * cellSize;
		cz = bounds.zmin + z * cellSize;
		if (x != xx)
			dx = (2.f * cellSize - p.position.x + cx) / cellSize;
		else
			dx = (p.position.x - cx) / cellSize;
		dy = (p.position.y - cy) / cellSize;
		if (z != zz)
			dz = (2.f * cellSize - p.position.z + cz) / cellSize;
		else
			dz = (p.position.z - cz) / cellSize;
		w000 = (1.f - dx) * (1.f - dy) * (1.f - dz) * one_npn;
		w100 = dx * (1.f - dy) * (1.f - dz) * one_npn;
		w010 = (1.f - dx) * dy * (1.f - dz) * one_npn;
		w001 = (1.f - dx) * (1.f - dy) * dz * one_npn;
		w110 = dx * dy * (1.f - dz) * one_npn;
		w101 = dx * (1.f - dy) * dz * one_npn;
		w011 = (1.f - dx) * dy * dz * one_npn;
		w111 = dx * dy * dz * one_npn;

		{
			//tbb::mutex::scoped_lock lock(tpv2gMutex);
			//tbb::queuing_mutex::scoped_lock lock(tpv2gMutex);

			if (y > 0)
			{
				vs(x, y, z) += p.velocity.y * w000;
				wv(x, y, z) += w000;
			}
			if (y > 0 && x + 1 < xcount)
			{
				vs(x + 1, y, z) += p.velocity.y * w100;
				wv(x + 1, y, z) += w100;
			}
			if (y + 1 < ycount)
			{
				vs(x, y + 1, z) += p.velocity.y * w010;
				wv(x, y + 1, z) += w010;
			}
			if (y > 0 && z + 1 < zcount)
			{
				vs(x, y, z + 1) += p.velocity.y * w001;
				wv(x, y, z + 1) += w001;
			}
			if (x + 1 < xcount && y + 1 < ycount)
			{
				vs(x + 1, y + 1, z) += p.velocity.y * w110;
				wv(x + 1, y + 1, z) += w110;
			}
			if (y > 0 && x + 1 < xcount && z + 1 < zcount)
			{
				vs(x + 1, y, z + 1) += p.velocity.y * w101;
				wv(x + 1, y, z + 1) += w101;
			}
			if (y + 1 < ycount && z + 1 < zcount)
			{
				vs(x, y + 1, z + 1) += p.velocity.y * w011;
				wv(x, y + 1, z + 1) += w011;
			}
			if (x + 1 < xcount && y + 1 < ycount && z + 1 < zcount)
			{
				vs(x + 1, y + 1, z + 1) += p.velocity.y * w111;
				wv(x + 1, y + 1, z + 1) += w111;
			}
		}

		// update w
		x = fmax(0.f, floor((p.position.x - .5f * cellSize - bounds.xmin - epsilon) / cellSize));
		y = fmax(0.f, floor((p.position.y - .5f * cellSize - bounds.ymin - epsilon) / cellSize));
		z = fmax(0.f, floor((p.position.z - bounds.zmin - epsilon) / cellSize));

		cx = bounds.xmin + x * cellSize;
		cy = bounds.ymin + y * cellSize;
		cz = bounds.zmin + z * cellSize;
		if (x != xx)
			dx = (2.f * cellSize - p.position.x + cx) / cellSize;
		else
			dx = (p.position.x - cx) / cellSize;
		if (y != yy)
			dy = (2.f * cellSize - p.position.y + cy) / cellSize;
		else
			dy = (p.position.y - cy) / cellSize;
		dz = (p.position.z - cz) / cellSize;
		w000 = (1.f - dx) * (1.f - dy) * (1.f - dz) * one_npn;
		w100 = dx * (1.f - dy) * (1.f - dz) * one_npn;
		w010 = (1.f - dx) * dy * (1.f - dz) * one_npn;
		w001 = (1.f - dx) * (1.f - dy) * dz * one_npn;
		w110 = dx * dy * (1.f - dz) * one_npn;
		w101 = dx * (1.f - dy) * dz * one_npn;
		w011 = (1.f - dx) * dy * dz * one_npn;
		w111 = dx * dy * dz * one_npn;

		{
			//tbb::mutex::scoped_lock lock(tpv2gMutex);
			//tbb::queuing_mutex::scoped_lock lock(tpv2gMutex);

			if (z > 0)
			{
				ws(x, y, z) += p.velocity.z * w000;
				ww(x, y, z) += w000;
			}
			if (z > 0 && x + 1 < xcount)
			{
				ws(x + 1, y, z) += p.velocity.z * w100;
				ww(x + 1, y, z) += w100;
			}
			if (z > 0 && y + 1 < ycount)
			{
				ws(x, y + 1, z) += p.velocity.z * w010;
				ww(x, y + 1, z) += w010;
			}
			if (z + 1 < zcount)
			{
				ws(x, y, z + 1) += p.velocity.z * w001;
				ww(x, y, z + 1) += w001;
			}
			if (x + 1 < xcount && y + 1 < ycount && z > 0)
			{
				ws(x + 1, y + 1, z) += p.velocity.z * w110;
				ww(x + 1, y + 1, z) += w110;
			}
			if (x + 1 < xcount && z + 1 < zcount)
			{
				ws(x + 1, y, z + 1) += p.velocity.z * w101;
				ww(x + 1, y, z + 1) += w101;
			}
			if (y + 1 < ycount && z + 1 < zcount)
			{
				ws(x, y + 1, z + 1) += p.velocity.z * w011;
				ww(x, y + 1, z + 1) += w011;
			}
			if (x + 1 < xcount && y + 1 < ycount && z + 1 < zcount)
			{
				ws(x + 1, y + 1, z + 1) += p.velocity.z * w111;
				ww(x + 1, y + 1, z + 1) += w111;
			}
		}
	}
	//});

	tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount + 1, 0, ycount, 0, zcount), [&](const tbb::blocked_range3d<int> &r) {
		for (int i = r.pages().begin(); i != r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j != r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k != r.cols().end(); ++k)
				{
					if (wu(i, j, k) > epsilon)
					{
						us(i, j, k) /= wu(i, j, k);
					}
				}
			}
		}
	});

	tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount, 0, ycount + 1, 0, zcount), [&](const tbb::blocked_range3d<int> &r) {
		for (int i = r.pages().begin(); i != r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j != r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k != r.cols().end(); ++k)
				{
					if (wv(i, j, k) > epsilon)
					{
						vs(i, j, k) /= wv(i, j, k);
					}
				}
			}
		}
	});

	tbb::parallel_for(tbb::blocked_range3d<int>(0, xcount, 0, ycount, 0, zcount + 1), [&](const tbb::blocked_range3d<int> &r) {
		for (int i = r.pages().begin(); i != r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j != r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k != r.cols().end(); ++k)
				{
					if (ww(i, j, k) > epsilon)
					{
						ws(i, j, k) /= ww(i, j, k);
					}
				}
			}
		}
	});
}


//void FS_MACGrid::transferParticleVelocityToGrid()
//{
//	us.zeromem();
//	vs.zeromem();
//	ws.zeromem();
//	cellTypes.zeromem();
//
//	const float ppc = 8.f; // expected number of particles per cell
//	const float npn = 27.f * ppc; // number of particles in one-ring neighbourhood
//	const float one_npn = 1.f / npn;
//	const float epsilon = 1e-5;
//
//	for (int h = 0; h < particles->size(); ++h)
//	{
//		FS_Particle &p = (*particles)[h];
//
//		// update maxSpeed
//		if (p.velocity.length() > maxVelocity.length())
//		{
//			maxVelocity = p.velocity;
//		}
//
//		int x = fmax(0.f, floor((p.position.x - bounds.xmin - epsilon) / cellSize));
//		int y = fmax(0.f, floor((p.position.y - bounds.ymin - epsilon) / cellSize));
//		int z = fmax(0.f, floor((p.position.z - bounds.zmin - epsilon) / cellSize));
//
//		cellTypes(x, y, z) = 1;
//
//		for (int i = x - 1; i <= x + 1; ++i)
//		{
//			for (int j = y - 1; j <= y + 1; ++j)
//			{
//				for (int k = z - 1; k <= z + 1; ++k)
//				{
//					float cx = bounds.xmin + i * cellSize + 0.5f * cellSize;
//					float cy = bounds.ymin + j * cellSize + 0.5f * cellSize;
//					float cz = bounds.zmin + k * cellSize + 0.5f * cellSize;
//					float one_d = 1.f / (1.f + sqrt((cx - p.position.x) * (cx - p.position.x) +
//						(cy - p.position.y) * (cy - p.position.y) +
//						(cz - p.position.z) * (cz - p.position.z)));
//
//					// update u
//					if (i >= 1 && j >= 0 && k >= 0 &&
//						i < xcount && j < ycount && k < zcount)
//					{
//						us(i, j, k) += p.velocity.x * one_d;
//					}
//
//					// update v
//					if (i >= 0 && j >= 1 && k >= 0 &&
//						i < xcount && j < ycount && k < zcount)
//					{
//						vs(i, j, k) += p.velocity.y * one_d;
//					}
//
//					// update w
//					if (i >= 0 && j >= 0 && k >= 1 &&
//						i < xcount && j < ycount && k < zcount)
//					{
//						ws(i, j, k) += p.velocity.z * one_d;
//					}
//				}
//			}
//		}
//	}
//
//	for (int i = 0; i < us.values.size(); ++i)
//	{
//		us.values[i] *= one_npn;
//	}
//
//	for (int i = 0; i < vs.values.size(); ++i)
//	{
//		vs.values[i] *= one_npn;
//	}
//
//	for (int i = 0; i < ws.values.size(); ++i)
//	{
//		ws.values[i] *= one_npn;
//	}
//}


// FLIP
void FS_MACGrid::interpolateVelocityDifference()
{
	std::shared_ptr<std::vector<FS_Particle> > particlesRead = particles;
	std::shared_ptr<std::vector<FS_Particle> > particlesWrite = (particles == particles1) ? particles2 : particles1;

	//for (int h = 0; h < particlesWrite->size(); ++h)
	//{
	tbb::parallel_for(tbb::blocked_range<int>(0, static_cast<int>(particlesWrite->size())), [&](const tbb::blocked_range<int> &r) {
		for (int h = r.begin(); h != r.end(); ++h)
		{
			FS_Particle &p = (*particlesRead)[h];
			FS_Particle &pw = (*particlesWrite)[h];

			pw.position = p.position;
			pw.velocity = p.velocity;

			int x = floor((p.position.x - bounds.xmin) / cellSize);
			int y = floor((p.position.y - bounds.ymin) / cellSize);
			int z = floor((p.position.z - bounds.zmin) / cellSize);
			int x1 = floor((p.position.x - bounds.xmin - 0.5f * cellSize) / cellSize);
			int y1 = floor((p.position.y - bounds.ymin - 0.5f * cellSize) / cellSize);
			int z1 = floor((p.position.z - bounds.zmin - 0.5f * cellSize) / cellSize);

			// update u
			float xstart = x * cellSize + bounds.xmin;
			float ystart = y1 * cellSize + bounds.ymin + 0.5f * cellSize;
			float zstart = z1 * cellSize + bounds.zmin + 0.5f * cellSize;
			float u = (p.position.x - xstart) / cellSize;
			float v = (p.position.y - ystart) / cellSize;
			float w = (p.position.z - zstart) / cellSize;

			float u11 = (1.f - u) * (us(x, y1, z1) - tmp_us(x, y1, z1)) +
				u * (us(x + 1, y1, z1) - tmp_us(x + 1, y1, z1));
			float u12 = (1.f - u) * (us(x, y1 + 1, z1) - tmp_us(x, y1 + 1, z1)) +
				u * (us(x + 1, y1 + 1, z1) - tmp_us(x + 1, y1 + 1, z1));
			float u13 = (1.f - u) * (us(x, y1, z1 + 1) - tmp_us(x, y1, z1 + 1)) +
				u * (us(x + 1, y1, z1 + 1) - tmp_us(x + 1, y1, z1 + 1));
			float u14 = (1.f - u) * (us(x, y1 + 1, z1 + 1) - tmp_us(x, y1 + 1, z1 + 1)) +
				u * (us(x + 1, y1 + 1, z1 + 1) - tmp_us(x + 1, y1 + 1, z1 + 1));
			float u21 = (1.f - v) * u11 + v * u12;
			float u22 = (1.f - v) * u13 + v * u14;
			float u31 = (1.f - w) * u21 + w * u22;
			pw.velocity.x += u31;

			// update v
			xstart = x1 * cellSize + bounds.xmin + 0.5f * cellSize;
			ystart = y * cellSize + bounds.ymin;
			zstart = z1 * cellSize + bounds.zmin + 0.5f * cellSize;
			u = (p.position.x - xstart) / cellSize;
			v = (p.position.y - ystart) / cellSize;
			w = (p.position.z - zstart) / cellSize;

			float v11 = (1.f - v) * (vs(x1, y, z1) - tmp_vs(x1, y, z1)) +
				v * (vs(x1, y + 1, z1) - tmp_vs(x1, y + 1, z1));
			float v12 = (1.f - v) * (vs(x1 + 1, y, z1) - tmp_vs(x1 + 1, y, z1)) +
				v * (vs(x1 + 1, y + 1, z1) - tmp_vs(x1 + 1, y + 1, z1));
			float v13 = (1.f - v) * (vs(x1, y, z1 + 1) - tmp_vs(x1, y, z1 + 1)) +
				v * (vs(x1, y + 1, z1 + 1) - tmp_vs(x1, y + 1, z1 + 1));
			float v14 = (1.f - v) * (vs(x1 + 1, y, z1 + 1) - tmp_vs(x1 + 1, y, z1 + 1)) +
				v * (vs(x1 + 1, y + 1, z1 + 1) - tmp_vs(x1 + 1, y + 1, z1 + 1));
			float v21 = (1.f - u) * v11 + u * v12;
			float v22 = (1.f - u) * v13 + u * v14;
			float v31 = (1.f - w) * v21 + w * v22;
			pw.velocity.y += v31;

			// update v
			xstart = x1 * cellSize + bounds.xmin + 0.5f * cellSize;
			ystart = y * cellSize + bounds.ymin;
			zstart = z1 * cellSize + bounds.zmin + 0.5f * cellSize;
			u = (p.position.x - xstart) / cellSize;
			v = (p.position.y - ystart) / cellSize;
			w = (p.position.z - zstart) / cellSize;

			float w11 = (1.f - w) * (ws(x1, y1, z) - tmp_ws(x1, y1, z)) +
				w * (ws(x1, y1, z + 1) - tmp_ws(x1, y1, z + 1));
			float w12 = (1.f - w) * (ws(x1 + 1, y1, z) - tmp_ws(x1 + 1, y1, z)) +
				w * (ws(x1 + 1, y1, z + 1) - tmp_ws(x1 + 1, y1, z + 1));
			float w13 = (1.f - w) * (ws(x1, y1 + 1, z) - tmp_ws(x1, y1 + 1, z)) +
				w * (ws(x1, y1 + 1, z + 1) - tmp_ws(x1, y1 + 1, z + 1));
			float w14 = (1.f - w) * (ws(x1 + 1, y1 + 1, z) - tmp_ws(x1 + 1, y1 + 1, z)) +
				w * (ws(x1 + 1, y1 + 1, z + 1) - tmp_ws(x1 + 1, y1 + 1, z + 1));
			float w21 = (1.f - u) * w11 + u * w12;
			float w22 = (1.f - u) * w13 + u * w14;
			float w31 = (1.f - v) * w21 + v * w22;
			pw.velocity.z += w31;

			pw.velocity *= 0.95f; // 95% FLIP
		}
	});
	//}
}


// PIC
void FS_MACGrid::interpolateVelocity()
{
	std::shared_ptr<std::vector<FS_Particle> > particlesWrite = (particles == particles1) ? particles2 : particles1;

	//for (int h = 0; h < particlesWrite->size(); ++h)
	//{
	//tbb::parallel_for(0, static_cast<int>(particlesWrite->size()), [&](int h) {
	tbb::parallel_for(tbb::blocked_range<int>(0, static_cast<int>(particlesWrite->size())), [&](const tbb::blocked_range<int> &r) {
		for (int h = r.begin(); h != r.end(); ++h)
		{
			FS_Particle &p = (*particlesWrite)[h];

			int x = floor((p.position.x - bounds.xmin) / cellSize);
			int y = floor((p.position.y - bounds.ymin) / cellSize);
			int z = floor((p.position.z - bounds.zmin) / cellSize);
			int x1 = floor((p.position.x - bounds.xmin - 0.5f * cellSize) / cellSize);
			int y1 = floor((p.position.y - bounds.ymin - 0.5f * cellSize) / cellSize);
			int z1 = floor((p.position.z - bounds.zmin - 0.5f * cellSize) / cellSize);

			// update u
			float xstart = x * cellSize + bounds.xmin;
			float ystart = y1 * cellSize + bounds.ymin + 0.5f * cellSize;
			float zstart = z1 * cellSize + bounds.zmin + 0.5f * cellSize;
			float u = (p.position.x - xstart) / cellSize;
			float v = (p.position.y - ystart) / cellSize;
			float w = (p.position.z - zstart) / cellSize;

			float u11 = (1.f - u) * us(x, y1, z1) + u * us(x + 1, y1, z1);
			float u12 = (1.f - u) * us(x, y1 + 1, z1) + u * us(x + 1, y1 + 1, z1);
			float u13 = (1.f - u) * us(x, y1, z1 + 1) + u * us(x + 1, y1, z1 + 1);
			float u14 = (1.f - u) * us(x, y1 + 1, z1 + 1) + u * us(x + 1, y1 + 1, z1 + 1);
			float u21 = (1.f - v) * u11 + v * u12;
			float u22 = (1.f - v) * u13 + v * u14;
			float u31 = (1.f - w) * u21 + w * u22;
			p.velocity.x += 0.05f * u31; // 5% PIC

			// update v
			xstart = x1 * cellSize + bounds.xmin + 0.5f * cellSize;
			ystart = y * cellSize + bounds.ymin;
			zstart = z1 * cellSize + bounds.zmin + 0.5f * cellSize;
			u = (p.position.x - xstart) / cellSize;
			v = (p.position.y - ystart) / cellSize;
			w = (p.position.z - zstart) / cellSize;

			float v11 = (1.f - v) * vs(x1, y, z1) + v * vs(x1, y + 1, z1);
			float v12 = (1.f - v) * vs(x1 + 1, y, z1) + v * vs(x1 + 1, y + 1, z1);
			float v13 = (1.f - v) * vs(x1, y, z1 + 1) + v * vs(x1, y + 1, z1 + 1);
			float v14 = (1.f - v) * vs(x1 + 1, y, z1 + 1) + v * vs(x1 + 1, y + 1, z1 + 1);
			float v21 = (1.f - u) * v11 + u * v12;
			float v22 = (1.f - u) * v13 + u * v14;
			float v31 = (1.f - w) * v21 + w * v22;
			p.velocity.y += 0.05f * v31;

			// update v
			xstart = x1 * cellSize + bounds.xmin + 0.5f * cellSize;
			ystart = y * cellSize + bounds.ymin;
			zstart = z1 * cellSize + bounds.zmin + 0.5f * cellSize;
			u = (p.position.x - xstart) / cellSize;
			v = (p.position.y - ystart) / cellSize;
			w = (p.position.z - zstart) / cellSize;

			float w11 = (1.f - w) * ws(x1, y1, z) + w * ws(x1, y1, z + 1);
			float w12 = (1.f - w) * ws(x1 + 1, y1, z) + w * ws(x1 + 1, y1, z + 1);
			float w13 = (1.f - w) * ws(x1, y1 + 1, z) + w * ws(x1, y1 + 1, z + 1);
			float w14 = (1.f - w) * ws(x1 + 1, y1 + 1, z) + w * ws(x1 + 1, y1 + 1, z + 1);
			float w21 = (1.f - u) * w11 + u * w12;
			float w22 = (1.f - u) * w13 + u * w14;
			float w31 = (1.f - v) * w21 + v * w22;
			p.velocity.z += 0.05f * w31;
		}
	});
	//}
}


bool FS_MACGrid::averageVelocityFromNeighbours(int i, int j, int k)
{
	int fluidCellCount = 0;
	glm::vec3 extrapolatedVelocity(0.f);

	for (int di = -1; di <= 1; ++di)
	{
		for (int dj = -1; dj <= 1; ++dj)
		{
			for (int dk = -1; dk <= 1; ++dk)
			{
				if (!di && !dj && !dk) // self
				{
					continue;
				}

				int ni = i + di;
				int nj = j + dj;
				int nk = k + dk;

				if (ni < 0 || ni >= xcount || nj < 0 || nj >= ycount || nk < 0 || nk >= zcount)
				{
					continue; // out of bound
				}

				if (cellTypes(ni, nj, nk) != 1) // neighbour is not fluid
				{
					continue;
				}

				++fluidCellCount;
				extrapolatedVelocity += glm::vec3(us(ni, nj, nk), vs(ni, nj, nk), ws(ni, nj, nk));
			}
		}
	}

	if (fluidCellCount == 0)
	{
		us(i, j, k) = 0.f;
		vs(i, j, k) = 0.f;
		ws(i, j, k) = 0.f;
		return false;
	}

	extrapolatedVelocity /= static_cast<float>(fluidCellCount);

	if (i > 0 && i < xcount) // not boundary
	{
		us(i, j, k) = extrapolatedVelocity.x;
	}
	if (j > 0 && j < ycount) // not boundary
	{
		vs(i, j, k) = extrapolatedVelocity.y;
	}
	if (k > 0 && k < zcount) // not boundary
	{
		ws(i, j, k) = extrapolatedVelocity.z;
	}
}


void FS_MACGrid::extrapolateVelocity(bool onlyBoundary)
{
	//for (int i = 0; i < xcount; ++i)
	//{
	//	for (int j = 0; j < ycount; ++j)
	//	{
	//		for (int k = 0; k < zcount; ++k)
	//		{
	tbb::parallel_for(tbb::blocked_range3d<int>(-1, xcount + 1, -1, ycount + 1, -1, zcount + 1), [&](const tbb::blocked_range3d<int> &r) {
		for (int i = r.pages().begin(); i != r.pages().end(); ++i)
		{
			for (int j = r.rows().begin(); j != r.rows().end(); ++j)
			{
				for (int k = r.cols().begin(); k != r.cols().end(); ++k)
				{
					if ((!onlyBoundary && cellTypes(i, j, k) == 0) || i == -1 || i == xcount || j == -1 || j == ycount || k == -1 || k == zcount)
					{
						averageVelocityFromNeighbours(i, j, k);
					}
				}
			}
		}
	});
	//		}
	//	}
	//}
}


void FS_MACGrid::setup()
{
	if (!bufferGenerated)
	{
		glGenVertexArrays(1, &debugVAO);
		glBindVertexArray(debugVAO);

		// Shaders and shader program
		std::string vsFileName(SHADERS_DIR);
		std::string fsFileName(SHADERS_DIR);
		vsFileName += "/boxSource_vs.glsl";
		fsFileName += "/boxSource_fs.glsl";
		GLuint vsn = loadShader(vsFileName.c_str(), GL_VERTEX_SHADER);
		GLuint fsn = loadShader(fsFileName.c_str(), GL_FRAGMENT_SHADER);

		debugProgram = glCreateProgram();
		glAttachShader(debugProgram, vsn);
		glAttachShader(debugProgram, fsn);
		glLinkProgram(debugProgram);

		glDeleteShader(vsn);
		glDeleteShader(fsn);

		// Check link status
		GLint result;
		int infoLogLength;
		glGetProgramiv(debugProgram, GL_LINK_STATUS, &result);
		glGetProgramiv(debugProgram, GL_INFO_LOG_LENGTH, &infoLogLength);
		if (infoLogLength > 0) {
			std::vector<char> msg(infoLogLength + 1);
			glGetProgramInfoLog(debugProgram, infoLogLength, NULL, &msg[0]);
			std::cout << msg.data() << '\n';
		}
		if (!result)
		{
			exit(EXIT_FAILURE);
		}

		glGenBuffers(1, &debugVBO1);
		glBindBuffer(GL_ARRAY_BUFFER, debugVBO1);
		glBufferData(GL_ARRAY_BUFFER, particles->size() * sizeof(FS_Particle), particles->data(), GL_DYNAMIC_READ);

		glGenBuffers(1, &debugVBO2);
		glBindBuffer(GL_ARRAY_BUFFER, debugVBO2);
		glBufferData(GL_ARRAY_BUFFER, particles->size() * sizeof(FS_Particle), particles->data(), GL_STATIC_DRAW);

		debugGrid.clear();
		for (int i = 0; i < xcount; ++i)
		{
			for (int j = 0; j < ycount; ++j)
			{
				for (int k = 0; k < zcount; ++k)
				{
					float xstart = bounds.xmin + i * cellSize;
					float ystart = bounds.ymin + j * cellSize;
					float zstart = bounds.zmin + k * cellSize;
					glm::vec3 v1(xstart, ystart, zstart);
					glm::vec3 v2(xstart + cellSize, ystart, zstart);
					glm::vec3 v3(xstart + cellSize, ystart + cellSize, zstart);
					glm::vec3 v4(xstart, ystart + cellSize, zstart);
					glm::vec3 v5(xstart, ystart, zstart + cellSize);
					glm::vec3 v6(xstart + cellSize, ystart, zstart + cellSize);
					glm::vec3 v7(xstart + cellSize, ystart + cellSize, zstart + cellSize);
					glm::vec3 v8(xstart, ystart + cellSize, zstart + cellSize);
					
					debugGrid.push_back(v1.x); debugGrid.push_back(v1.y); debugGrid.push_back(v1.z);
					debugGrid.push_back(v2.x); debugGrid.push_back(v2.y); debugGrid.push_back(v2.z);

					debugGrid.push_back(v2.x); debugGrid.push_back(v2.y); debugGrid.push_back(v2.z);
					debugGrid.push_back(v3.x); debugGrid.push_back(v3.y); debugGrid.push_back(v3.z);

					debugGrid.push_back(v3.x); debugGrid.push_back(v3.y); debugGrid.push_back(v3.z);
					debugGrid.push_back(v4.x); debugGrid.push_back(v4.y); debugGrid.push_back(v4.z);

					debugGrid.push_back(v4.x); debugGrid.push_back(v4.y); debugGrid.push_back(v4.z);
					debugGrid.push_back(v1.x); debugGrid.push_back(v1.y); debugGrid.push_back(v1.z);

					debugGrid.push_back(v1.x); debugGrid.push_back(v1.y); debugGrid.push_back(v1.z);
					debugGrid.push_back(v5.x); debugGrid.push_back(v5.y); debugGrid.push_back(v5.z);

					debugGrid.push_back(v2.x); debugGrid.push_back(v2.y); debugGrid.push_back(v2.z);
					debugGrid.push_back(v6.x); debugGrid.push_back(v6.y); debugGrid.push_back(v6.z);

					debugGrid.push_back(v3.x); debugGrid.push_back(v3.y); debugGrid.push_back(v3.z);
					debugGrid.push_back(v7.x); debugGrid.push_back(v7.y); debugGrid.push_back(v7.z);

					debugGrid.push_back(v4.x); debugGrid.push_back(v4.y); debugGrid.push_back(v4.z);
					debugGrid.push_back(v8.x); debugGrid.push_back(v8.y); debugGrid.push_back(v8.z);

					debugGrid.push_back(v5.x); debugGrid.push_back(v5.y); debugGrid.push_back(v5.z);
					debugGrid.push_back(v6.x); debugGrid.push_back(v6.y); debugGrid.push_back(v6.z);

					debugGrid.push_back(v6.x); debugGrid.push_back(v6.y); debugGrid.push_back(v6.z);
					debugGrid.push_back(v7.x); debugGrid.push_back(v7.y); debugGrid.push_back(v7.z);

					debugGrid.push_back(v7.x); debugGrid.push_back(v7.y); debugGrid.push_back(v7.z);
					debugGrid.push_back(v8.x); debugGrid.push_back(v8.y); debugGrid.push_back(v8.z);

					debugGrid.push_back(v8.x); debugGrid.push_back(v8.y); debugGrid.push_back(v8.z);
					debugGrid.push_back(v5.x); debugGrid.push_back(v5.y); debugGrid.push_back(v5.z);
				}
			}
		}

		glGenBuffers(1, &debugVBO3);
		glBindBuffer(GL_ARRAY_BUFFER, debugVBO3);
		glBufferData(GL_ARRAY_BUFFER, xcount * ycount * zcount * 24 * 3 * sizeof(float), debugGrid.data(), GL_STATIC_DRAW);

		glGenBuffers(1, &debugVBO4);
		glBindBuffer(GL_ARRAY_BUFFER, debugVBO4);
		glBufferData(GL_ARRAY_BUFFER, xcount * ycount * zcount * 12 * 3 * sizeof(float), NULL, GL_STATIC_DRAW);

		glBindVertexArray(0);
		bufferGenerated = true;
	}
}


void FS_MACGrid::render(std::shared_ptr<FS_Camera> pCam)
{
	glBindVertexArray(debugVAO);

	glBindBuffer(GL_COPY_READ_BUFFER, debugVBO1);
	float *mapped = reinterpret_cast<float *>(glMapBufferRange(GL_COPY_READ_BUFFER, 0, particles->size() * sizeof(FS_Particle), GL_MAP_READ_BIT));
	memcpy(&(*particles)[0], mapped, particles->size() * sizeof(FS_Particle));
	glUnmapBuffer(GL_COPY_READ_BUFFER);

	//buildSDF(); // uncomment to enable writing VDB files

	// particles
	glBindBuffer(GL_ARRAY_BUFFER, debugVBO2);
	glBufferSubData(GL_ARRAY_BUFFER, 0, particles->size() * sizeof(FS_Particle), particles->data());
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(FS_Particle), 0);

	glUseProgram(debugProgram);
	glm::mat4 MVP = pCam->proj * pCam->view; // model matrix is identity
	GLint unif_mvp = glGetUniformLocation(debugProgram, "MVP");
	glUniformMatrix4fv(unif_mvp, 1, GL_FALSE, &MVP[0][0]);

	GLint uniColorLoc = glGetUniformLocation(debugProgram, "uniColor");
	glUniform3f(uniColorLoc, 0.f, 0.f, 1.f);

	glPointSize(4);
	glDrawArrays(GL_POINTS, 0, particles->size());
	glPointSize(1);

	// velocity
	//debugVelocity.clear();

	//for (int i = 0; i < xcount + 1; ++i)
	//{
	//	for (int j = 0; j < ycount; ++j)
	//	{
	//		for (int k = 0; k < zcount; ++k)
	//		{
	//			float xstart = bounds.xmin + i * cellSize;
	//			float ystart = bounds.ymin + j * cellSize + 0.5f * cellSize;
	//			float zstart = bounds.zmin + k * cellSize + 0.5f * cellSize;
	//			float u = us(i, j, k);

	//			debugVelocity.push_back(xstart);
	//			debugVelocity.push_back(ystart);
	//			debugVelocity.push_back(zstart);
	//			debugVelocity.push_back(xstart + u);
	//			debugVelocity.push_back(ystart);
	//			debugVelocity.push_back(zstart);
	//		}
	//	}
	//}

	//for (int i = 0; i < xcount; ++i)
	//{
	//	for (int j = 0; j < ycount + 1; ++j)
	//	{
	//		for (int k = 0; k < zcount; ++k)
	//		{
	//			float xstart = bounds.xmin + i * cellSize + 0.5f * cellSize;
	//			float ystart = bounds.ymin + j * cellSize;
	//			float zstart = bounds.zmin + k * cellSize + 0.5f * cellSize;
	//			float v = vs(i, j, k);

	//			debugVelocity.push_back(xstart);
	//			debugVelocity.push_back(ystart);
	//			debugVelocity.push_back(zstart);
	//			debugVelocity.push_back(xstart);
	//			debugVelocity.push_back(ystart + v);
	//			debugVelocity.push_back(zstart);
	//		}
	//	}
	//}

	//for (int i = 0; i < xcount; ++i)
	//{
	//	for (int j = 0; j < ycount; ++j)
	//	{
	//		for (int k = 0; k < zcount + 1; ++k)
	//		{
	//			float xstart = bounds.xmin + i * cellSize + 0.5f * cellSize;
	//			float ystart = bounds.ymin + j * cellSize + 0.5f * cellSize;
	//			float zstart = bounds.zmin + k * cellSize;
	//			float w = ws(i, j, k);

	//			debugVelocity.push_back(xstart);
	//			debugVelocity.push_back(ystart);
	//			debugVelocity.push_back(zstart);
	//			debugVelocity.push_back(xstart);
	//			debugVelocity.push_back(ystart);
	//			debugVelocity.push_back(zstart + w);
	//		}
	//	}
	//}

	//glBindBuffer(GL_ARRAY_BUFFER, debugVBO4);
	//glBufferSubData(GL_ARRAY_BUFFER, 0, debugVelocity.size() * sizeof(float), debugVelocity.data());
	//glEnableVertexAttribArray(0);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

	//glUniform3f(uniColorLoc, 1.f, 0.f, 0.f);

	//glLineWidth(2);
	//glDrawArrays(GL_LINES, 0, debugVelocity.size());
	//glLineWidth(1);

	// grid
	//glBindBuffer(GL_ARRAY_BUFFER, debugVBO3);
	//glEnableVertexAttribArray(0);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

	//glUniform3f(uniColorLoc, 0.f, 0.f, 0.f);

	//glDrawArrays(GL_LINES, 0, debugGrid.size());

	// indicators
	//drawCellTypeIndicators(uniColorLoc);

	//glBindVertexArray(0);
}


void FS_MACGrid::cleanup()
{
	glDeleteBuffers(1, &debugVBO1);
	glDeleteBuffers(1, &debugVBO2);
	glDeleteBuffers(1, &debugVBO3);
	glDeleteBuffers(1, &debugVBO4);
	glDeleteBuffers(1, &debugVAO);
}

void FS_MACGrid::updateCellTypeDebugBuffer()
{
	airIndicators.clear();
	fluidIndicators.clear();
	solidIndicators.clear();

	for (int x = 0; x < xcount; ++x)
	{
		for (int y = 0; y < ycount; ++y)
		{
			for (int z = 0; z < zcount; ++z)
			{
				float cpx = bounds.xmin + x * cellSize + 0.5f * cellSize;
				float cpy = bounds.ymin + y * cellSize + 0.5f * cellSize;
				float cpz = bounds.zmin + z * cellSize + 0.5f * cellSize;
				int ct = cellTypes(x, y, z);

				if (ct == 0) // air
				{
					airIndicators.push_back(cpx);
					airIndicators.push_back(cpy);
					airIndicators.push_back(cpz);
				}
				else if (ct == 1) // fluid
				{
					fluidIndicators.push_back(cpx);
					fluidIndicators.push_back(cpy);
					fluidIndicators.push_back(cpz);
				}
				else // solid
				{
					solidIndicators.push_back(cpx);
					solidIndicators.push_back(cpy);
					solidIndicators.push_back(cpz);
				}
			}
		}
	}

	if (!indicatorBufferGenerated)
	{
		glGenBuffers(1, &airVertexBufferName);

		glGenBuffers(1, &fluidVertexBufferName);

		glGenBuffers(1, &solidVertexBufferName);
		indicatorBufferGenerated = true;
	}

	if (airIndicators.size() > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, airVertexBufferName);
		glBufferData(GL_ARRAY_BUFFER, airIndicators.size() * sizeof(float), NULL, GL_STREAM_DRAW); // orphaning
		glBufferSubData(GL_ARRAY_BUFFER, 0, airIndicators.size() * sizeof(float), airIndicators.data());
	}

	if (fluidIndicators.size() > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, fluidVertexBufferName);
		glBufferData(GL_ARRAY_BUFFER, fluidIndicators.size() * sizeof(float), NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, fluidIndicators.size() * sizeof(float), fluidIndicators.data());
	}

	if (solidIndicators.size() > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, solidVertexBufferName);
		glBufferData(GL_ARRAY_BUFFER, solidIndicators.size() * sizeof(float), NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, solidIndicators.size() * sizeof(float), solidIndicators.data());
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0); // unbind
}


void FS_MACGrid::drawCellTypeIndicators(GLint uniColorLoc)
{
	updateCellTypeDebugBuffer();

	glPointSize(6.f);

	if (airIndicators.size() > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, airVertexBufferName);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
		glUniform3f(uniColorLoc, 1.f, 1.f, 1.f);
		glDrawArrays(GL_POINTS, 0, airIndicators.size() / 3);
	}

	if (fluidIndicators.size() > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, fluidVertexBufferName);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
		glUniform3f(uniColorLoc, 0.f, 1.f, 1.f);
		glDrawArrays(GL_POINTS, 0, fluidIndicators.size() / 3);
	}

	if (solidIndicators.size() > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, solidVertexBufferName);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
		glUniform3f(uniColorLoc, 1.f, 1.f, 0.f);
		glDrawArrays(GL_POINTS, 0, solidIndicators.size() / 3);
	}

	glPointSize(1.f);
}


//class VDBParticleList
//{
//public:
//	typedef openvdb::Vec3R  value_type;
//
//	std::shared_ptr<std::vector<FS_Particle> > particles;
//	openvdb::Real radius;
//
//	VDBParticleList(std::shared_ptr<std::vector<FS_Particle> > ps, openvdb::Real r)
//		: particles(ps), radius(r) {}
//
//	size_t size() const { return particles->size(); }
//
//	void getPos(size_t n, openvdb::Vec3R &pos) const
//	{
//		std::vector<FS_Particle> &ps = *particles;
//		pos.x() = ps[n].position.x;
//		pos.y() = ps[n].position.y;
//		pos.z() = ps[n].position.z;
//	}
//
//	void getPosRad(size_t n, openvdb::Vec3R& pos, openvdb::Real& rad) const
//	{
//		getPos(n, pos);
//		rad = radius;
//	}
//
//	void getPosRadVel(size_t n, openvdb::Vec3R& pos, openvdb::Real& rad, openvdb::Vec3R& vel) const
//	{
//		getPosRad(n, pos, rad);
//		std::vector<FS_Particle> &ps = *particles;
//		vel.x() = ps[n].velocity.x;
//		vel.y() = ps[n].velocity.y;
//		vel.z() = ps[n].velocity.z;
//	}
//};
//
//
//void FS_MACGrid::buildSDF()
//{
//	static int frameCount = 0;
//
//	VDBParticleList pls(particles, cellSize * .5f);
//
//	const float voxelSize = cellSize * .25f, halfWidth = 6.f;
//	openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);
//	openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid> raster(*ls);
//
//	raster.setGrainSize(particles->size() / 16); //a value of zero disables threading
//	raster.rasterizeSpheres(pls);
//	raster.finalize();
//
//	//size_t test1 = raster.getMinCount();
//	//size_t test2 = raster.getMaxCount();
//
//	ls->setName("FluidSurfaceLevelSet");
//	openvdb::GridPtrVec grids;
//	grids.push_back(ls);
//	std::stringstream ss;
//	ss << "C:/Users/Jian Ru/Documents/CIS563/fluidsolver/CIS563-FluidSolver/vdbfiles/"
//		<< "FluidSurfaceLevelSet_" << frameCount << ".vdb";
//	openvdb::io::File file(ss.str().c_str());
//	file.write(grids);
//	file.close();
//	++frameCount;
//}