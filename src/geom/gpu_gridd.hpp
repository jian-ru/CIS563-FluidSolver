#ifndef GPU_GRID_HPP
#define GPU_GRID_HPP

#include "../main.hpp"
#include <random>
#include <chrono>
#include "geom.hpp"


class GPUOPERATIONS;


// The grid always starts at (0.0, 0.0, 0.0) in the world
// The uVels, vVels, wVels, pressures, and cellTypes are padded at borders so they have the same size
class FS_GPU_MACGRID : public FS_Geometry
{
public:
	std::shared_ptr<GPUOPERATIONS> gpuops;

	// Meta data
	const int segmentSize = 2048;
	int numSamples1D; // how many times to sample one dimemsion of a cell when seeding particles
	int numParticles;
	int numCells;
	int xcount, ycount, zcount;
	float cellSize;
	float maxDeltaTime, minDeltaTime;
	float kCFL;

	// xcount - int, 4 bytes
	// ycount - int, 4 bytes
	// zcount - int, 4 bytes
	// cellSize - float, 4 bytes
	// numParticles - int, 4 bytes
	// numCells - int, 4 bytes
	GLuint cbMeta;

	// Particle data on GPU
	GLuint particles;

	// GPU cell buffers
	GLuint uVels1, uVels2;
	GLuint vVels1, vVels2;
	GLuint wVels1, wVels2;
	GLuint pressures;
	GLuint cellTypes;

	// GPU buffers and programs used in transferParticleVelocitiesToGrid()
	GLuint countNumParticleInCellsProgram;
	GLuint buildParticleIndexBufferProgram;
	GLuint gatherParticleVelocitiesProgram;
	GLuint saveVelocitiesProgram;
	GLuint uParticleCounts, vParticleCounts, wParticleCounts; // how many particles influence each cell
	GLuint uParticleIndexOffsets, vParticleIndexOffsets, wParticleIndexOffsets;
	GLuint uParticleIndexBuffer, vParticleIndexBuffer, wParticleIndexBuffer;
	GLuint uParticleCurCounts, vParticleCurCounts, wParticleCurCounts; // current number particle indices in a cell

	// GPU buffers and programs used in accelerateByGravity()
	GLuint accByGravityProgram;

	// GPU buffers and programs used for pressure solve
	const int entrySize = 16; // in bytes
	const int bucketSize = 2;
	int numBuckets;
	GLuint cbHashMap; // numBuckets - int, 4 bytes; bucketSize - int, 4bytes
	GLuint bucketLocks;
	GLuint cellIdxToColNumMap; // A is SPD so rowNum == colNum
	GLuint diagA, one_diagA, offDiagA, offsetsSizesBuffer, colNums;
	GLuint x, b;
	GLuint ax, r, d, q, s;
	GLuint numFluidCells;
	int numFluidCells_CPU;
	GLuint buildCellIdxToColNumHashMapProgram;
	GLuint buildPressureSolveLinearSystemProgram;
	GLuint transferPressureBackToGridProgram;
	GLuint updateVelocitiesUsingPressureGradientsProgram;

	// GPU buffers and programs used for transferGridVelocitiesToParticles()
	GLuint transferGridVelocitiesToParticlesProgram;

	// GPU buffers and programs used for advectParticles()
	GLuint advectParticlesProgram;

	// GPU buffers and programs for extrapolateGridVelocities()
	GLuint extrapolateGridVelocitiesProgram;

	// GPU buffers and programs for findMaxParticleSpeed()
	float cpuMaxParticleSpeed;
	GLuint inSpeedBuffer;
	GLuint outSpeedBuffer;
	GLuint numGlobalIterationsBuffer;
	GLuint findMaxParticleSpeedProgram;

	// Random number generators
	std::uniform_real_distribution<float> rng;
	std::mt19937 generator;

	// GPU buffers and programs used for rendering
	GLuint vao;
	GLuint renderProgram;


	FS_GPU_MACGRID(int xc, int yc, int zc, float cs, int ns1d = 2,
		float maxDT = 1.f/24.f, float minDT = 1.f/60.f, float kcfl = .5f);

	virtual ~FS_GPU_MACGRID();

	void updateGridsandParticles()
	{
		transferParticleVelocitiesToGrid();
		accelerateByGravity();
		pressureSolve();
		extrapolateGridVelocities();
		transferGridVelocitiesToParticles();
		findMaxParticleSpeed();
		advectParticles();
	}

	virtual void setup() {}
	virtual void render(std::shared_ptr<FS_Camera> pCam);
	virtual void cleanup() {}

private:
	// main steps
	void transferParticleVelocitiesToGrid();

	void accelerateByGravity();

	void pressureSolve();

	void transferGridVelocitiesToParticles();

	void advectParticles();

	void extrapolateGridVelocities();

	void findMaxParticleSpeed();

	// Supporting methods
	void buildPressureSolveLinearSystem();

	void seedParticles();

	void genGPUBuffers();

	void genShaderPrograms();

	void setCbMeta();

	void setDeltaTime();

	void setCbHashMap();
};

#endif // GPU_GRID_HPP