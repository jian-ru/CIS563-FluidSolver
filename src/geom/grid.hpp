#ifndef GRID_HPP
#define GRID_HPP

#include <vector>
#include <random>

#include "geom.hpp"

template <class T>
class FS_Grid
{
public:
	std::vector<T> values;
	int xsize, ysize, zsize;

	FS_Grid() {}
	FS_Grid(int xcount, int ycount, int zcount);
	
	virtual ~FS_Grid() {}

	void resize(int xcount, int ycount, int zcount);

	void zeromem();

	inline T &operator()(int i, int j, int k) { return values[(i + 1) * ysize * zsize + (j + 1) * zsize + (k + 1)]; }
};


class FS_MACGrid : public FS_Geometry
{
public:
	FS_Grid<float> us, vs, ws, ps;
	FS_Grid<float> tmp_us, tmp_vs, tmp_ws;
	FS_Grid<int> cellTypes; // 0 - air, 1 - fluid, 2 - solid
	std::shared_ptr<std::vector<FS_Particle> > particles;
	std::shared_ptr<std::vector<FS_Particle> > particles1;
	std::shared_ptr<std::vector<FS_Particle> > particles2;
	glm::vec3 maxVelocity;
	
	FS_BBox bounds;
	float cellSize;
	int xcount, ycount, zcount;

	std::uniform_real_distribution<float> rng;
	std::mt19937 generator;

	bool bufferGenerated = false;
	std::vector<float> debugGrid;
	std::vector<float> debugVelocity;
	GLuint debugVAO, debugVBO1, debugVBO2, debugVBO3, debugVBO4;
	GLuint debugProgram;

	GLuint buff_tmp[12]; // for gpu pressure solve


	FS_MACGrid(const FS_BBox &b, float csize);

	virtual ~FS_MACGrid() {}

	// fill the grid with particles
	void init();

	void transferParticleVelocityToGrid();

	// From grid to particle (PIC)
	void interpolateVelocity();

	// Call this before interpolateVelocity()
	// Also copy updated positions
	// FLIP
	void interpolateVelocityDifference();

	void saveVelocities();

	void accelerateByGravity(float deltaTime, float amount = -9.8f);

	void extrapolateVelocity();
	// return false if (i, j, k) has no fluid neighbour
	bool averageVelocityFromNeighbours(int i, int j, int k);

	void swapActiveParticleArray();

	void updatePressureAndVelocity(float deltaTime);

	void gpu_updatePressureAndVelocity(float deltaTime);

	float findMaxSpeed();

	// sample the grid and build a SDF
	// write the resulting SDF as a VDB file
	void buildSDF();

	virtual void setup();
	virtual void render(std::shared_ptr<FS_Camera> pCam);
	virtual void cleanup();

	// for debug drawing of cell types
	bool indicatorBufferGenerated;
	GLuint airVertexBufferName, fluidVertexBufferName, solidVertexBufferName;
	std::vector<float> airIndicators, fluidIndicators, solidIndicators; // position

	void updateCellTypeDebugBuffer();
	void drawCellTypeIndicators(GLint uniColorLoc);
};

#endif // GRIP_HPP