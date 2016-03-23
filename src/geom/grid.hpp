#ifndef GRID_HPP
#define GRID_HPP

#include <vector>
#include <random>

#include "geom.hpp"


class FS_Grid
{
public:
	std::vector<float> values;
	int xsize, ysize, zsize;

	FS_Grid() {}
	FS_Grid(int xcount, int ycount, int zcount);
	
	virtual ~FS_Grid() {}

	void resize(int xcount, int ycount, int zcount);

	void zeromem();

	float &operator()(int i, int j, int k);
};


class FS_MACGrid : public FS_Geometry
{
public:
	FS_Grid us, vs, ws, ps;
	std::vector<FS_Particle> particles;
	
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


	FS_MACGrid(const FS_BBox &b, float csize);

	virtual ~FS_MACGrid() {}

	void init();

	void transferParticleVelocityToGrid();

	// From grid to particle;
	void interpolateVelocity();

	virtual void setup();
	virtual void render(std::shared_ptr<FS_Camera> pCam);
	virtual void cleanup();
};

#endif // GRIP_HPP