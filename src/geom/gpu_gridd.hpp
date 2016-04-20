#ifndef GPU_GRID_HPP
#define GPU_GRID_HPP

#include <random>
#include <chrono>
#include "../main.hpp"
#include "geom.hpp"

// The grid always starts at (0.0, 0.0, 0.0) in the world
// The uVels, vVels, wVels, pressures, and cellTypes are padded at borders so they have the same size
class FS_GPU_MACGRID
{
public:
	// Meta data
	int numSamples1D; // how many times to sample one dimemsion of a cell when seeding particles
	int numParticles;
	int numCells;
	int xcount, ycount, zcount;
	float cellSize;

	// Particle data on GPU
	GLuint particles;

	// GPU cell buffers
	GLuint uVels1, uVels2;
	GLuint vVels1, vVels2;
	GLuint wVels1, wVels2;
	GLuint pressures;
	GLuint cellTypes;

	// Random number generators
	std::uniform_real_distribution<float> rng;
	std::mt19937 generator;


	FS_GPU_MACGRID(int xc, int yc, int zc, float cs, int ns1d = 2)
		: numSamples1D(ns1d)
	{
		xcount = xc + 2;
		ycount = yc + 2;
		zcount = zc + 2;
		cellSize = cs;
		numCells = xcount * ycount * zcount;

		rng = std::uniform_real_distribution<float>(0.f, 1.f);
		generator = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());

		seedParticles();
		genGPUBuffers();
	}

	virtual ~FS_GPU_MACGRID()
	{
		glDeleteBuffers(1, &particles);
		glDeleteBuffers(1, &uVels1);
		glDeleteBuffers(1, &uVels2);
		glDeleteBuffers(1, &vVels1);
		glDeleteBuffers(1, &vVels2);
		glDeleteBuffers(1, &wVels1);
		glDeleteBuffers(1, &wVels2);
		glDeleteBuffers(1, &pressures);
		glDeleteBuffers(1, &cellTypes);
	}

	void transferParticleVelocitiesToGrid();

private:
	void seedParticles();

	void genGPUBuffers();
};

#endif // GPU_GRID_HPP