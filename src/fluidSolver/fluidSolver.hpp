//
//  fluidSolver.hpp
//  Thanda

#ifndef fluidSolver_hpp
#define fluidSolver_hpp

#include <Windows.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <memory>

#include "../geom/grid.hpp"


class FS_Solver
{
public:
	virtual ~FS_Solver() {}

	// vbo should contain data of all particles packed together
	// additionalInfo may contain other useful info (e.g. boundaries) depend on the requirement of a specific solver
	virtual void solve(float deltaTime, GLuint vbo, void *additionalInfo) = 0;
};


struct FS_TestSolverInfo
{
	int numParticles;
	float gravity;
	float bounds[6];
	GLint isOpen[6];
	std::shared_ptr<FS_MACGrid> grid;
};


// Considers gravity and simple AABB collision
class FS_TestSolver : public FS_Solver
{
public:
	FS_TestSolver();
	
	virtual void solve(float deltaTime, GLuint vbo, void *additionalInfo);

private:
	GLuint program; // compute program
};

#endif /* fluidSolver_hpp */
