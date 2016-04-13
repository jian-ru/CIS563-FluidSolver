//
//  fluidSolver.cpp
//  Thanda


#include <string>
#include <vector>
#include <iostream>
#include "fluidSolver.hpp"
#include "../main.hpp"


FS_TestSolver::FS_TestSolver()
{
	// Shaders and shader program
	std::string csFileName(SHADERS_DIR);
	csFileName += "/testSolver_cs.glsl";
	GLuint csn = loadShader(csFileName.c_str(), GL_COMPUTE_SHADER);

	program = glCreateProgram();
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
}


void FS_TestSolver::solve(float deltaTime, GLuint vbo, void * additionalInfo)
{
	FS_TestSolverInfo *info = reinterpret_cast<FS_TestSolverInfo *>(additionalInfo);

	info->grid->transferParticleVelocityToGrid();
	info->grid->extrapolateVelocity();
	info->grid->saveVelocities();
	info->grid->accelerateByGravity(deltaTime);
	// pressure solve and update velocities according to pressure gradients
	//info->grid->updatePressureAndVelocity(deltaTime);
	info->grid->gpu_updatePressureAndVelocity(deltaTime);
	//info->grid->extrapolateVelocity(); // Don't extrapolate here. It will ruin divergence free velocity field
	info->grid->interpolateVelocityDifference(); // FLIP
	info->grid->interpolateVelocity(); // PIC
	info->grid->swapActiveParticleArray();

	glUseProgram(program);
	
	// Set uniforms
	GLint dt_loc = glGetUniformLocation(program, "deltaTime");
	GLint g_loc = glGetUniformLocation(program, "gravity");
	GLint b_loc = glGetUniformLocation(program, "bounds");
	GLint io_loc = glGetUniformLocation(program, "isOpen");
	glUniform1f(dt_loc, deltaTime);
	glUniform3f(g_loc, 0.f, info->gravity, 0.f);
	glUniform1fv(b_loc, 6, info->bounds);
	glUniform1iv(io_loc, 6, info->isOpen);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, vbo);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, info->grid->particles->size() * sizeof(FS_Particle), info->grid->particles->data());

	// Set SSBO
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vbo);

	int numGroups = std::ceil(info->numParticles / 256.0);
	glDispatchCompute(numGroups, 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFinish();
}
