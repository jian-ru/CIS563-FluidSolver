//
//  main.hpp
//  Thanda

#ifndef main_hpp
#define main_hpp

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <Windows.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range3d.h>
#include <tbb/mutex.h>
#include <tbb/queuing_mutex.h>


#define _GPU_INFO

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define SCENE_FILE_NAME "C:/Users/Jian Ru/Documents/CIS563/fluidsolver/CIS563-FluidSolver/src/scene/scene.json"
#define SHADERS_DIR "C:/Users/Jian Ru/Documents/CIS563/fluidsolver/CIS563-FluidSolver/src/shaders"


// Don't forget to delete it after attaching to shader program(s)
GLuint loadShader(const char *fileName, GLenum type);

bool readWholeFile(const char *fileName, std::string &ret_content);

#endif /* main_hpp */

