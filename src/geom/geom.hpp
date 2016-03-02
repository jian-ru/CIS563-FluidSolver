//
//  geom.hpp
//  Thanda

#ifndef GEOM_HPP
#define GEOM_HPP

#include <Windows.h>
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "../camera/camera.hpp"


// Make sure this struct is padded to 4-byte boundary (no gap)
struct FS_Particle
{
	glm::vec3 position;
	glm::vec3 velocity;
};


class FS_Geometry
{
public:
	virtual ~FS_Geometry() {}

	virtual void setup() = 0;
	virtual void render(std::shared_ptr<FS_Camera> pCam) = 0;
	virtual void cleanup() = 0;
};

#endif /* geom_hpp */
