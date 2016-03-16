#ifndef BOX_CONTAINER_HPP
#define BOX_CONTAINER_HPP

#include "glm/glm.hpp"
#include "geom.hpp"
#include <vector>


class FS_BoxContainer : public FS_Geometry
{
public:
	// Default size is (1, 1, 1)
	FS_BoxContainer(glm::vec3 &pos, glm::vec3 &scale_xyz, bool open[6] = { false });
	virtual ~FS_BoxContainer() {}

	virtual void render(std::shared_ptr<FS_Camera> pCam);
	virtual void cleanup();

	float minX, maxX;
	float minY, maxY;
	float minZ, maxZ;
	bool isOpen[6]; // -x, +x, -y, +y, -z, +z

private:
	virtual void setup();

	GLuint vao, vbo, ibo;
	GLuint program;
	std::vector<float> vertexBuffer; // positions
	std::vector<unsigned int> indexBuffer;
};

#endif // BOX_CONTAINER_HPP