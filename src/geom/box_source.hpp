#ifndef BOX_SOURCE_HPP
#define BOX_SOURCE_HPP

#include <vector>
#include "geom.hpp"


class FS_BoxSource : public FS_Geometry
{
public:
	// Base size if (1, 1, 1)
	FS_BoxSource(glm::vec3 &pos, glm::vec3 &scale_xyz, float p_sep);
	virtual ~FS_BoxSource() {}

	virtual void render(std::shared_ptr<FS_Camera> pCam);
	virtual void cleanup();

	std::vector<FS_Particle> baseParticles;
	GLuint vao, vbo;
	GLuint program;

private:
	virtual void setup();
};

#endif // BOX_SOURCE_HPP