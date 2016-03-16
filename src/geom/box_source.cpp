#include <iostream>
#include "box_source.hpp"
#include "../main.hpp"


FS_BoxSource::FS_BoxSource(glm::vec3 & pos, glm::vec3 & scale_xyz, float p_sep)
{
	glm::vec3 halfScale = scale_xyz * .5f;
	float xStart = pos.x - halfScale.x;
	float yStart = pos.y - halfScale.y;
	float zStart = pos.z - halfScale.z;
	float r_p_sep = 1.f / p_sep;
	int xCount = static_cast<int>(scale_xyz.x * r_p_sep);
	int yCount = static_cast<int>(scale_xyz.y * r_p_sep);
	int zCount = static_cast<int>(scale_xyz.z * r_p_sep);

	for (int k = 0; k < zCount; ++k)
	{
		for (int j = 0; j < yCount; ++j)
		{
			for (int i = 0; i < xCount; ++i)
			{
				FS_Particle p;
				p.position = glm::vec3(xStart + i * p_sep, yStart + j * p_sep, zStart + k * p_sep);
				p.velocity = glm::vec3(0.f);
				baseParticles.push_back(std::move(p));
			}
		}
	}

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// Shaders and shader program
	std::string vsFileName(SHADERS_DIR);
	std::string fsFileName(SHADERS_DIR);
	vsFileName += "/boxSource_vs.glsl";
	fsFileName += "/boxSource_fs.glsl";
	GLuint vsn = loadShader(vsFileName.c_str(), GL_VERTEX_SHADER);
	GLuint fsn = loadShader(fsFileName.c_str(), GL_FRAGMENT_SHADER);

	program = glCreateProgram();
	glAttachShader(program, vsn);
	glAttachShader(program, fsn);
	glLinkProgram(program);

	glDeleteShader(vsn);
	glDeleteShader(fsn);

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

	// Vertex buffer
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, baseParticles.size() * sizeof(FS_Particle), baseParticles.data(), GL_DYNAMIC_COPY);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(FS_Particle), NULL);
}


void FS_BoxSource::setup()
{
	glBindVertexArray(vao);
	glUseProgram(program);
}


void FS_BoxSource::render(std::shared_ptr<FS_Camera> pCam)
{
	setup();

	glm::mat4 MVP = pCam->proj * pCam->view; // model matrix is identity
	GLint unif_mvp = glGetUniformLocation(program, "MVP");
	glUniformMatrix4fv(unif_mvp, 1, GL_FALSE, &MVP[0][0]);

	glPointSize(4);
	glDrawArrays(GL_POINTS, 0, baseParticles.size());
	glPointSize(1);
}


void FS_BoxSource::cleanup()
{
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
	glDeleteProgram(program);
}