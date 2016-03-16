//
//  geom.cpp
//  Thanda

#include <iostream>
#include "box_container.hpp"
#include "../main.hpp"


FS_BoxContainer::FS_BoxContainer(glm::vec3 & pos, glm::vec3 & scale_xyz, bool open[6])
{
	glm::vec3 halfScale = scale_xyz / 2.f;
	//glm::vec3 halfScale = glm::vec3(1, 1, 0);
	minX = pos.x - halfScale.x;
	maxX = pos.x + halfScale.x;
	minY = pos.y - halfScale.y;
	maxY = pos.y + halfScale.y;
	minZ = pos.z - halfScale.z;
	maxZ = pos.z + halfScale.z;
	memcpy(isOpen, open, sizeof(isOpen));

	vertexBuffer.resize(24); // 8 (vertices) * 3 (position_vec3)
	vertexBuffer[0] = vertexBuffer[9] = vertexBuffer[12] = vertexBuffer[21] = minX;
	vertexBuffer[3] = vertexBuffer[6] = vertexBuffer[15] = vertexBuffer[18] = maxX;
	vertexBuffer[1] = vertexBuffer[4] = vertexBuffer[13] = vertexBuffer[16] = minY;
	vertexBuffer[7] = vertexBuffer[10] = vertexBuffer[19] = vertexBuffer[22] = maxY;
	vertexBuffer[14] = vertexBuffer[17] = vertexBuffer[20] = vertexBuffer[23] = minZ;
	vertexBuffer[2] = vertexBuffer[5] = vertexBuffer[8] = vertexBuffer[11] = maxZ;
	//float vpos[] = { -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f,  1.0f, 0.0f };
	//vertexBuffer = std::vector<float>(vpos, vpos + 9);


	// 12 lines * 2 endpoints = 24 vertices
	unsigned int indices[] =
	{
		0, 1,
		1, 2,
		2, 3,
		3, 0,
		0, 4,
		1, 5,
		2, 6,
		3, 7,
		4, 5,
		5, 6,
		6, 7,
		7, 4
	};
	indexBuffer = std::vector<unsigned int>(indices, indices + 24);
	//unsigned int indices[] = { 0, 2 };
	//indexBuffer = std::vector<unsigned int>(indices, indices + 2);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// Shaders and shader program
	std::string vsFileName(SHADERS_DIR);
	std::string fsFileName(SHADERS_DIR);
	vsFileName += "/boxContainer_vs.glsl";
	fsFileName += "/boxContainer_fs.glsl";
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

	// Vertex and index buffers
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER,
				 vertexBuffer.size() * sizeof(float), // Don't use sizeof(vertexBuffer.data()).
				 vertexBuffer.data(),
				 GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBuffer.size() * sizeof(unsigned int), indexBuffer.data(), GL_STATIC_DRAW);
}


void FS_BoxContainer::setup()
{
	glBindVertexArray(vao);
	//glBindBuffer(GL_ARRAY_BUFFER, vbo);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(vertexBuffer.data()), vertexBuffer.data(), GL_STATIC_DRAW);
	//glEnableVertexAttribArray(0);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glUseProgram(program);
}


void FS_BoxContainer::render(std::shared_ptr<FS_Camera> pCam)
{
	setup();

	glm::mat4 MVP = pCam->proj * pCam->view; // model matrix is identity

	GLint unif_mvp = glGetUniformLocation(program, "MVP");
	glUniformMatrix4fv(unif_mvp, 1, GL_FALSE, &MVP[0][0]);

	glLineWidth(2);
	glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, NULL);
	glLineWidth(1);
	//glDrawArrays(GL_TRIANGLES, 0, 3);
	//glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL);
	//glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, NULL);
}


void FS_BoxContainer::cleanup()
{
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &ibo);
	glDeleteProgram(program);
}
