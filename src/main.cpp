#include <stdio.h>  
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "main.hpp"
#include "scene/scene.hpp"


static std::shared_ptr<FS_Camera> pCam;


bool readWholeFile(const char *fileName, std::string &ret_content)
{
	bool success = true;
	std::ifstream ifs(fileName, std::ifstream::in | std::ifstream::binary);

	if (ifs)
	{
		ifs.seekg(0, std::ifstream::end);
		int length = ifs.tellg();
		ifs.seekg(0, std::ifstream::beg);

		ret_content.resize(length);
		ifs.read(&ret_content[0], length);

		ifs.close();
	}
	else
	{
		std::cout << "Cannot open shader file: " << fileName << '\n';
		return !success;
	}

	return success;
}


GLuint loadShader(const char * fileName, GLenum type)
{
	std::string shaderSrc;
	readWholeFile(fileName, shaderSrc);

	GLuint shader = glCreateShader(type);
	const char *srcRaw = shaderSrc.c_str();
	glShaderSource(shader, 1, &srcRaw, NULL);
	glCompileShader(shader);

	// Check for errors
	GLint status = GL_FALSE;
	int infoLogLength = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
	if (infoLogLength > 0) {
		std::vector<char> msg(infoLogLength + 1);
		glGetShaderInfoLog(shader, infoLogLength, NULL, &msg[0]);
		std::cout << msg.data() << '\n';
	}
	if (!status)
	{
		exit(EXIT_FAILURE);
	}

	return shader;
}


static void error_callback(int error, const char* description)
{
	fputs(description, stderr);
	_fgetchar();
}


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
	else if (key == GLFW_KEY_W && action == GLFW_PRESS)
	{
		pCam->moveForward();
	}
	else if (key == GLFW_KEY_S && action == GLFW_PRESS)
	{
		pCam->moveBackward();
	}
	else if (key == GLFW_KEY_A && action == GLFW_PRESS)
	{
		pCam->moveLeft();
	}
	else if (key == GLFW_KEY_D && action == GLFW_PRESS)
	{
		pCam->moveRight();
	}
	else if (key == GLFW_KEY_Q && action == GLFW_PRESS)
	{
		pCam->moveDown();
	}
	else if (key == GLFW_KEY_E && action == GLFW_PRESS)
	{
		pCam->moveUp();
	}
	else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
	{
		pCam->rotateLeft();
	}
	else if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
	{
		pCam->rotateRight();
	}
	else if (key == GLFW_KEY_UP && action == GLFW_PRESS)
	{
		pCam->rotateUp();
	}
	else if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
	{
		pCam->rotateDown();
	}
}


void displayGPUInfo()
{
	GLint64 localSizes[3]; // maximum numbers of work items may be allowed in x, y, and z dimensions for 1 work group
	GLint64 globalSizes[3]; // maximum numbers of work groups may be allowed in x, y, and z dimensions
	GLint localTotalInvocations; // local_size_x * local_size_y * local_size_z <= localTotalInvocations
	GLint sharedMemSize;

	glGetInteger64i_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &localSizes[0]);
	glGetInteger64i_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &localSizes[1]);
	glGetInteger64i_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &localSizes[2]);
	glGetInteger64i_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &globalSizes[0]);
	glGetInteger64i_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &globalSizes[1]);
	glGetInteger64i_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &globalSizes[2]);
	glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &localTotalInvocations);
	glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &sharedMemSize);

	std::cout
		<< "OpenGL version: " << static_cast<const unsigned char *>(glGetString(GL_VERSION)) << '\n'
		<< "Compute work group sizes: " << localSizes[0] << ", " << localSizes[1] << ", " << localSizes[2] << '\n'
		<< "Compute work group counts: " << globalSizes[0] << ", " << globalSizes[1] << ", " << globalSizes[2] << '\n'
		<< "Total # of invocations (local): " << localTotalInvocations << '\n'
		<< "Shared memory size: " << sharedMemSize << " Bytes\n";
}


GLFWwindow *initializeGLFW_GLEW()
{
	glfwSetErrorCallback(error_callback);

	//Initialize GLFW  
	if (!glfwInit())
	{
		exit(EXIT_FAILURE);
	}

	// Use OpenGL 4.5
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	//glfwWindowHint(GLFW_SAMPLES, 4); //Request 4x antialiasing  
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window;

	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Preview", NULL, NULL);
	if (!window)
	{
		fprintf(stderr, "Failed to open GLFW window.\n");
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);

	glfwSetKeyCallback(window, key_callback);

	// Initialize GLEW
	glewExperimental = true;
	GLenum err = glewInit();
	if (err != GLEW_OK)
	{
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	return window;
}


std::shared_ptr<FS_Scene> initializeScene()
{
	std::ifstream jsonStream(SCENE_FILE_NAME, std::ifstream::in);
	return FS_Scene::createFS_Scene(jsonStream);
}


//static const char *vssrc =
//"#version 450 core\n"
//"layout(location = 0) in vec3 position;"
//"uniform mat4 MVP;"
//"void main()"
//"{"
//"	gl_Position = vec4(position, 1.0);"
//"}";
//
//static const char *fssrc =
//"#version 450 core\n"
//"out vec4 color;"
//"void main()"
//"{"
//"	color = vec4(1.0, 0.0, 0.0, 1.0);"
//"}";
//
//static float vpos[] = { -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f,  1.0f, 0.0f };
//static std::vector<float> vertexBuffer = std::vector<float>(vpos, vpos + 9);


int main()
{
	GLFWwindow *window = initializeGLFW_GLEW();

#ifdef _GPU_INFO
	displayGPUInfo();
#endif

	std::shared_ptr<FS_Scene> pScene = initializeScene();
	pScene->grid->setup();
	pCam = FS_Camera::createFS_Camera(WINDOW_WIDTH, WINDOW_HEIGHT);

	// DEBUG
	//GLuint VertexArrayID;
	//glGenVertexArrays(1, &VertexArrayID);
	//glBindVertexArray(VertexArrayID);

	//GLuint vsn = glCreateShader(GL_VERTEX_SHADER);
	//glShaderSource(vsn, 1, &vssrc, NULL);
	//glCompileShader(vsn);
	//GLuint fsn = glCreateShader(GL_FRAGMENT_SHADER);
	//glShaderSource(fsn, 1, &fssrc, NULL);
	//glCompileShader(fsn);
	//GLuint program = glCreateProgram();
	//glAttachShader(program, vsn);
	//glAttachShader(program, fsn);
	//glLinkProgram(program);
	//glDeleteShader(vsn);
	//glDeleteShader(fsn);
	//std::string vsFileName(SHADERS_DIR);
	//std::string fsFileName(SHADERS_DIR);
	//vsFileName += "/boxContainer_vs.glsl";
	//fsFileName += "/boxContainer_fs.glsl";
	//GLuint vsn = loadShader(vsFileName.c_str(), GL_VERTEX_SHADER);
	//GLuint fsn = loadShader(fsFileName.c_str(), GL_FRAGMENT_SHADER);
	//GLuint program = glCreateProgram();
	//glAttachShader(program, vsn);
	//glAttachShader(program, fsn);
	//glLinkProgram(program);
	//glDeleteShader(vsn);
	//glDeleteShader(fsn);

	//GLuint vertexbuffer;
	//glGenBuffers(1, &vertexbuffer);
	//glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	//glBufferData(GL_ARRAY_BUFFER, vertexBuffer.size() * sizeof(float), vertexBuffer.data(), GL_STATIC_DRAW);
	// DEBUG

	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
	//glLineWidth(5);

	while (!glfwWindowShouldClose(window))
	{
		glClear(GL_COLOR_BUFFER_BIT);

		//glUseProgram(program);

		//glEnableVertexAttribArray(0);
		//glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		//glVertexAttribPointer(
		//	0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
		//	3,                  // size
		//	GL_FLOAT,           // type
		//	GL_FALSE,           // normalized?
		//	0,                  // stride
		//	(void*)0            // array buffer offset
		//	);

		//// Draw the triangle !
		//glDrawArrays(GL_TRIANGLES, 0, 3); // 3 indices starting at 0 -> 1 triangle

		//glDisableVertexAttribArray(0);

		pScene->update();
		pScene->render(pCam);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
 
	pScene->cleanup();
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}

