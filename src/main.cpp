#include "main.hpp"
#include <stdio.h>  
#include <stdlib.h>
#include <iostream>
#include <fstream>


#include "scene/scene.hpp"


static std::shared_ptr<FS_Camera> pCam;


bool createProgramWithShaders(const std::vector<GLenum> &types, const std::vector<const char *> &fileNames, GLuint &program)
{
	if (types.size() != fileNames.size() || types.empty())
	{
		return false; // failed;
	}

	int numShaders = types.size();
	program = glCreateProgram();
	std::vector<GLuint> shaders;

	for (int i = 0; i < numShaders; ++i)
	{
		GLuint shader = loadShader(fileNames[i], types[i]);
		shaders.push_back(shader);
		glAttachShader(program, shader);
	}

	glLinkProgram(program);

	for (int i = 0; i < shaders.size(); ++i)
	{
		glDeleteShader(shaders[i]);
	}

	// check link status
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
		glDeleteProgram(program);
		return false;
	}

	return true; // success
}


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
	GLint maxComputeShaderStorageBlocks;
	GLint maxShaderStorageBufferBindings;

	glGetInteger64i_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &localSizes[0]);
	glGetInteger64i_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &localSizes[1]);
	glGetInteger64i_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &localSizes[2]);
	glGetInteger64i_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &globalSizes[0]);
	glGetInteger64i_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &globalSizes[1]);
	glGetInteger64i_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &globalSizes[2]);
	glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &localTotalInvocations);
	glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &sharedMemSize);
	glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &maxComputeShaderStorageBlocks);
	glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &maxShaderStorageBufferBindings);

	std::cout
		<< "OpenGL version: " << static_cast<const unsigned char *>(glGetString(GL_VERSION)) << '\n'
		<< "Compute work group sizes: " << localSizes[0] << ", " << localSizes[1] << ", " << localSizes[2] << '\n'
		<< "Compute work group counts: " << globalSizes[0] << ", " << globalSizes[1] << ", " << globalSizes[2] << '\n'
		<< "Total # of invocations (local): " << localTotalInvocations << '\n'
		<< "Shared memory size: " << sharedMemSize << " Bytes\n"
		<< "Max compute shader storage blocks: " << maxComputeShaderStorageBlocks << '\n'
		<< "Max shader storage buffer bindings: " << maxShaderStorageBufferBindings << '\n';
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


// Populate the given grid with a narrow-band level set representation of a sphere.
// The width of the narrow band is determined by the grid's background value.
// (Example code only; use tools::createSphereSDF() in production.)

//typedef openvdb::FloatGrid GridType;
//void
//makeSphere(GridType& grid, float radius, const openvdb::Vec3f& c)
//{
//	typedef typename GridType::ValueType ValueT;
//	// Distance value for the constant region exterior to the narrow band
//	const ValueT outside = grid.background();
//	// Distance value for the constant region interior to the narrow band
//	// (by convention, the signed distance is negative in the interior of
//	// a level set)
//	const ValueT inside = -outside;
//	// Use the background value as the width in voxels of the narrow band.
//	// (The narrow band is centered on the surface of the sphere, which
//	// has distance 0.)
//	int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));
//	// The bounding box of the narrow band is 2*dim voxels on a side.
//	int dim = int(radius + padding);
//	// Get a voxel accessor.
//	typename GridType::Accessor accessor = grid.getAccessor();
//	// Compute the signed distance from the surface of the sphere of each
//	// voxel within the bounding box and insert the value into the grid
//	// if it is smaller in magnitude than the background value.
//	openvdb::Coord ijk;
//	int &i = ijk[0], &j = ijk[1], &k = ijk[2];
//	for (i = c[0] - dim; i < c[0] + dim; ++i) {
//		const float x2 = openvdb::math::Pow2(i - c[0]);
//		for (j = c[1] - dim; j < c[1] + dim; ++j) {
//			const float x2y2 = openvdb::math::Pow2(j - c[1]) + x2;
//			for (k = c[2] - dim; k < c[2] + dim; ++k) {
//				// The distance from the sphere surface in voxels
//				const float dist = openvdb::math::Sqrt(x2y2
//					+ openvdb::math::Pow2(k - c[2])) - radius;
//				// Convert the floating-point distance to the grid's value type.
//				ValueT val = ValueT(dist);
//				// Only insert distances that are smaller in magnitude than
//				// the background value.
//				if (val < inside || outside < val) continue;
//				// Set the distance for voxel (i,j,k).
//				accessor.setValue(ijk, val);
//			}
//		}
//	}
//	// Propagate the outside/inside sign information from the narrow band
//	// throughout the grid.
//	openvdb::v3_1_0::tools::signedFloodFill(grid.tree());
//}
//
//
//void openvdbTest()
//{
//	//openvdb::initialize();
//	//// Create a FloatGrid and populate it with a narrow-band
//	//// signed distance field of a sphere.
//	//openvdb::FloatGrid::Ptr grid =
//	//	openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
//	//		/*radius=*/50.0, /*center=*/openvdb::Vec3f(1.5, 2, 3),
//	//		/*voxel size=*/0.5, /*width=*/4.0);
//	//// Associate some metadata with the grid.
//	//grid->insertMeta("radius", openvdb::FloatMetadata(50.0));
//	//// Name the grid "LevelSetSphere".
//	//grid->setName("LevelSetSphere");
//	//// Create a VDB file object.
//	//openvdb::io::File file("C:/Users/Jian Ru/Documents/CIS563/fluidsolver/CIS563-FluidSolver/mygrids.vdb");
//	//// Add the grid pointer to a container.
//	//openvdb::GridPtrVec grids;
//	//grids.push_back(grid);
//	//// Write out the contents of the container.
//	//file.write(grids);
//	//file.close();
//
//	openvdb::initialize();
//	// Create a shared pointer to a newly-allocated grid of a built-in type:
//	// in this case, a FloatGrid, which stores one single-precision floating point
//	// value per voxel.  Other built-in grid types include BoolGrid, DoubleGrid,
//	// Int32Grid and Vec3SGrid (see openvdb.h for the complete list).
//	// The grid comprises a sparse tree representation of voxel data,
//	// user-supplied metadata and a voxel space to world space transform,
//	// which defaults to the identity transform.
//	openvdb::FloatGrid::Ptr grid =
//		openvdb::FloatGrid::create(/*background value=*/2.0);
//	// Populate the grid with a sparse, narrow-band level set representation
//	// of a sphere with radius 50 voxels, located at (1.5, 2, 3) in index space.
//	makeSphere(*grid, /*radius=*/50.0, /*center=*/openvdb::Vec3f(1.5, 2, 3));
//	// Associate some metadata with the grid.
//	grid->insertMeta("radius", openvdb::FloatMetadata(50.0));
//	// Associate a scaling transform with the grid that sets the voxel size
//	// to 0.5 units in world space.
//	grid->setTransform(
//		openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.5));
//	// Identify the grid as a level set.
//	grid->setGridClass(openvdb::GRID_LEVEL_SET);
//	// Name the grid "LevelSetSphere".
//	grid->setName("LevelSetSphere");
//	// Create a VDB file object.
//	openvdb::io::File file("C:/Users/Jian Ru/Documents/CIS563/fluidsolver/CIS563-FluidSolver/mygrids.vdb");
//	// Add the grid pointer to a container.
//	openvdb::GridPtrVec grids;
//	grids.push_back(grid);
//	// Write out the contents of the container.
//	file.write(grids);
//	file.close();
//}


int main()
{
	GLFWwindow *window = initializeGLFW_GLEW();

#ifdef _GPU_INFO
	displayGPUInfo();
#endif

	std::shared_ptr<FS_Scene> pScene = initializeScene();
	//pScene->grid->setup();

	glm::vec3 gridCenter =
		glm::vec3
			(
				pScene->gpu_grid->xcount * pScene->gpu_grid->cellSize * .5f,
				pScene->gpu_grid->ycount * pScene->gpu_grid->cellSize * .5f,
				pScene->gpu_grid->zcount * pScene->gpu_grid->cellSize * .5f
			);
	glm::vec3 camPos = gridCenter + glm::vec3(0.f, 0.f, gridCenter.z * 4.f);

	pCam = FS_Camera::createFS_Camera(WINDOW_WIDTH, WINDOW_HEIGHT, camPos, gridCenter);

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
 
	//openvdbTest();

	pScene->cleanup();
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}

