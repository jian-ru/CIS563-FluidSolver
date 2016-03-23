//
//  scene.cpp
//  Thanda

#include <iostream>
#include "scene.hpp"


std::shared_ptr<FS_Scene> FS_Scene::createFS_Scene(std::ifstream &jsonStream)
{
	std::shared_ptr<FS_Scene> newScene(new FS_Scene);
	Json::CharReaderBuilder builder;
	builder["collectComments"] = false;
	std::string errs;
	bool ok = Json::parseFromStream(builder, jsonStream, &newScene->sceneDesc, &errs);

	if (!ok)
	{
		std::cout << "Cannot parse scene file\n";
		return nullptr;
	}

	// Container
	float c_scaleX = newScene->sceneDesc["containerDim"]["scaleX"].asFloat();
	float c_scaleY = newScene->sceneDesc["containerDim"]["scaleX"].asFloat();
	float c_scaleZ = newScene->sceneDesc["containerDim"]["scaleX"].asFloat();
	bool isOpen[6] = { false, false, false, false, false, false };
	newScene->containers.push_back(
		std::make_shared<FS_BoxContainer>(
			glm::vec3(0.f), glm::vec3(c_scaleX, c_scaleY, c_scaleZ), isOpen));

	FS_BBox bbox;
	bbox.xmin = -c_scaleX / 2.f;
	bbox.ymin = -c_scaleY / 2.f;
	bbox.zmin = -c_scaleZ / 2.f;
	bbox.xmax = c_scaleX / 2.f;
	bbox.ymax = c_scaleY / 2.f;
	bbox.zmax = c_scaleZ / 2.f;
	newScene->grid = std::shared_ptr<FS_MACGrid>(new FS_MACGrid(bbox, 1.f));
	newScene->grid->init();

	// Source
	float s_scaleX = newScene->sceneDesc["particleDim"]["boundX"].asFloat();
	float s_scaleY = newScene->sceneDesc["particleDim"]["boundY"].asFloat();
	float s_scaleZ = newScene->sceneDesc["particleDim"]["boundZ"].asFloat();
	float p_sep = newScene->sceneDesc["particleSeparation"].asFloat();
	newScene->sources.push_back(
		std::make_shared<FS_BoxSource>(
			glm::vec3(0.f), glm::vec3(s_scaleX, s_scaleY, s_scaleZ), p_sep));

	// Solver
	newScene->solver = std::make_shared<FS_TestSolver>();

	return newScene;
}

void FS_Scene::update()
{
	static bool onceThrough = false;
	static double timeLastFrame;

	if (!onceThrough)
	{
		timeLastFrame = glfwGetTime();
		onceThrough = true;
	}

	double curTime = glfwGetTime();
	float deltaTime = curTime - timeLastFrame;
	timeLastFrame = curTime;
	
	FS_TestSolverInfo info;
	std::shared_ptr<FS_BoxContainer> c = std::dynamic_pointer_cast<FS_BoxContainer>(containers[0]);
	std::shared_ptr<FS_BoxSource> s = std::dynamic_pointer_cast<FS_BoxSource>(sources[0]);
	info.bounds[0] = c->minX;
	info.bounds[1] = c->maxX;
	info.bounds[2] = c->minY;
	info.bounds[3] = c->maxY;
	info.bounds[4] = c->minZ;
	info.bounds[5] = c->maxZ;
	info.gravity = -9.8f; // test value
	info.numParticles = grid->particles.size();
	info.isOpen[0] = (int)c->isOpen[0];
	info.isOpen[1] = (int)c->isOpen[1];
	info.isOpen[2] = (int)c->isOpen[2];
	info.isOpen[3] = (int)c->isOpen[3];
	info.isOpen[4] = (int)c->isOpen[4];
	info.isOpen[5] = (int)c->isOpen[5];

	solver->solve(deltaTime, grid->debugVBO1, &info);
}


void FS_Scene::render(std::shared_ptr<FS_Camera> pCam)
{
	//for (int i = 0; i < containers.size(); ++i)
	//{
	//	containers[i]->render(pCam);
	//}

	//for (int i = 0; i < sources.size(); ++i)
	//{
	//	sources[i]->render(pCam);
	//}

	grid->render(pCam);
}


void FS_Scene::cleanup()
{
	for (int i = 0; i < containers.size(); ++i)
	{
		containers[i]->cleanup();
	}

	for (int i = 0; i < sources.size(); ++i)
	{
		sources[i]->cleanup();
	}

	grid->cleanup();
}