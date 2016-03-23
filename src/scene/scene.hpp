//
//  scene.hpp
//  Thanda

#ifndef SCENE_HPP
#define SCENE_HPP

#include <memory>
#include <fstream>
#include "json/json.h"
#include "../camera/camera.hpp"
#include "../geom/box_container.hpp"
#include "../geom/box_source.hpp"
#include "../fluidSolver/fluidSolver.hpp"
#include "../geom/grid.hpp"


class FS_Scene
{
public:
	static std::shared_ptr<FS_Scene> createFS_Scene(std::ifstream &jsonStream);

	virtual ~FS_Scene() {}

	void update();
	void render(std::shared_ptr<FS_Camera> pCam);
	void cleanup();

	std::shared_ptr<FS_MACGrid> grid;
	std::vector<std::shared_ptr<FS_Geometry> > containers;
	std::vector<std::shared_ptr<FS_Geometry> > sources;
	std::shared_ptr<FS_Solver> solver;
	Json::Value sceneDesc;

protected:
	FS_Scene() {}
};

#endif // SCENE_HPP