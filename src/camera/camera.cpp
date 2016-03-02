//
//  camera.cpp
//  Thanda
//

#include "camera.hpp"
#include "glm/gtx/rotate_vector.hpp"


std::shared_ptr<FS_Camera> FS_Camera::createFS_Camera(float width, float height,
	glm::vec3 & pos, glm::vec3 & at, float fovy, float near, float far)
{
	return std::shared_ptr<FS_Camera>(new FS_Camera(width, height, pos, at, fovy, near, far));
}


FS_Camera::FS_Camera(float w, float h, glm::vec3 & pos, glm::vec3 & at,
					 float fovy, float near, float far)
	: position(pos), aimDir(glm::normalize(at - pos)), fov(fovy), aspectRatio(w / h),
	  nearClip(near), farClip(far)
{
	updateViewMatrix();
	updateProjMatrix();
}


void FS_Camera::moveLeft(float amount)
{
	glm::vec3 right = glm::normalize(glm::cross(WORLD_UP_DIRECTION, -aimDir));
	position -= amount * right;
	updateViewMatrix();
}


void FS_Camera::moveRight(float amount)
{
	glm::vec3 right = glm::normalize(glm::cross(WORLD_UP_DIRECTION, -aimDir));
	position += amount * right;
	updateViewMatrix();
}


void FS_Camera::moveUp(float amount)
{
	position += amount * WORLD_UP_DIRECTION;
	updateViewMatrix();
}


void FS_Camera::moveDown(float amount)
{
	position -= amount * WORLD_UP_DIRECTION;
	updateViewMatrix();
}


void FS_Camera::moveForward(float amount)
{
	position += amount * aimDir;
	updateViewMatrix();
}


void FS_Camera::moveBackward(float amount)
{
	position -= amount * aimDir;
	updateViewMatrix();
}


void FS_Camera::rotateLeft(float angle)
{
	aimDir = glm::rotate(aimDir, angle, WORLD_UP_DIRECTION);
	updateViewMatrix();
}


void FS_Camera::rotateRight(float angle)
{
	aimDir = glm::rotate(aimDir, -angle, WORLD_UP_DIRECTION);
	updateViewMatrix();
}


void FS_Camera::rotateUp(float angle)
{
	if (glm::dot(aimDir, WORLD_UP_DIRECTION) < COS_MAX_TILT_ANGLE)
	{
		glm::vec3 right = glm::normalize(glm::cross(WORLD_UP_DIRECTION, -aimDir));
		aimDir = glm::rotate(aimDir, angle, right);
		updateViewMatrix();
	}
}


void FS_Camera::rotateDown(float angle)
{
	if (glm::dot(aimDir, WORLD_UP_DIRECTION) > -COS_MAX_TILT_ANGLE)
	{
		glm::vec3 right = glm::normalize(glm::cross(WORLD_UP_DIRECTION, -aimDir));
		aimDir = glm::rotate(aimDir, -angle, right);
		updateViewMatrix();
	}
}