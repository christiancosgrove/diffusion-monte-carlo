#pragma once
#include <glm\glm.hpp>
#include <array>
template<int electrons>
class Walker
{
public:
	Walker(const std::array<glm::dvec3, electrons>& pos) : positions(pos), alive(true) {}

	bool alive;
	std::array<glm::dvec3, electrons> positions;

};

