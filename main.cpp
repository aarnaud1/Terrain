/*
 * Copyright (C) 2024 Adrien ARNAUD
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "TerrainGenerator.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main(int /*argc*/, char** /*argv*/)
{
    const float refDist = 10.0f;
    const float baseResolution = 0.01f;
    const float baseDim = refDist / baseResolution;
    const float zFactor = 0.5f;

    cg::TerrainGenerator generator{};
    generator.setBaseResolution(baseResolution, baseResolution, zFactor);
    generator.setHeightWaveLength(0.2f * baseDim);
    generator.setMoistureWaveLength(0.5f * baseDim);

    generator.generateTile(glm::vec2{0, 0}, 10.0f, 45.0f, 0.0f);
    generator.generateTile(glm::vec2{0, 0}, 5.0f, 45.0f, 30.0f);
    generator.generateTile(glm::vec2{0, 0}, 5.0f, 45.0f, 60.0f);

    return EXIT_SUCCESS;
}