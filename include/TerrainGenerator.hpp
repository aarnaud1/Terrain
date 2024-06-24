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

#pragma once

#include "ValueNoiseGenerator.hpp"

#include <cstdlib>
#include <ctime>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

#define DEBUG_GENERATION

namespace cg
{
class TerrainGenerator
{
  public:
    TerrainGenerator()
    {
        static constexpr double seedRange = 1000.0;

        srand(time(NULL));
        heightRandomSeed_ = 2.0f * seedRange * double(rand()) / double(RAND_MAX);
        moistureRandomSeed_ = 2.0f * seedRange * double(rand()) / double(RAND_MAX);

        heightMapGenerator_.setOctaveCount(10);
        heightMapGenerator_.setSeed(heightRandomSeed_);

        moistureMapGenerator_.setOctaveCount(8);
        moistureMapGenerator_.setSeed(moistureRandomSeed_);
    }
    TerrainGenerator(const size_t n) : TerrainGenerator() { preallocate(n); }

    TerrainGenerator(const TerrainGenerator&) = delete;
    TerrainGenerator(TerrainGenerator&&) = delete;

    TerrainGenerator& operator=(const TerrainGenerator&) = delete;
    TerrainGenerator& operator=(TerrainGenerator&&) = delete;

    ~TerrainGenerator() = default;

    void setBaseResolution(const float x, const float y, const float z)
    {
        resolution_ = glm::vec3{x, y, z};
    }
    void setBaseResolution(const glm::vec3& res) { resolution_ = res; }

    void setHeightWaveLength(const float fact) { heightWaveLength_ = fact; }
    void setMoistureWaveLength(const float fact) { moistureWaveLength_ = fact; }

    void generateTile(
        const size_t sizeX,
        const size_t sizeY,
        const glm::vec2& offset = glm::vec2{0},
        const float theta = 0.0f);
    void generateTile(
        const glm::vec2& position, const float far, const float fov, const float theta);

    void clear()
    {
        heightMap_.clear();

        vertices_.clear();
        normals_.clear();
        colors_.clear();

        faces_.clear();
    }

    void preallocate(const size_t n)
    {
        heightMap_.reserve(n);
        vertices_.resize(n);
        normals_.resize(n);
        colors_.resize(n);
    }

    auto& vertices() { return vertices_; }
    const auto& vertices() const { return vertices_; }

    auto& normals() { return normals_; }
    const auto& normals() const { return normals_; }

    auto& colors() { return colors_; }
    const auto& colors() const { return colors_; }

    auto& faces() { return faces_; }
    const auto& faces() const { return faces_; }

  private:
    float heightRandomSeed_{0.0f};
    float moistureRandomSeed_{0.0f};

    size_t sizeX_{0};
    size_t sizeY_{0};
    glm::vec3 resolution_{1.0f, 1.0f, 1.0f};
    float heightWaveLength_{1.0f};
    float moistureWaveLength_{1.0f};

    ValueNoiseGenerator heightMapGenerator_{};
    ValueNoiseGenerator moistureMapGenerator_{};

    void generateMaps(const glm::vec2& offset, const float theta);

    void computeVertices(const glm::vec2& offset = glm::vec2{0}, const float theta = 0.0f);

    void computeNormals();

    void computeColors();

    void computeFaces();

    struct ImgPixel
    {
        uint8_t b, g, r;
    };
    std::vector<double> heightMap_{};
    std::vector<double> moistureMap_{};
    std::vector<ImgPixel> colorMap_{};

    std::vector<glm::vec3> vertices_{};
    std::vector<glm::vec3> normals_{};
    std::vector<glm::vec4> colors_{};

    std::vector<glm::ivec3> faces_{};

    static void reshapeMap(std::vector<double>& map);
    static void normalizeMap(std::vector<double>& map);

#ifdef DEBUG_GENERATION
    void debugMaps(const glm::vec2& offset, const float theta);
    void exportPLY(const char* filename, const glm::vec2& offset, const float theta);
#endif
};
} // namespace cg