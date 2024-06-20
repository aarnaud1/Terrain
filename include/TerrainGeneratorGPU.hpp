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

#include <cstdint>
#include <glm/glm.hpp>
#include <vkWrappers/wrappers.hpp>

// #define DEBUG_TERRAIN

namespace cg
{
class TerrainGeneratorGPU
{
  public:
    TerrainGeneratorGPU() = delete;
    TerrainGeneratorGPU(vk::Device& device);

    TerrainGeneratorGPU(const TerrainGeneratorGPU&) = delete;
    TerrainGeneratorGPU(TerrainGeneratorGPU&&) = delete;

    TerrainGeneratorGPU& operator=(const TerrainGeneratorGPU&) = delete;
    TerrainGeneratorGPU& operator=(TerrainGeneratorGPU&&) = delete;

    void setRefDistance(const float dist) { refDist_ = dist; }
    void setBaseResolution(const float res) { baseResolution_ = res; }

    void initStorage(const uint32_t sizeX, const uint32_t sizeY);
    void generate(const float offsetX, const float offsetY, const float theta);

    auto& vertices() { return *vertices_; }
    const auto& vertices() const { return *vertices_; }

    auto& normals() { return *normals_; }
    const auto& normals() const { return *normals_; }

    auto& colors() { return *colors_; }
    const auto& colors() const { return *colors_; }

    auto& faces() { return *faces_; }
    const auto& faces() const { return *faces_; }

    uint32_t vertexCount() const { return sizeX_ * sizeY_; }
    uint32_t faceCount() const { return 2 * (sizeX_ - 1) * (sizeY_ - 1); }

  private:
    static constexpr uint32_t maxComputeBlockSize = 1024;

    vk::Device* device_{nullptr};

    float refDist_{1.0f};
    float baseResolution_{1.0f};

    uint32_t sizeX_;
    uint32_t sizeY_;

    vk::Memory vertexMemory_{};
    vk::Buffer<glm::vec3>* vertices_{nullptr};
    vk::Buffer<glm::vec3>* normals_{nullptr};
    vk::Buffer<glm::vec4>* colors_{nullptr};

    vk::Memory facesMemory_{};
    vk::Buffer<glm::uvec3>* faces_{nullptr};

    vk::Memory mapsMemory_{};
    vk::Buffer<float>* heightMap_{nullptr};
    vk::Buffer<float>* moistureMap_{nullptr};

#ifdef DEBUG_TERRAIN
    vk::Memory verticesStagingMem_{};
    vk::Buffer<glm::vec3>* verticesStaging_{nullptr};

    vk::Memory normalsStagingMem_{};
    vk::Buffer<glm::vec3>* normalsStaging_{nullptr};

    vk::Memory colorStagingMem_{};
    vk::Buffer<glm::vec4>* colorsStaging_{nullptr};

    vk::Memory facesStagingMem_{};
    vk::Buffer<glm::uvec3>* facesStaging_{nullptr};
#endif

    vk::Queue<vk::QueueFamilyType::COMPUTE> computeQueue_{};
    vk::CommandPool<vk::QueueFamilyType::COMPUTE> computeCommandPool_{};

    // Algorithms data
    struct
    {
        uint32_t dimX;
        uint32_t dimY;
    } initFacesConstants_;
    vk::ComputeProgram initFacesProgram_;

    struct
    {
        uint32_t sizeX;
        uint32_t sizeY;
        uint32_t heightOctaves;
        uint32_t moistureOctaves;
        float heightWaveLength;
        float moistureWaveLength;
        float offX;
        float offY;
        float theta;
    } computeMapConstants_;
    vk::ComputeProgram computeMapsProgram_;

    struct
    {
        uint32_t pointCount;
    } computeColorsConstants_;
    vk::ComputeProgram computeColorsProgram_;

    struct
    {
        uint32_t sizeX;
        uint32_t sizeY;
        float triangleRes;
        float zScale;
    } computeVerticesConstants_;
    vk::ComputeProgram computeVerticesProgram_;

    bool allocated_{false};

    void initFaces();
};
} // namespace cg