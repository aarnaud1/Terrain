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

namespace cg
{
class TerrainGeneratorGPU
{
  public:
    TerrainGeneratorGPU() = delete;
    TerrainGeneratorGPU(vkw::Device& device);

    TerrainGeneratorGPU(const TerrainGeneratorGPU&) = delete;
    TerrainGeneratorGPU(TerrainGeneratorGPU&&) = delete;

    TerrainGeneratorGPU& operator=(const TerrainGeneratorGPU&) = delete;
    TerrainGeneratorGPU& operator=(TerrainGeneratorGPU&&) = delete;

    void setRefDistance(const float dist) { refDist_ = dist; }
    void setBaseResolution(const float res) { terrainResolution_ = res; }
    void setWaterResolution(const float res) { waterResolution_ = res; }

    void initStorage(const uint32_t sizeX, const uint32_t sizeY);
    void generate(
        const float offsetX,
        const float offsetY,
        const float theta,
        vkw::Semaphore& terrainSemaphote,
        vkw::Semaphore& waterSemaphore);

    auto& vertices() { return *vertices_; }
    const auto& vertices() const { return *vertices_; }

    auto& normals() { return *normals_; }
    const auto& normals() const { return *normals_; }

    auto& colors() { return *colors_; }
    const auto& colors() const { return *colors_; }

    auto& faces() { return *faces_; }
    const auto& faces() const { return *faces_; }

    auto& waterVertices() { return *waterVertices_; }
    const auto& waterVertices() const { return *waterVertices_; }

    auto& waterNormals() { return *waterNormals_; }
    const auto& waterNormals() const { return *waterNormals_; }

    auto& waterFaces() { return *waterFaces_; }
    const auto& waterFaces() const { return *waterFaces_; }

    uint32_t vertexCount() const { return sizeX_ * sizeY_; }
    uint32_t faceCount() const { return 2 * (sizeX_ - 1) * (sizeY_ - 1); }

    uint32_t waterVertexCount() const { return waterSizeX_ * waterSizeY_; }
    uint32_t waterFacesCount() const { return 2 * (waterSizeX_ - 1) * (waterSizeY_ - 1); }

  private:
    static constexpr uint32_t maxComputeBlockSize = 1024;

    vkw::Device* device_{nullptr};

    float refDist_{1.0f};
    float terrainResolution_{1.0f};
    float waterResolution_{0.5f};

    uint32_t sizeX_;
    uint32_t sizeY_;

    uint32_t waterSizeX_;
    uint32_t waterSizeY_;

    vkw::Memory vertexMemory_{};
    vkw::Buffer<glm::vec3>* vertices_{nullptr};
    vkw::Buffer<glm::vec3>* normals_{nullptr};
    vkw::Buffer<glm::vec4>* colors_{nullptr};
    vkw::Buffer<glm::vec3>* waterVertices_{nullptr};
    vkw::Buffer<glm::vec3>* waterNormals_{nullptr};

    vkw::Memory facesMemory_{};
    vkw::Buffer<glm::uvec3>* faces_{nullptr};
    vkw::Buffer<glm::uvec3>* waterFaces_{nullptr};

    vkw::Memory mapsMemory_{};
    vkw::Buffer<float>* heightMap_{nullptr};
    vkw::Buffer<float>* moistureMap_{nullptr};
    vkw::Buffer<float>* waterMap_{nullptr};

    vkw::Queue<vkw::QueueFamilyType::COMPUTE> computeQueue_{};
    vkw::CommandPool<vkw::QueueFamilyType::COMPUTE> computeCommandPool_{};
    vkw::CommandBuffer<vkw::QueueFamilyType::COMPUTE> computeTerrainCommandBuffer_{};
    vkw::CommandBuffer<vkw::QueueFamilyType::COMPUTE> computeWaterCommandBuffer_{};

    vkw::Fence terrainFence_{};
    vkw::Fence waterFence_{};
    vkw::Semaphore terrainGeneratedSemaphore_{};

    // Algorithms data
    struct
    {
        uint32_t dimX;
        uint32_t dimY;
    } initFacesConstants_;
    vkw::ComputeProgram initFacesProgram_;
    vkw::ComputeProgram initWaterFacesProgram_;

    struct ComputeMapConstants
    {
        uint32_t sizeX;
        uint32_t sizeY;
        uint32_t octaves;
        float waveLength;
        float offX;
        float offY;
        float theta;
    };
    vkw::ComputeProgram computeHeightMapProgram_;
    vkw::ComputeProgram computeMoistureMapProgram_;
    vkw::ComputeProgram computeWaterMapProgram_;

    struct ComputeColorsConstants
    {
        uint32_t pointCount;
    };
    vkw::ComputeProgram computeColorsProgram_;

    struct ComputeVerticesConstants
    {
        uint32_t sizeX;
        uint32_t sizeY;
        float triangleRes;
        float zScale;
    };
    vkw::ComputeProgram computeVerticesProgram_;

    struct ComputeWaterConstants
    {
        uint32_t sizeX;
        uint32_t sizeY;
        float triangleRes;
        float zScale;
    };
    vkw::ComputeProgram computeWaterProgram_;

    bool allocated_{false};

    void initFaces();
    void initWaterFaces();

    void updateCommandBuffers(const float offsetX, const float offsetY, const float theta);
};
} // namespace cg