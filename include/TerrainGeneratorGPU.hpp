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

#define DEBUG_SKY_MAPS

namespace cg
{
// TODO : consider adding this wrappper to vkw
class CubeMapSampler
{
  public:
    CubeMapSampler() = default;
    void init(vkw::Device& device)
    {
        device_ = &device;

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.pNext = nullptr;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        CHECK_VK(
            vkCreateSampler(device_->getHandle(), &samplerInfo, nullptr, &sampler_),
            "Creating color attachment sampler");
    }

    auto getHandle() const { return sampler_; }

    ~CubeMapSampler()
    {
        if(sampler_ != VK_NULL_HANDLE)
        {
            vkDestroySampler(device_->getHandle(), sampler_, nullptr);
        }
    }

  private:
    vkw::Device* device_{nullptr};
    VkSampler sampler_{VK_NULL_HANDLE};
};

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
    void setVerticalScale(const float scale) { verticalScale_ = scale; }

    void initStorage(const float farDistance, const float fov);

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

    auto& cubeMapVertices() { return *cubeMapVertices_; }
    const auto& cubeMapVertices() const { return *cubeMapVertices_; }

    auto& cubeMapSampler() { return cubeMapSampler_; }
    const auto& cubeMapSampler() const { return cubeMapSampler_; }

    auto& cubeMapImageView() { return skyCubeMapImageView_; }
    const auto& cubeMapImageView() const { return skyCubeMapImageView_; }

    auto& cubeMapImage() { return *cubeMapImage_; }
    const auto& cubeMapImage() const { return *cubeMapImage_; }

    uint32_t vertexCount() const { return terrainSizeX_ * terrainSizeY_; }
    uint32_t faceCount() const { return 2 * (terrainSizeX_ - 1) * (terrainSizeY_ - 1); }

    uint32_t waterVertexCount() const { return waterSizeX_ * waterSizeY_; }
    uint32_t waterFacesCount() const { return 2 * (waterSizeX_ - 1) * (waterSizeY_ - 1); }

    float waterHeight() const { return verticalScale_ * waterHeight_; }
    float mapWidth() const { return mapWidth_; }
    float mapHeight() const { return mapHeight_; }
    float maxAltitude() const { return maxAltitude_; }
    float cubeMapDim() const { return std::max(mapWidth_, mapHeight_); }
    glm::vec3 cubeMapOffset() const
    {
        return glm::vec3(0.0f, 0.5f * cubeMapDim(), 0.5f * cubeMapDim());
    }

  private:
    static constexpr uint32_t maxComputeBlockSize = 1024;

    vkw::Device* device_{nullptr};

    static constexpr float waterHeight_ = 0.1f;
    float mapWidth_{0.0f};
    float mapHeight_{0.0f};
    float maxAltitude_{0.0f};
    float refDist_{1.0f};
    float terrainResolution_{1.0f};
    float waterResolution_{0.5f};
    float verticalScale_{1.0f};

    uint32_t terrainSizeX_;
    uint32_t terrainSizeY_;

    uint32_t waterSizeX_;
    uint32_t waterSizeY_;

    vkw::Memory vertexMemory_{};
    vkw::Buffer<glm::vec3>* vertices_{nullptr};
    vkw::Buffer<glm::vec3>* normals_{nullptr};
    vkw::Buffer<glm::vec4>* colors_{nullptr};
    vkw::Buffer<glm::vec3>* waterVertices_{nullptr};
    vkw::Buffer<glm::vec3>* waterNormals_{nullptr};

    vkw::Memory cubeMapVertexMemory_{};
    vkw::Buffer<glm::vec3>* cubeMapVertices_{nullptr};

    vkw::Memory facesMemory_{};
    vkw::Buffer<glm::uvec3>* faces_{nullptr};
    vkw::Buffer<glm::uvec3>* waterFaces_{nullptr};

    vkw::Memory mapsMemory_{};
    vkw::Buffer<float>* heightMap_{nullptr};
    vkw::Buffer<float>* moistureMap_{nullptr};
    vkw::Buffer<float>* waterMap_{nullptr};

    static constexpr uint32_t cubeMapSize = 512;
    static constexpr size_t mapCount = 6;
    static constexpr uint32_t POSITIVE_X = 0;
    static constexpr uint32_t NEGATIVE_X = 1;
    static constexpr uint32_t POSITIVE_Y = 2;
    static constexpr uint32_t NEGATIVE_Y = 3;
    static constexpr uint32_t POSITIVE_Z = 4;
    static constexpr uint32_t NEGATIVE_Z = 5;

    CubeMapSampler cubeMapSampler_;
    vkw::Memory cubeMapsMemory_{};
    vkw::Image* cubeMapImage_{nullptr};
    std::array<vkw::ImageView, mapCount> computeSkyImageViews_{};
    vkw::ImageView skyCubeMapImageView_{};

#ifdef DEBUG_SKY_MAPS
    vkw::Memory stagingMem_{};
    vkw::Buffer<glm::vec4>* skyMapStaging_{};
#endif

    vkw::Queue<vkw::QueueFamilyType::COMPUTE> computeQueue_{};
    vkw::CommandPool<vkw::QueueFamilyType::COMPUTE> computeCommandPool_{};
    vkw::CommandBuffer<vkw::QueueFamilyType::COMPUTE> computeTerrainCommandBuffer_{};
    vkw::CommandBuffer<vkw::QueueFamilyType::COMPUTE> computeWaterCommandBuffer_{};
    vkw::CommandBuffer<vkw::QueueFamilyType::COMPUTE> computeSkyMapsCommandBuffer_{};

    vkw::Fence terrainFence_{};
    vkw::Fence waterFence_{};
    vkw::Fence skyFence_{};
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

    struct ComputeSkyMapConstants
    {
        uint32_t mapSize;
        uint32_t octaves;
        float waveLength;
        float theta;
    };
    vkw::ComputeProgram computePXSkyMapProgram_;
    vkw::ComputeProgram computeMXSkyMapProgram_;
    vkw::ComputeProgram computePYSkyMapProgram_;
    vkw::ComputeProgram computeMYSkyMapProgram_;
    vkw::ComputeProgram computePZSkyMapProgram_;
    vkw::ComputeProgram computeMZSkyMapProgram_;

    bool allocated_{false};

    void initFaces();
    void initWaterFaces();
    void initImageLayouts();
    void initCubeMapVertices();

    void updateCommandBuffers(const float offsetX, const float offsetY, const float theta);

#ifdef DEBUG_SKY_MAPS
    void debugSkyMaps();
#endif
};
} // namespace cg