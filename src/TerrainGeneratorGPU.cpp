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

#include "TerrainGeneratorGPU.hpp"

#include <chrono>
#include <stdexcept>

#ifdef DEBUG_SKY_MAPS
#    include <algorithm>
#    include <opencv2/core.hpp>
#    include <opencv2/highgui.hpp>
#    include <opencv2/imgcodecs.hpp>
#    include <opencv2/imgproc.hpp>
#endif

namespace cg
{
TerrainGeneratorGPU::TerrainGeneratorGPU(vkw::Device& device)
    : device_{&device}
    , initFacesProgram_{device, "output/spv/initFaces_comp.spv"}
    , initWaterFacesProgram_{device, "output/spv/initFaces_comp.spv"}
    , computeHeightMapProgram_{device, "output/spv/computeMaps_comp.spv"}
    , computeMoistureMapProgram_{device, "output/spv/computeMaps_comp.spv"}
    , computeWaterMapProgram_{device, "output/spv/computeMaps_comp.spv"}
    , computeColorsProgram_{device, "output/spv/computeColors_comp.spv"}
    , computeVerticesProgram_{device, "output/spv/computeVertices_comp.spv"}
    , computeWaterProgram_{device, "output/spv/computeWater_comp.spv"}
    , computePXSkyMapProgram_{device, "output/spv/computeSkyMap_comp.spv"}
    , computeMXSkyMapProgram_{device, "output/spv/computeSkyMap_comp.spv"}
    , computePYSkyMapProgram_{device, "output/spv/computeSkyMap_comp.spv"}
    , computeMYSkyMapProgram_{device, "output/spv/computeSkyMap_comp.spv"}
    , computePZSkyMapProgram_{device, "output/spv/computeSkyMap_comp.spv"}
    , computeMZSkyMapProgram_{device, "output/spv/computeSkyMap_comp.spv"}
{}

void TerrainGeneratorGPU::initStorage(const float farDistance, const float fov)
{
    static constexpr double seedRange = 1000.0;

    srand(time(NULL));
    const float heightRandomSeed = 2.0f * seedRange * double(rand()) / double(RAND_MAX);
    const float moistRandomSeed = 2.0f * seedRange * double(rand()) / double(RAND_MAX);
    const float waterRandomSeed = 2.0f * seedRange * double(rand()) / double(RAND_MAX);
    const float skyRandomSeed = 2.0f * seedRange * double(rand()) / double(RAND_MAX);

    const float d = farDistance * glm::tan(glm::radians(fov));
    const uint32_t terrainSize = std::max(
        uint32_t((2.0f * d) / terrainResolution_), uint32_t(farDistance / terrainResolution_));
    terrainSizeX_ = terrainSize; // uint32_t((2.0f * d) / terrainResolution_);
    terrainSizeY_ = terrainSize; // uint32_t(farDistance / terrainResolution_);

    const uint32_t waterSize = std::max(
        uint32_t((2.0f * d) / waterResolution_), uint32_t(farDistance / waterResolution_));
    waterSizeX_ = waterSize; // uint32_t((2.0f * d) / waterResolution_);
    waterSizeY_ = waterSize; // uint32_t(farDistance / waterResolution_);

    maxAltitude_ = farDistance;
    mapWidth_ = float(terrainSizeX_) * terrainResolution_;
    mapHeight_ = float(terrainSizeY_) * terrainResolution_;

    const uint32_t waterVertexCount = waterSizeX_ * waterSizeY_;
    const uint32_t waterFaceCount = 2 * (waterSizeX_ - 1) * (waterSizeY_ - 1);

    const uint32_t mapSize = terrainSizeX_ * terrainSizeY_;
    mapsMemory_.init(*device_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    heightMap_ = &mapsMemory_.createBuffer<float>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mapSize);
    moistureMap_ = &mapsMemory_.createBuffer<float>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mapSize);
    waterMap_ = &mapsMemory_.createBuffer<float>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, waterSizeX_ * waterSizeY_);
    mapsMemory_.allocate();

    const uint32_t vertexCount = terrainSizeX_ * terrainSizeY_;
    vertexMemory_.init(*device_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vertices_ = &vertexMemory_.createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCount);
    normals_ = &vertexMemory_.createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCount);
    colors_ = &vertexMemory_.createBuffer<glm::vec4>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCount);
    waterVertices_ = &vertexMemory_.createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, waterVertexCount);
    waterNormals_ = &vertexMemory_.createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, waterVertexCount);
    vertexMemory_.allocate();

    cubeMapVertexMemory_.init(
        *device_,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    cubeMapVertices_
        = &cubeMapVertexMemory_.createBuffer<glm::vec3>(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 36);
    cubeMapVertexMemory_.allocate();

    const uint32_t faceCount = 2 * (terrainSizeX_ - 1) * (terrainSizeY_ - 1);
    facesMemory_.init(*device_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    faces_ = &facesMemory_.createBuffer<glm::uvec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, faceCount);
    waterFaces_ = &facesMemory_.createBuffer<glm::uvec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, waterFaceCount);
    facesMemory_.allocate();

    const VkExtent3D imgExtent{cubeMapSize, cubeMapSize, 1};
    const VkImageUsageFlags usage
        = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    const VkImageCreateFlags createFlags
        = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    cubeMapsMemory_.init(*device_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    cubeMapImage_ = &cubeMapsMemory_.createImage(
        VK_IMAGE_TYPE_2D,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        imgExtent,
        usage,
        6,
        VK_IMAGE_TILING_OPTIMAL,
        1,
        createFlags);
    cubeMapsMemory_.allocate();

#ifdef DEBUG_SKY_MAPS
    const uint32_t cubeMapRes = cubeMapSize * cubeMapSize;
    stagingMem_.init(
        *device_,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    skyMapStaging_ = &stagingMem_.createBuffer<glm::vec4>(
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, cubeMapRes);
    stagingMem_.allocate();
#endif

    for(uint32_t i = 0; i < mapCount; ++i)
    {
        computeSkyImageViews_[i].init(
            *device_,
            *cubeMapImage_,
            VK_IMAGE_VIEW_TYPE_2D,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, i, 1});
    }
    skyCubeMapImageView_.init(
        *device_,
        *cubeMapImage_,
        VK_IMAGE_VIEW_TYPE_CUBE,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6});
    cubeMapSampler_.init(*device_);

    initFacesProgram_.bindStorageBuffers(*faces_)
        .spec(maxComputeBlockSize)
        .pushConstantsRange(sizeof(initFacesConstants_))
        .create();

    computeHeightMapProgram_.bindStorageBuffers(*heightMap_)
        .spec(maxComputeBlockSize, heightRandomSeed)
        .pushConstantsRange(sizeof(ComputeMapConstants))
        .create();
    computeMoistureMapProgram_.bindStorageBuffers(*moistureMap_)
        .spec(maxComputeBlockSize, moistRandomSeed)
        .pushConstantsRange(sizeof(ComputeMapConstants))
        .create();
    computeWaterMapProgram_.bindStorageBuffers(*waterMap_)
        .spec(maxComputeBlockSize, waterRandomSeed)
        .pushConstantsRange(sizeof(ComputeMapConstants))
        .create();

    computeColorsProgram_.bindStorageBuffers(*heightMap_, *moistureMap_, *colors_)
        .spec(maxComputeBlockSize)
        .pushConstantsRange(sizeof(ComputeColorsConstants))
        .create();

    computeVerticesProgram_.bindStorageBuffers(*heightMap_, *moistureMap_, *vertices_, *normals_)
        .spec(maxComputeBlockSize)
        .pushConstantsRange(sizeof(ComputeVerticesConstants))
        .create();

    initWaterFacesProgram_.bindStorageBuffers(*waterFaces_)
        .spec(maxComputeBlockSize)
        .pushConstantsRange(sizeof(initFacesConstants_))
        .create();

    computeWaterProgram_.bindStorageBuffers(*waterMap_, *waterVertices_, *waterNormals_)
        .spec(maxComputeBlockSize)
        .pushConstantsRange(sizeof(ComputeWaterConstants))
        .create();

    computeMXSkyMapProgram_.bindStorageImages(computeSkyImageViews_[NEGATIVE_X])
        .spec(maxComputeBlockSize, skyRandomSeed, NEGATIVE_X)
        .pushConstantsRange(sizeof(ComputeSkyMapConstants))
        .create();
    computePXSkyMapProgram_.bindStorageImages(computeSkyImageViews_[POSITIVE_X])
        .spec(maxComputeBlockSize, skyRandomSeed, POSITIVE_X)
        .pushConstantsRange(sizeof(ComputeSkyMapConstants))
        .create();
    computeMYSkyMapProgram_.bindStorageImages(computeSkyImageViews_[NEGATIVE_Y])
        .spec(maxComputeBlockSize, skyRandomSeed, NEGATIVE_Y)
        .pushConstantsRange(sizeof(ComputeSkyMapConstants))
        .create();
    computePYSkyMapProgram_.bindStorageImages(computeSkyImageViews_[POSITIVE_Y])
        .spec(maxComputeBlockSize, skyRandomSeed, POSITIVE_Y)
        .pushConstantsRange(sizeof(ComputeSkyMapConstants))
        .create();
    computePZSkyMapProgram_.bindStorageImages(computeSkyImageViews_[POSITIVE_Z])
        .spec(maxComputeBlockSize, skyRandomSeed, POSITIVE_Z)
        .pushConstantsRange(sizeof(ComputeSkyMapConstants))
        .create();
    computeMZSkyMapProgram_.bindStorageImages(computeSkyImageViews_[NEGATIVE_Z])
        .spec(maxComputeBlockSize, skyRandomSeed, NEGATIVE_Z)
        .pushConstantsRange(sizeof(ComputeSkyMapConstants))
        .create();

    computeQueue_.init(*device_);
    computeCommandPool_.init(*device_);

    allocated_ = true;

    initFaces();
    initWaterFaces();
    initImageLayouts();
    initCubeMapVertices();

    terrainFence_.init(*device_, true);
    waterFence_.init(*device_, true);
    skyFence_.init(*device_, true);

    computeTerrainCommandBuffer_ = computeCommandPool_.createCommandBuffer();
    computeWaterCommandBuffer_ = computeCommandPool_.createCommandBuffer();
    computeSkyMapsCommandBuffer_ = computeCommandPool_.createCommandBuffer();
}

void TerrainGeneratorGPU::generate(
    const float offsetX,
    const float offsetY,
    const float theta,
    vkw::Semaphore& terrainSemaphore,
    vkw::Semaphore& waterSemaphore)
{
    if(!allocated_)
    {
        throw std::runtime_error("Terrain generation engine must be allocated before being used");
    }

    this->updateCommandBuffers(offsetX, offsetY, theta);

    skyFence_.waitAndReset();
    computeQueue_.submit(computeSkyMapsCommandBuffer_, {}, {}, {}, skyFence_);
    terrainFence_.waitAndReset();
    computeQueue_.submit(computeTerrainCommandBuffer_, {}, {}, {&terrainSemaphore}, terrainFence_);

    waterFence_.waitAndReset();
    computeQueue_.submit(computeWaterCommandBuffer_, {}, {}, {&waterSemaphore}, waterFence_);

#ifdef DEBUG_SKY_MAPS
    debugSkyMaps();
#endif
}

void TerrainGeneratorGPU::initFaces()
{
    const uint32_t halfFaceCount = (terrainSizeX_ - 1) * (terrainSizeY_ - 1);
    initFacesConstants_.dimX = terrainSizeX_ - 1;
    initFacesConstants_.dimY = terrainSizeY_ - 1;

    auto cmdBuffer = computeCommandPool_.createCommandBuffer();
    cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        .bindComputeProgram(initFacesProgram_, initFacesConstants_)
        .dispatch(vkw::divUp(halfFaceCount, maxComputeBlockSize))
        .end();

    computeQueue_.submit(cmdBuffer).waitIdle();
}

void TerrainGeneratorGPU::initWaterFaces()
{
    const uint32_t halfFaceCount = (waterSizeX_ - 1) * (waterSizeY_ - 1);
    initFacesConstants_.dimX = waterSizeX_ - 1;
    initFacesConstants_.dimY = waterSizeY_ - 1;

    auto cmdBuffer = computeCommandPool_.createCommandBuffer();
    cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        .bindComputeProgram(initWaterFacesProgram_, initFacesConstants_)
        .dispatch(vkw::divUp(halfFaceCount, maxComputeBlockSize))
        .end();

    computeQueue_.submit(cmdBuffer).waitIdle();
}

void TerrainGeneratorGPU::initImageLayouts()
{
    auto cmdBuffer = computeCommandPool_.createCommandBuffer();
    cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        .imageMemoryBarrier(
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vkw::createImageMemoryBarrier(
                *cubeMapImage_,
                VK_ACCESS_NONE,
                VK_ACCESS_NONE,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                0,
                6))
        .end();

    computeQueue_.submit(cmdBuffer).waitIdle();
}

void TerrainGeneratorGPU::initCubeMapVertices()
{
    const float baseDim = cubeMapDim();
    const float x0 = -0.5f * baseDim;
    const float x1 = 0.5f * baseDim;

    const float y0 = -0.5f * baseDim;
    const float y1 = 0.5f * baseDim;

    const float z0 = -0.5f * baseDim;
    const float z1 = 0.5f * baseDim;

    const std::vector<glm::vec3> cubeVertices{// X+ plane
                                              glm::vec3{x1, y1, z0},
                                              glm::vec3{x1, y1, z1},
                                              glm::vec3{x1, y0, z1},
                                              glm::vec3{x1, y1, z0},
                                              glm::vec3{x1, y0, z1},
                                              glm::vec3{x1, y0, z0},
                                              // X- plane
                                              glm::vec3{x0, y0, z0},
                                              glm::vec3{x0, y0, z1},
                                              glm::vec3{x0, y1, z1},
                                              glm::vec3{x0, y0, z0},
                                              glm::vec3{x0, y1, z1},
                                              glm::vec3{x0, y1, z0},
                                              //  Y+ plane
                                              glm::vec3{x0, y1, z0},
                                              glm::vec3{x0, y1, z1},
                                              glm::vec3{x1, y1, z1},
                                              glm::vec3{x0, y1, z0},
                                              glm::vec3{x1, y1, z1},
                                              glm::vec3{x1, y1, z0},
                                              // Y- plane
                                              glm::vec3{},
                                              glm::vec3{},
                                              glm::vec3{},
                                              glm::vec3{},
                                              glm::vec3{},
                                              glm::vec3{},
                                              // Z+ plane
                                              glm::vec3{x0, y1, z1},
                                              glm::vec3{x0, y0, z1},
                                              glm::vec3{x1, y0, z1},
                                              glm::vec3{x0, y1, z1},
                                              glm::vec3{x1, y0, z1},
                                              glm::vec3{x1, y1, z1},
                                              // Z- plane
                                              glm::vec3{x0, y1, z0},
                                              glm::vec3{x1, y1, z0},
                                              glm::vec3{x1, y0, z0},
                                              glm::vec3{x0, y1, z0},
                                              glm::vec3{x1, y0, z0},
                                              glm::vec3{x0, y0, z0}};
    cubeMapVertexMemory_.copyFromHost<glm::vec3>(&cubeVertices[0], 0, cubeVertices.size());
}

void TerrainGeneratorGPU::updateCommandBuffers(
    const float offsetX, const float offsetY, const float theta)
{
    // Terrain generation
    const float baseDim = refDist_ / terrainResolution_;

    ComputeMapConstants computeHeightMapConstants;
    computeHeightMapConstants.sizeX = terrainSizeX_;
    computeHeightMapConstants.sizeY = terrainSizeY_;
    computeHeightMapConstants.octaves = 10;
    computeHeightMapConstants.waveLength = 0.2f * baseDim;
    computeHeightMapConstants.offX = offsetX / terrainResolution_;
    computeHeightMapConstants.offY = offsetY / terrainResolution_;
    computeHeightMapConstants.theta = glm::radians(theta);

    ComputeMapConstants computeMoistureMapConstants;
    computeMoistureMapConstants.sizeX = terrainSizeX_;
    computeMoistureMapConstants.sizeY = terrainSizeY_;
    computeMoistureMapConstants.octaves = 8;
    computeMoistureMapConstants.waveLength = 0.5f * baseDim;
    computeMoistureMapConstants.offX = offsetX / terrainResolution_;
    computeMoistureMapConstants.offY = offsetY / terrainResolution_;
    computeMoistureMapConstants.theta = glm::radians(theta);

    ComputeColorsConstants computeColorsConstants;
    computeColorsConstants.pointCount = terrainSizeX_ * terrainSizeY_;

    ComputeVerticesConstants computeVerticesConstants;
    computeVerticesConstants.sizeX = terrainSizeX_;
    computeVerticesConstants.sizeY = terrainSizeY_;
    computeVerticesConstants.triangleRes = terrainResolution_;
    computeVerticesConstants.zScale = verticalScale_;

    computeTerrainCommandBuffer_.reset();
    computeTerrainCommandBuffer_.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        .bindComputeProgram(computeHeightMapProgram_, computeHeightMapConstants)
        .dispatch(vkw::divUp(terrainSizeX_ * terrainSizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeMoistureMapProgram_, computeMoistureMapConstants)
        .dispatch(vkw::divUp(terrainSizeX_ * terrainSizeY_, maxComputeBlockSize))
        .bufferMemoryBarriers(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vkw::createBufferMemoryBarrier(
                *heightMap_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            vkw::createBufferMemoryBarrier(
                *moistureMap_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT))
        .bindComputeProgram(computeColorsProgram_, computeColorsConstants)
        .dispatch(vkw::divUp(terrainSizeX_ * terrainSizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeVerticesProgram_, computeVerticesConstants)
        .dispatch(vkw::divUp(terrainSizeX_ * terrainSizeY_, maxComputeBlockSize))
        .end();

    // Water generation
    ComputeMapConstants computeWaterMapConstants;
    computeWaterMapConstants.sizeX = waterSizeX_;
    computeWaterMapConstants.sizeY = waterSizeY_;
    computeWaterMapConstants.octaves = 2;
    computeWaterMapConstants.waveLength = 0.01f * baseDim;
    computeWaterMapConstants.offX = offsetX / waterResolution_;
    computeWaterMapConstants.offY = offsetY / waterResolution_;
    computeWaterMapConstants.theta = glm::radians(theta);

    ComputeWaterConstants computeWaterConstants;
    computeWaterConstants.sizeX = waterSizeX_;
    computeWaterConstants.sizeY = waterSizeY_;
    computeWaterConstants.triangleRes = waterResolution_;
    computeWaterConstants.zScale = verticalScale_;

    computeWaterCommandBuffer_.reset();
    computeWaterCommandBuffer_.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        .bindComputeProgram(computeHeightMapProgram_, computeHeightMapConstants)
        .dispatch(vkw::divUp(terrainSizeX_ * terrainSizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeMoistureMapProgram_, computeMoistureMapConstants)
        .dispatch(vkw::divUp(terrainSizeX_ * terrainSizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeWaterMapProgram_, computeWaterMapConstants)
        .dispatch(vkw::divUp(terrainSizeX_ * terrainSizeY_, maxComputeBlockSize))
        .bufferMemoryBarriers(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vkw::createBufferMemoryBarrier(
                *heightMap_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            vkw::createBufferMemoryBarrier(
                *moistureMap_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            vkw::createBufferMemoryBarrier(
                *waterMap_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT))
        .bindComputeProgram(computeColorsProgram_, computeColorsConstants)
        .dispatch(vkw::divUp(terrainSizeX_ * terrainSizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeVerticesProgram_, computeVerticesConstants)
        .dispatch(vkw::divUp(terrainSizeX_ * terrainSizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeWaterProgram_, computeWaterConstants)
        .dispatch(vkw::divUp(waterSizeX_ * waterSizeY_, maxComputeBlockSize))
        .end();

    // Skymaps generation
    ComputeSkyMapConstants computeSkyMapConstants;
    computeSkyMapConstants.mapSize = cubeMapSize;
    computeSkyMapConstants.octaves = 10;
    computeSkyMapConstants.waveLength = 1.0f;
    computeSkyMapConstants.theta = glm::radians(theta);

    computeSkyMapsCommandBuffer_.reset();
    computeSkyMapsCommandBuffer_.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        .bindComputeProgram(computeMXSkyMapProgram_, computeSkyMapConstants)
        .dispatch(vkw::divUp(cubeMapSize * cubeMapSize, maxComputeBlockSize))
        .bindComputeProgram(computePXSkyMapProgram_, computeSkyMapConstants)
        .dispatch(vkw::divUp(cubeMapSize * cubeMapSize, maxComputeBlockSize))
        .bindComputeProgram(computeMYSkyMapProgram_, computeSkyMapConstants)
        .dispatch(vkw::divUp(cubeMapSize * cubeMapSize, maxComputeBlockSize))
        .bindComputeProgram(computePYSkyMapProgram_, computeSkyMapConstants)
        .dispatch(vkw::divUp(cubeMapSize * cubeMapSize, maxComputeBlockSize))
        .bindComputeProgram(computePZSkyMapProgram_, computeSkyMapConstants)
        .dispatch(vkw::divUp(cubeMapSize * cubeMapSize, maxComputeBlockSize))
        .bindComputeProgram(computeMZSkyMapProgram_, computeSkyMapConstants)
        .dispatch(vkw::divUp(cubeMapSize * cubeMapSize, maxComputeBlockSize))
        .end();
}

#ifdef DEBUG_SKY_MAPS
void TerrainGeneratorGPU::debugSkyMaps()
{
    device_->waitIdle();

    std::vector<glm::vec4> map;
    map.resize(cubeMapSize * cubeMapSize);

    const uint32_t cubeMapRes = cubeMapSize * cubeMapSize;
    for(uint32_t id = 0; id < mapCount; ++id)
    {
        std::array<VkBufferImageCopy, 1> copyInfo
            = {{skyMapStaging_->getOffset(),
                cubeMapSize,
                cubeMapSize,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, id, 1}, // Subresource range
                {0, 0, 0},                             // offset
                {cubeMapSize, cubeMapSize, 1}}};       // Extent
        auto cmdBuffer = computeCommandPool_.createCommandBuffer();
        cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
            .imageMemoryBarrier(
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                vkw::createImageMemoryBarrier(
                    *cubeMapImage_,
                    0,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0,
                    1,
                    id,
                    1))
            .copyImageToBuffer(
                *cubeMapImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *skyMapStaging_, copyInfo)
            .imageMemoryBarrier(
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                vkw::createImageMemoryBarrier(
                    *cubeMapImage_,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    0,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0,
                    1,
                    id,
                    1))
            .end();
        computeQueue_.submit(cmdBuffer).waitIdle();
        stagingMem_.copyFromDevice<glm::vec4>(map.data(), skyMapStaging_->getOffset(), cubeMapRes);

        std::vector<glm::vec4> imgData;
        imgData.resize(4 * cubeMapRes);
        for(size_t i = 0; i < cubeMapRes; ++i)
        {
            const auto& col = map[i];
            imgData[i] = 255.0f * glm::vec4(col.b, col.g, col.r, col.a);
        }

        // Create image
        char imgName[512];
        sprintf(imgName, "img_%d.png", int(id));
        cv::Mat img(cubeMapSize, cubeMapSize, CV_32FC4, imgData.data());
        cv::imwrite(imgName, img);
    }
}
#endif
} // namespace cg