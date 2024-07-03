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
{}

void TerrainGeneratorGPU::initStorage(const uint32_t sizeX, const uint32_t sizeY)
{
    static constexpr double seedRange = 1000.0;

    srand(time(NULL));
    const float heightRandomSeed = 2.0f * seedRange * double(rand()) / double(RAND_MAX);
    const float moistRandomSeed = 2.0f * seedRange * double(rand()) / double(RAND_MAX);
    const float waterRandomSeed = 2.0f * seedRange * double(rand()) / double(RAND_MAX);

    sizeX_ = sizeX;
    sizeY_ = sizeY;

    waterSizeX_ = uint32_t((float(sizeX_) * terrainResolution_) / waterResolution_);
    waterSizeY_ = uint32_t((float(sizeY_) * terrainResolution_) / waterResolution_);
    const uint32_t waterVertexCount = waterSizeX_ * waterSizeY_;
    const uint32_t waterFaceCount = 2 * (waterSizeX_ - 1) * (waterSizeY_ - 1);

    const uint32_t mapSize = sizeX_ * sizeY_;
    mapsMemory_.init(*device_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    heightMap_ = &mapsMemory_.createBuffer<float>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mapSize);
    moistureMap_ = &mapsMemory_.createBuffer<float>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mapSize);
    waterMap_ = &mapsMemory_.createBuffer<float>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, waterSizeX_ * waterSizeY_);
    mapsMemory_.allocate();

    const uint32_t vertexCount = sizeX_ * sizeY_;
    vertexMemory_.init(*device_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    const uint32_t faceCount = 2 * (sizeX_ - 1) * (sizeY_ - 1);
    facesMemory_.init(*device_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

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

    faces_ = &facesMemory_.createBuffer<glm::uvec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, faceCount);
    waterFaces_ = &facesMemory_.createBuffer<glm::uvec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, waterFaceCount);

    vertexMemory_.allocate();
    facesMemory_.allocate();

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

    computeQueue_.init(*device_);
    computeCommandPool_.init(*device_);

    allocated_ = true;

    initFaces();
    initWaterFaces();

    terrainFence_.init(*device_, true);
    waterFence_.init(*device_, true);
    computeTerrainCommandBuffer_ = computeCommandPool_.createCommandBuffer();
    computeWaterCommandBuffer_ = computeCommandPool_.createCommandBuffer();
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

    terrainFence_.waitAndReset();
    computeQueue_.submit(computeTerrainCommandBuffer_, {}, {}, {&terrainSemaphore}, terrainFence_);

    waterFence_.waitAndReset();
    computeQueue_.submit(computeWaterCommandBuffer_, {}, {}, {&waterSemaphore}, waterFence_);
}

void TerrainGeneratorGPU::initFaces()
{
    const uint32_t halfFaceCount = (sizeX_ - 1) * (sizeY_ - 1);
    initFacesConstants_.dimX = sizeX_ - 1;
    initFacesConstants_.dimY = sizeY_ - 1;

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

void TerrainGeneratorGPU::updateCommandBuffers(
    const float offsetX, const float offsetY, const float theta)
{
    // Terrain generation
    const float baseDim = refDist_ / terrainResolution_;

    ComputeMapConstants computeHeightMapConstants;
    computeHeightMapConstants.sizeX = sizeX_;
    computeHeightMapConstants.sizeY = sizeY_;
    computeHeightMapConstants.octaves = 10;
    computeHeightMapConstants.waveLength = 0.2f * baseDim;
    computeHeightMapConstants.offX = offsetX / terrainResolution_;
    computeHeightMapConstants.offY = offsetY / terrainResolution_;
    computeHeightMapConstants.theta = glm::radians(theta);

    ComputeMapConstants computeMoistureMapConstants;
    computeMoistureMapConstants.sizeX = sizeX_;
    computeMoistureMapConstants.sizeY = sizeY_;
    computeMoistureMapConstants.octaves = 8;
    computeMoistureMapConstants.waveLength = 0.5f * baseDim;
    computeMoistureMapConstants.offX = offsetX / terrainResolution_;
    computeMoistureMapConstants.offY = offsetY / terrainResolution_;
    computeMoistureMapConstants.theta = glm::radians(theta);

    ComputeColorsConstants computeColorsConstants;
    computeColorsConstants.pointCount = sizeX_ * sizeY_;

    ComputeVerticesConstants computeVerticesConstants;
    computeVerticesConstants.sizeX = sizeX_;
    computeVerticesConstants.sizeY = sizeY_;
    computeVerticesConstants.triangleRes = terrainResolution_;
    computeVerticesConstants.zScale = verticalScale_;

    computeTerrainCommandBuffer_.reset();
    computeTerrainCommandBuffer_.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        .bindComputeProgram(computeHeightMapProgram_, computeHeightMapConstants)
        .dispatch(vkw::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeMoistureMapProgram_, computeMoistureMapConstants)
        .dispatch(vkw::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        .bufferMemoryBarriers(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vkw::createBufferMemoryBarrier(
                *heightMap_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            vkw::createBufferMemoryBarrier(
                *moistureMap_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT))
        .bindComputeProgram(computeColorsProgram_, computeColorsConstants)
        .dispatch(vkw::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeVerticesProgram_, computeVerticesConstants)
        .dispatch(vkw::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
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
        .dispatch(vkw::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeMoistureMapProgram_, computeMoistureMapConstants)
        .dispatch(vkw::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeWaterMapProgram_, computeWaterMapConstants)
        .dispatch(vkw::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
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
        .dispatch(vkw::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeVerticesProgram_, computeVerticesConstants)
        .dispatch(vkw::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeWaterProgram_, computeWaterConstants)
        .dispatch(vkw::divUp(waterSizeX_ * waterSizeY_, maxComputeBlockSize))
        .end();
}
} // namespace cg