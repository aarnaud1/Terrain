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

#ifdef DEBUG_TERRAIN
#    include <fstream>
#    include <iostream>
#    include <opencv2/core.hpp>
#    include <opencv2/highgui.hpp>
#    include <opencv2/imgcodecs.hpp>
#    include <tinyply.h>

namespace ply = tinyply;
static void saveSurface(
    const std::string& filename,
    const std::vector<glm::vec3>& xyz,
    const std::vector<glm::vec4>& rgb,
    const std::vector<glm::vec3>& normals,
    const std::vector<glm::uvec3>& triangles);
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
{}

void TerrainGeneratorGPU::initStorage(const uint32_t sizeX, const uint32_t sizeY)
{
    static constexpr double seedRange = 1000.0;

    // srand(time(NULL));
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

#ifdef DEBUG_TERRAIN
    vertices_ = &vertexMemory_.createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        vertexCount);
    normals_ = &vertexMemory_.createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        vertexCount);
    colors_ = &vertexMemory_.createBuffer<glm::vec4>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        vertexCount);

    faces_ = &facesMemory_.createBuffer<glm::uvec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        faceCount);

    verticesStagingMem_.init(
        *device_, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    verticesStaging_ = &verticesStagingMem_.createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vertexCount);
    verticesStagingMem_.allocate();

    colorStagingMem_.init(
        *device_, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    colorsStaging_ = &colorStagingMem_.createBuffer<glm::vec4>(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vertexCount);
    colorStagingMem_.allocate();

    normalsStagingMem_.init(
        *device_, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    normalsStaging_ = &normalsStagingMem_.createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vertexCount);
    normalsStagingMem_.allocate();

    facesStagingMem_.init(
        *device_, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    facesStaging_ = &facesStagingMem_.createBuffer<glm::uvec3>(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, faceCount);
    facesStagingMem_.allocate();
#else
    vertices_ = &vertexMemory_.createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCount);
    normals_ = &vertexMemory_.createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCount);
    colors_ = &vertexMemory_.createBuffer<glm::vec4>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCount);

    faces_ = &facesMemory_.createBuffer<glm::uvec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, faceCount);
#endif
    waterVertices_ = &vertexMemory_.createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, waterVertexCount);
    waterNormals_ = &vertexMemory_.createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, waterVertexCount);

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
        .pushConstantsRange(sizeof(computeColorsConstants_))
        .create();

    computeVerticesProgram_.bindStorageBuffers(*heightMap_, *moistureMap_, *vertices_, *normals_)
        .spec(maxComputeBlockSize)
        .pushConstantsRange(sizeof(computeVerticesConstants_))
        .create();

    initWaterFacesProgram_.bindStorageBuffers(*waterFaces_)
        .spec(maxComputeBlockSize)
        .pushConstantsRange(sizeof(initFacesConstants_))
        .create();

    computeWaterProgram_.bindStorageBuffers(*waterMap_, *waterVertices_, *waterNormals_)
        .spec(maxComputeBlockSize)
        .pushConstantsRange(sizeof(computeWaterConstants_))
        .create();

    computeQueue_.init(*device_);
    computeCommandPool_.init(*device_);

    allocated_ = true;

    initFaces();
    initWaterFaces();
}

void TerrainGeneratorGPU::generate(const float offsetX, const float offsetY, const float theta)
{
    if(!allocated_)
    {
        throw std::runtime_error("Terrain generation engine must be allocated before being used");
    }

    // fprintf(stdout, "[DEBUG] Launching terrain generation\n");
    // const auto start = std::chrono::high_resolution_clock::now();

    // Fill push constants
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

    ComputeMapConstants computeWaterMapConstants;
    computeWaterMapConstants.sizeX = waterSizeX_;
    computeWaterMapConstants.sizeY = waterSizeY_;
    computeWaterMapConstants.octaves = 2;
    computeWaterMapConstants.waveLength = 0.01f * baseDim;
    computeWaterMapConstants.offX = offsetX / waterResolution_;
    computeWaterMapConstants.offY = offsetY / waterResolution_;
    computeWaterMapConstants.theta = glm::radians(theta);

    computeVerticesConstants_.sizeX = sizeX_;
    computeVerticesConstants_.sizeY = sizeY_;
    computeVerticesConstants_.triangleRes = terrainResolution_;
    computeVerticesConstants_.zScale = 2.5f;

    computeColorsConstants_.pointCount = sizeX_ * sizeY_;

    computeWaterConstants_.sizeX = waterSizeX_;
    computeWaterConstants_.sizeY = waterSizeY_;
    computeWaterConstants_.triangleRes = waterResolution_;
    computeWaterConstants_.zScale = 2.5f;

    auto cmdBuffer = computeCommandPool_.createCommandBuffer();
    cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
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
                *moistureMap_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT))
        .bindComputeProgram(computeColorsProgram_, computeColorsConstants_)
        .dispatch(vkw::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeVerticesProgram_, computeVerticesConstants_)
        .dispatch(vkw::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeWaterProgram_, computeWaterConstants_)
        .dispatch(vkw::divUp(waterSizeX_ * waterSizeY_, maxComputeBlockSize))
#ifdef DEBUG_TERRAIN
        .bufferMemoryBarriers(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            vkw::createBufferMemoryBarrier(
                *vertices_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            vkw::createBufferMemoryBarrier(
                *colors_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            vkw::createBufferMemoryBarrier(
                *normals_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            vkw::createBufferMemoryBarrier(
                *faces_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT))
        .copyBuffer(*vertices_, *verticesStaging_)
        .copyBuffer(*colors_, *colorsStaging_)
        .copyBuffer(*normals_, *normalsStaging_)
        .copyBuffer(*faces_, *facesStaging_)
#endif
        .end();

    // TODO : use a sync object
    computeQueue_.submit(cmdBuffer).waitIdle();

#ifdef DEBUG_TERRAIN
    static int imgCount = 0;
    const size_t vertexCount = sizeX_ * sizeY_;
    std::vector<glm::vec4> colors;
    colors.resize(vertexCount);

    colorStagingMem_.copyFromDevice<glm::vec4>(
        colors.data(), colorsStaging_->getOffset(), vertexCount);

    std::vector<uint8_t> imgData;
    imgData.resize(3 * vertexCount);

    for(size_t i = 0; i < vertexCount; ++i)
    {
        const auto& c = colors[i];
        imgData[3 * i + 0] = 255.0f * c.b;
        imgData[3 * i + 1] = 255.0f * c.g;
        imgData[3 * i + 2] = 255.0f * c.r;
    }
    cv::Mat colorImg(sizeY_, sizeX_, CV_8UC3, imgData.data());
    char imgName[512];
    snprintf(imgName, 512, "img_%d.png", imgCount);
    cv::imwrite(imgName, colorImg);

    // Get vertices and faces
    std::vector<glm::vec3> vertices;
    vertices.resize(vertexCount);

    std::vector<glm::vec3> normals;
    normals.resize(vertexCount);

    const size_t faceCount = 2 * (sizeX_ - 1) * (sizeY_ - 1);
    std::vector<glm::uvec3> faces;
    faces.resize(faceCount);

    verticesStagingMem_.copyFromDevice<glm::vec3>(
        vertices.data(), verticesStaging_->getOffset(), vertexCount);
    normalsStagingMem_.copyFromDevice<glm::vec3>(
        normals.data(), normalsStaging_->getOffset(), vertexCount);
    facesStagingMem_.copyFromDevice<glm::uvec3>(
        faces.data(), facesStaging_->getOffset(), faceCount);

    char plyName[512];
    snprintf(plyName, 512, "output_%d.ply", imgCount);
    saveSurface(plyName, vertices, colors, normals, faces);

    imgCount++;
#endif

    // const auto stop = std::chrono::high_resolution_clock::now();
    // const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // fprintf(stdout, "[DEBUG] Generation took : %f [ms]\n", (double) elapsed.count() / 1000.0f);
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
} // namespace cg

#ifdef DEBUG_TERRAIN
void saveSurface(
    const std::string& filename,
    const std::vector<glm::vec3>& xyz,
    const std::vector<glm::vec4>& rgb,
    const std::vector<glm::vec3>& normals,
    const std::vector<glm::uvec3>& triangles)
{
    std::vector<uint8_t> colorData;
    colorData.resize(3 * rgb.size());

    for(size_t i = 0; i < rgb.size(); ++i)
    {
        colorData[3 * i + 0] = 255.0f * rgb[i].r;
        colorData[3 * i + 1] = 255.0f * rgb[i].g;
        colorData[3 * i + 2] = 255.0f * rgb[i].b;
    }

    if(rgb.size() != xyz.size())
    {
        throw std::runtime_error("Error exporting .ply fie : xyz and rgb sizes mismatch");
    }

    std::filebuf fb;
    fb.open(filename, std::ios::out | std::ios::binary);
    std::ostream os(&fb);
    if(os.fail())
    {
        throw std::runtime_error("Error exporting .ply file");
    }

    ply::PlyFile outFile;
    outFile.add_properties_to_element(
        "vertex",
        {"x", "y", "z"},
        ply::Type::FLOAT32,
        xyz.size(),
        reinterpret_cast<const uint8_t*>(xyz.data()),
        ply::Type::INVALID,
        0);
    outFile.add_properties_to_element(
        "vertex",
        {"nx", "ny", "nz"},
        ply::Type::FLOAT32,
        xyz.size(),
        reinterpret_cast<const uint8_t*>(normals.data()),
        ply::Type::INVALID,
        0);
    outFile.add_properties_to_element(
        "vertex",
        {"red", "green", "blue"},
        ply::Type::UINT8,
        colorData.size(),
        reinterpret_cast<const uint8_t*>(colorData.data()),
        ply::Type::INVALID,
        0);
    outFile.add_properties_to_element(
        "face",
        {"vertex_indices"},
        ply::Type::INT32,
        triangles.size(),
        reinterpret_cast<const uint8_t*>(triangles.data()),
        ply::Type::UINT8,
        3);

    outFile.get_comments().push_back("GPU Fusion V1.0");
    outFile.write(os, true);
}
#endif