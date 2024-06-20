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
TerrainGeneratorGPU::TerrainGeneratorGPU(vk::Device& device)
    : device_{&device}
    , initFacesProgram_{device, "output/spv/initFaces_comp.spv"}
    , computeMapsProgram_{device, "output/spv/computeMaps_comp.spv"}
    , computeColorsProgram_{device, "output/spv/computeColors_comp.spv"}
    , computeVerticesProgram_{device, "output/spv/computeVertices_comp.spv"}
{}

void TerrainGeneratorGPU::initStorage(const uint32_t sizeX, const uint32_t sizeY)
{
    static constexpr double seedRange = 1000.0;

    // srand(time(NULL));
    const float heightRandomSeed = 2.0f * seedRange * double(rand()) / double(RAND_MAX);
    const float moistRandomSeed = 2.0f * seedRange * double(rand()) / double(RAND_MAX);

    sizeX_ = sizeX;
    sizeY_ = sizeY;

    const uint32_t mapSize = sizeX_ * sizeY_;
    mapsMemory_.init(*device_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    heightMap_ = &mapsMemory_.createBuffer<float>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mapSize);
    moistureMap_ = &mapsMemory_.createBuffer<float>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mapSize);
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
    vertexMemory_.allocate();
    facesMemory_.allocate();

    initFacesProgram_.bindBuffer(*faces_)
        .spec(maxComputeBlockSize)
        .pushConstantsRange(sizeof(initFacesConstants_))
        .create();

    computeMapsProgram_.bindBuffer(*heightMap_)
        .bindBuffer(*moistureMap_)
        .spec(maxComputeBlockSize, heightRandomSeed, moistRandomSeed)
        .pushConstantsRange(sizeof(computeMapConstants_))
        .create();

    computeColorsProgram_.bindBuffer(*heightMap_)
        .bindBuffer(*moistureMap_)
        .bindBuffer(*colors_)
        .spec(maxComputeBlockSize)
        .pushConstantsRange(sizeof(sizeof(computeColorsConstants_)))
        .create();

    computeVerticesProgram_.bindBuffer(*heightMap_)
        .bindBuffer(*moistureMap_)
        .bindBuffer(*vertices_)
        .bindBuffer(*normals_)
        .spec(maxComputeBlockSize)
        .pushConstantsRange(sizeof(computeVerticesConstants_))
        .create();

    computeQueue_.init(*device_);
    computeCommandPool_.init(*device_);

    allocated_ = true;

    initFaces();
}

void TerrainGeneratorGPU::generate(const float offsetX, const float offsetY, const float theta)
{
    if(!allocated_)
    {
        throw std::runtime_error("Terrain generation engine must be allocated before being used");
    }

    fprintf(stdout, "[DEBUG] Launching terrain generation\n");

    const auto start = std::chrono::high_resolution_clock::now();

    // Fill push constants
    const float baseDim = refDist_ / baseResolution_;
    computeMapConstants_.sizeX = sizeX_;
    computeMapConstants_.sizeY = sizeY_;
    computeMapConstants_.heightWaveLength = 0.2f * baseDim;
    computeMapConstants_.moistureWaveLength = 0.5f * baseDim;
    computeMapConstants_.heightOctaves = 10;
    computeMapConstants_.moistureOctaves = 6;
    computeMapConstants_.offX = offsetX / baseResolution_;
    computeMapConstants_.offY = offsetY / baseResolution_;
    computeMapConstants_.theta = glm::radians(theta);

    computeVerticesConstants_.sizeX = sizeX_;
    computeVerticesConstants_.sizeY = sizeY_;
    computeVerticesConstants_.triangleRes = baseResolution_;
    computeVerticesConstants_.zScale = 2.5f;

    computeColorsConstants_.pointCount = sizeX_ * sizeY_;

    auto cmdBuffer = computeCommandPool_.createCommandBuffer();
    cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        .bindComputeProgram(computeMapsProgram_, computeMapConstants_)
        .dispatch(vk::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        .bufferMemoryBarriers(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk::createBufferMemoryBarrier(
                *heightMap_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            vk::createBufferMemoryBarrier(
                *moistureMap_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT))
        .bindComputeProgram(computeColorsProgram_, computeColorsConstants_)
        .dispatch(vk::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        .bindComputeProgram(computeVerticesProgram_, computeVerticesConstants_)
        .dispatch(vk::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
#ifdef DEBUG_TERRAIN
        .bufferMemoryBarriers(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk::createBufferMemoryBarrier(
                *vertices_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            vk::createBufferMemoryBarrier(
                *colors_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            vk::createBufferMemoryBarrier(
                *normals_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            vk::createBufferMemoryBarrier(
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

    const auto stop = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    fprintf(stdout, "[DEBUG] Generation took : %f [ms]\n", (double) elapsed.count() / 1000.0f);
}

void TerrainGeneratorGPU::initFaces()
{
    const uint32_t halfFaceCount = (sizeX_ - 1) * (sizeY_ - 1);
    initFacesConstants_.dimX = sizeX_ - 1;
    initFacesConstants_.dimY = sizeY_ - 1;

    auto cmdBuffer = computeCommandPool_.createCommandBuffer();
    cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        .bindComputeProgram(initFacesProgram_, initFacesConstants_)
        .dispatch(vk::divUp(halfFaceCount, maxComputeBlockSize))
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