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

#include "TerrainEngine.hpp"

#include <chrono>
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>

#ifdef DEBUG_BUFFERS
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
TerrainEngine::TerrainEngine(
    GLFWwindow* window, const uint32_t initWidth, const uint32_t initHeight)
    : window_{window}, instance_{window}, device_{instance_}, width_{initWidth}, height_{initHeight}
{}

void TerrainEngine::prepare()
{
    graphicsCommandPool_.init(device_);
    graphicsQueue_.init(device_);
    presentQueue_.init(device_);

    initStorage();
    initComputePipelines();
    initGraphicsPipeline();

    allocateMeshData(swapchain_.imageCount());
    allocateUBO(swapchain_.imageCount());
    allocateDescriptorPools(swapchain_.imageCount());
    allocateGraphicsCommandBuffers(swapchain_.imageCount());

    imageAvailableSemaphore_.init(device_);
    renderFinishedSemaphore_.init(device_);
    graphicsFence_.init(device_, true);
    computeFence_.init(device_, true);

    initFaces(swapchain_.imageCount());
}

void TerrainEngine::renderFrame(const bool generateTerrain)
{
    // Perform rendering
    const float theta = 90.0f;
    const float phi = -30.0f;
    const float h = 6.0f;

    const float cosTheta = glm::cos(glm::radians(theta));
    const float sinTheta = glm::sin(glm::radians(theta));

    const float cosPhi = glm::cos(glm::radians(phi));
    const float sinPhi = glm::sin(glm::radians(phi));

    const glm::vec3 vert{0.0f, 0.0f, 1.0f};
    const glm::vec3 dir{cosPhi * cosTheta, cosPhi * sinTheta, sinPhi};
    const glm::vec3 pos{0.0f /*offsetX_*/, 0.0f /*offsetY_*/, h};

    MatrixBlock mvp{glm::mat4{1.0f}, glm::mat4{1.0f}, glm::mat4{1.0f}};
    mvp.model = glm::mat4{1.0f};
    mvp.view = glm::lookAt(pos, pos + dir, glm::vec3{0.0f, 0.0f, 1.0f});
    mvp.proj = glm::perspective(glm::radians(fov_), float(width_) / float(height_), 0.1f, 30.0f);

    graphicsFence_.waitAndReset();
    uint32_t imageIndex;
    auto res = swapchain_.getNextImage(imageIndex, imageAvailableSemaphore_);
    if(res == VK_ERROR_OUT_OF_DATE_KHR /*|| (width_ != w) || (height_ != h)*/)
    {
        recreateSwapchain();
        return;
    }

    // if(generateTerrain)
    {
        genTerrain(imageIndex);
    }

    uboMemory_->copyFromHost<MatrixBlock>(&mvp, imageIndex * sizeof(MatrixBlock), 1);
    graphicsQueue_.submit(
        graphicsCommandBuffers_[imageIndex],
        {&imageAvailableSemaphore_},
        {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
        {&renderFinishedSemaphore_},
        graphicsFence_);
    presentQueue_.present(swapchain_, {&renderFinishedSemaphore_}, imageIndex);
}

void TerrainEngine::initStorage()
{
    fprintf(stdout, "[DEBUG] Initializing buffer storages\n");
    if(storageInitialized_)
    {
        return;
    }

    const float d = farDistance_ * glm::tan(glm::radians(fov_));
    sizeX_ = uint32_t((2.0f * d) / baseResolution_);
    sizeY_ = uint32_t(farDistance_ / baseResolution_);

    fprintf(stdout, "[DEBUG]\tsizeX_ = %u\n", sizeX_);
    fprintf(stdout, "[DEBUG]\tsizeY_ = %u\n", sizeY_);

    const uint32_t mapSize = sizeX_ * sizeY_;
    mapsMemory_.reset(new vk::Memory(device_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    heightMap_ = &mapsMemory_->createBuffer<float>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mapSize);
    moistureMap_ = &mapsMemory_->createBuffer<float>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mapSize);
    mapsMemory_->allocate();

#ifdef DEBUG_BUFFERS
    verticesStagingMem_.reset(new vk::Memory(
        device_, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    verticesStaging_ = &verticesStagingMem_->createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vertexCount);
    verticesStagingMem_->allocate();

    colorStagingMem_.reset(new vk::Memory(
        device_, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    colorsStaging_ = &colorStagingMem_->createBuffer<glm::vec4>(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vertexCount);
    colorStagingMem_->allocate();

    normalsStagingMem_.reset(new vk::Memory(
        device_, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    normalsStaging_ = &normalsStagingMem_->createBuffer<glm::vec3>(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vertexCount);
    normalsStagingMem_->allocate();

    facesStagingMem_.reset(new vk::Memory(
        device_, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    facesStaging_ = &facesStagingMem_->createBuffer<glm::uvec3>(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, faceCount);
    facesStagingMem_->allocate();
#endif

    storageInitialized_ = true;
}

void TerrainEngine::initComputePipelines()
{
    static constexpr double seedRange = 1000.0;

    // srand(time(NULL));
    const float heightRandomSeed = 2.0f * seedRange * double(rand()) / double(RAND_MAX);
    const float moistRandomSeed = 2.0f * seedRange * double(rand()) / double(RAND_MAX);

    fprintf(stdout, "[DEBUG] Initializing pipelines\n");

    // faces initialization
    initFacesLayout_.init(device_, 1);
    initFacesLayout_.getDescriptorSetlayoutInfo(0).addStorageBufferBinding(
        VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);

    initFacesConstantsOffset_ = initFacesLayout_.addPushConstantRange(
        VK_SHADER_STAGE_COMPUTE_BIT, sizeof(initFacesConstants_));
    initFacesLayout_.create();

    initFacesPipeline_.init(device_, "output/spv/initFaces_comp.spv");
    initFacesPipeline_.addSpec<uint32_t>(maxComputeBlockSize).createPipeline(initFacesLayout_);

    // Maps computation
    computeMapsLayout_.init(device_, 1);
    computeMapsLayout_.getDescriptorSetlayoutInfo(0)
        .addStorageBufferBinding(VK_SHADER_STAGE_COMPUTE_BIT, 0, 1)
        .addStorageBufferBinding(VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);

    computeMapConstantsOffset_ = computeMapsLayout_.addPushConstantRange(
        VK_SHADER_STAGE_COMPUTE_BIT, sizeof(computeMapConstants_));
    computeMapsLayout_.create();

    computeMapsPipeline_.init(device_, "output/spv/computeMaps_comp.spv");
    computeMapsPipeline_.addSpec<uint32_t>(maxComputeBlockSize)
        .addSpec<float>(heightRandomSeed)
        .addSpec<float>(moistRandomSeed)
        .createPipeline(computeMapsLayout_);

    // Colors computation
    computeColorsLayout_.init(device_, 1);
    computeColorsLayout_.getDescriptorSetlayoutInfo(0)
        .addStorageBufferBinding(VK_SHADER_STAGE_COMPUTE_BIT, 0, 1)
        .addStorageBufferBinding(VK_SHADER_STAGE_COMPUTE_BIT, 1, 1)
        .addStorageBufferBinding(VK_SHADER_STAGE_COMPUTE_BIT, 2, 1);

    computeColorsConstantsOffset_ = computeColorsLayout_.addPushConstantRange(
        VK_SHADER_STAGE_COMPUTE_BIT, sizeof(computeColorsConstants_));
    computeColorsLayout_.create();

    computeColorsPipeline_.init(device_, "output/spv/computeColors_comp.spv");
    computeColorsPipeline_.addSpec<uint32_t>(maxComputeBlockSize)
        .createPipeline(computeColorsLayout_);

    // Vertices computation
    computeVerticesLayout_.init(device_, 1);
    computeVerticesLayout_.getDescriptorSetlayoutInfo(0)
        .addStorageBufferBinding(VK_SHADER_STAGE_COMPUTE_BIT, 0, 1)
        .addStorageBufferBinding(VK_SHADER_STAGE_COMPUTE_BIT, 1, 1)
        .addStorageBufferBinding(VK_SHADER_STAGE_COMPUTE_BIT, 2, 1)
        .addStorageBufferBinding(VK_SHADER_STAGE_COMPUTE_BIT, 3, 1);

    computeVerticesConstantsOffset_ = computeVerticesLayout_.addPushConstantRange(
        VK_SHADER_STAGE_COMPUTE_BIT, sizeof(computeVerticesConstants_));
    computeVerticesLayout_.create();

    computeVerticesPipeline_.init(device_, "output/spv/computeVertices_comp.spv");
    computeVerticesPipeline_.addSpec<uint32_t>(maxComputeBlockSize)
        .createPipeline(computeVerticesLayout_);

    // Init compute command pool
    computeCommandPool_.init(device_);
    computeQueue_.init(device_);
}

void TerrainEngine::initGraphicsPipeline()
{
    graphicsLayout_.init(device_, 1);
    graphicsLayout_.getDescriptorSetlayoutInfo(0)
        .addUniformBufferBinding(VK_SHADER_STAGE_VERTEX_BIT, 0, 1)
        .addUniformBufferBinding(VK_SHADER_STAGE_FRAGMENT_BIT, 1, 1);
    graphicsLayout_.create();

    renderpass_.init(device_);
    renderpass_
        .addColorAttachment(
            colorFormat,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_SAMPLE_COUNT_1_BIT)
        .addDepthAttachment(
            depthStencilFormat,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_SAMPLE_COUNT_1_BIT)
        .addSubPass({0}, {1})
        .addSubpassDependency(
            VK_SUBPASS_EXTERNAL,
            0,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
        .create();

    graphicsPipeline_.init(device_);
    graphicsPipeline_
        .addShaderStage(VK_SHADER_STAGE_VERTEX_BIT, "output/spv/terrainDisplay_vert.spv")
        .addShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, "output/spv/terrainDisplay_frag.spv")
        .addVertexBinding(0, sizeof(glm::vec3))
        .addVertexBinding(1, sizeof(glm::vec4))
        .addVertexBinding(2, sizeof(glm::vec3))
        .addVertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
        .addVertexAttribute(1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0)
        .addVertexAttribute(2, 2, VK_FORMAT_R32G32B32_SFLOAT, 0);
    graphicsPipeline_.setViewport(0.0f, 0.0f, float(width_), float(height_))
        .setScissors(0, 0, width_, height_)
        .setPrimitiveType(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .createPipeline(renderpass_, graphicsLayout_);

    swapchain_.init(instance_, device_, renderpass_, width_, height_, colorFormat);
}

void TerrainEngine::initFaces(const uint32_t imageCount)
{
    const uint32_t halfFaceCount = (sizeX_ - 1) * (sizeY_ - 1);
    initFacesConstants_.dimX = sizeX_ - 1;
    initFacesConstants_.dimY = sizeY_ - 1;

    for(size_t i = 0; i < imageCount; ++i)
    {
        auto cmdBuffer = computeCommandPool_.createCommandBuffer();
        cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
            .bindComputePipeline(initFacesPipeline_)
            .bindComputeDescriptorSets(initFacesLayout_, initFacesPools_[i])
            .pushConstants(
                initFacesLayout_,
                VK_SHADER_STAGE_COMPUTE_BIT,
                initFacesConstantsOffset_,
                &initFacesConstants_)
            .dispatch(vk::divUp(halfFaceCount, maxComputeBlockSize))
            .end();

        computeQueue_.submit(cmdBuffer).waitIdle();
    }
}

void TerrainEngine::recreateSwapchain()
{
    graphicsCommandBuffers_.clear();
    uboBuffers_.clear();
    uboMemory_.reset(nullptr);
    swapchain_.reCreate(width_, height_, colorFormat);

    const uint32_t imageCount = swapchain_.imageCount();
    allocateMeshData(imageCount);
    allocateUBO(imageCount);
    allocateDescriptorPools(imageCount);
    allocateGraphicsCommandBuffers(imageCount);
    graphicsFence_ = vk::Fence(device_, true);

    initFaces(swapchain_.imageCount());
}

void TerrainEngine::allocateMeshData(const uint32_t imageCount)
{
    vertices_.clear();
    normals_.clear();
    colors_.clear();
    faces_.clear();

    vertexMemory_.reset(nullptr);
    facesMemory_.reset(nullptr);

    const uint32_t vertexCount = sizeX_ * sizeY_;
    vertexMemory_.reset(new vk::Memory(device_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

    const uint32_t faceCount = 2 * (sizeX_ - 1) * (sizeY_ - 1);
    facesMemory_.reset(new vk::Memory(device_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

    vertices_.resize(imageCount);
    normals_.resize(imageCount);
    colors_.resize(imageCount);
    faces_.resize(imageCount);

    for(size_t i = 0; i < imageCount; ++i)
    {
#ifdef DEBUG_BUFFERS
        vertices_[i] = &vertexMemory_->createBuffer<glm::vec3>(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vertexCount);
        normals_[i] = &vertexMemory_->createBuffer<glm::vec3>(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vertexCount);
        colors_[i] = &vertexMemory_->createBuffer<glm::vec4>(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vertexCount);
#else
        vertices_[i] = &vertexMemory_->createBuffer<glm::vec3>(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCount);
        normals_[i] = &vertexMemory_->createBuffer<glm::vec3>(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCount);
        colors_[i] = &vertexMemory_->createBuffer<glm::vec4>(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCount);
#endif

#ifdef DEBUG_BUFFERS
        faces_[i] = &facesMemory_->createBuffer<glm::uvec3>(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            faceCount);
#else
        faces_[i] = &facesMemory_->createBuffer<glm::uvec3>(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, faceCount);
#endif
    }
    vertexMemory_->allocate();
    facesMemory_->allocate();
}

void TerrainEngine::allocateUBO(const uint32_t imageCount)
{
    if(!uboBuffers_.empty())
    {
        uboBuffers_.clear();
    }

    uboMemory_.reset(new vk::Memory(
        device_,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    for(uint32_t i = 0; i < imageCount; ++i)
    {
        uboBuffers_.emplace_back(
            &uboMemory_->createBuffer<MatrixBlock>(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 1));
    }
    uboMemory_->allocate();
}

void TerrainEngine::allocateDescriptorPools(const uint32_t imageCount)
{
    initFacesPools_.clear();
    computeMapsPools_.clear();
    computeVerticesPools_.clear();
    computeColorsPools_.clear();

    graphicsPools_.clear();

    initFacesPools_.resize(imageCount);
    computeMapsPools_.resize(imageCount);
    computeVerticesPools_.resize(imageCount);
    computeColorsPools_.resize(imageCount);

    graphicsPools_.resize(imageCount);
    for(uint32_t i = 0; i < imageCount; ++i)
    {
        initFacesPools_[i].init(device_, initFacesLayout_, VK_SHADER_STAGE_COMPUTE_BIT);
        initFacesPools_[i].bindStorageBuffer(0, 0, faces_[i]->getFullSizeInfo());

        computeMapsPools_[i].init(device_, computeMapsLayout_, VK_SHADER_STAGE_COMPUTE_BIT);
        computeMapsPools_[i]
            .bindStorageBuffer(0, 0, heightMap_->getFullSizeInfo())
            .bindStorageBuffer(0, 1, moistureMap_->getFullSizeInfo());

        computeColorsPools_[i].init(device_, computeColorsLayout_, VK_SHADER_STAGE_COMPUTE_BIT);
        computeColorsPools_[i]
            .bindStorageBuffer(0, 0, heightMap_->getFullSizeInfo())
            .bindStorageBuffer(0, 1, moistureMap_->getFullSizeInfo())
            .bindStorageBuffer(0, 2, colors_[i]->getFullSizeInfo());

        computeVerticesPools_[i].init(device_, computeVerticesLayout_, VK_SHADER_STAGE_COMPUTE_BIT);
        computeVerticesPools_[i]
            .bindStorageBuffer(0, 0, heightMap_->getFullSizeInfo())
            .bindStorageBuffer(0, 1, moistureMap_->getFullSizeInfo())
            .bindStorageBuffer(0, 2, vertices_[i]->getFullSizeInfo())
            .bindStorageBuffer(0, 3, normals_[i]->getFullSizeInfo());

        graphicsPools_[i].init(
            device_, graphicsLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        graphicsPools_[i].bindUniformBuffer(0, 0, uboBuffers_[i]->getFullSizeInfo());
    }
}

void TerrainEngine::allocateGraphicsCommandBuffers(const uint32_t imageCount)
{
    if(!graphicsCommandBuffers_.empty())
    {
        graphicsCommandBuffers_.clear();
    }

    graphicsCommandBuffers_ = graphicsCommandPool_.createCommandBuffers(imageCount);
    for(uint32_t i = 0; i < imageCount; ++i)
    {
        const uint32_t faceCount = 2 * (sizeX_ - 1) * (sizeY_ - 1);

        auto& graphicsCmdBuffer = graphicsCommandBuffers_[i];
        graphicsCmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)
            .beginRenderPass(
                renderpass_,
                swapchain_.getFramebuffer(i),
                VkOffset2D{0, 0},
                swapchain_.getExtent(),
                glm::vec4{0.259f, 0.557f, 0.914f, 1.0f})
            .bindGraphicsPipeline(graphicsPipeline_)
            .setViewport(0, 0, swapchain_.getExtent().width, swapchain_.getExtent().height)
            .setScissor({0, 0}, swapchain_.getExtent())
            .bindGraphicsDescriptorSets(graphicsLayout_, graphicsPools_[i])
            .bindVertexBuffer(0, *vertices_[i], 0)
            .bindVertexBuffer(1, *colors_[i], 0)
            .bindVertexBuffer(2, *normals_[i], 0)
            .bindIndexBuffer(*faces_[i], VK_INDEX_TYPE_UINT32)
            .drawIndexed(3 * faceCount, 1, 0, 0, 0)
            .endRenderPass()
            .end();
    }
}

void TerrainEngine::genTerrain(const uint32_t id)
{
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
    computeMapConstants_.offX = offsetX_ / baseResolution_;
    computeMapConstants_.offY = offsetY_ / baseResolution_;
    computeMapConstants_.theta = glm::radians(theta_);

    computeVerticesConstants_.sizeX = sizeX_;
    computeVerticesConstants_.sizeY = sizeY_;
    computeVerticesConstants_.triangleRes = baseResolution_;
    computeVerticesConstants_.zScale = 2.5f;

    computeColorsConstants_.pointCount = sizeX_ * sizeY_;

    auto cmdBuffer = computeCommandPool_.createCommandBuffer();
    cmdBuffer
        .begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        // Compute maps
        .bindComputePipeline(computeMapsPipeline_)
        .bindComputeDescriptorSets(computeMapsLayout_, computeMapsPools_[id])
        .pushConstants(
            computeMapsLayout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            computeMapConstantsOffset_,
            &computeMapConstants_)
        .dispatch(vk::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        .bufferMemoryBarriers(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk::createBufferMemoryBarrier(
                *heightMap_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            vk::createBufferMemoryBarrier(
                *moistureMap_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT))
        // Compute colors
        .bindComputePipeline(computeColorsPipeline_)
        .bindComputeDescriptorSets(computeColorsLayout_, computeColorsPools_[id])
        .pushConstants(
            computeColorsLayout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            computeColorsConstantsOffset_,
            &computeColorsConstants_)
        .dispatch(vk::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
        // Compute vertices
        .bindComputePipeline(computeVerticesPipeline_)
        .bindComputeDescriptorSets(computeVerticesLayout_, computeVerticesPools_[id])
        .pushConstants(
            computeVerticesLayout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            computeVerticesConstantsOffset_,
            &computeVerticesConstants_)
        .dispatch(vk::divUp(sizeX_ * sizeY_, maxComputeBlockSize))
#ifdef DEBUG_BUFFERS
        .bufferMemoryBarriers(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk::createBufferMemoryBarrier(
                *vertices_[id], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            vk::createBufferMemoryBarrier(
                *colors_[id], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            vk::createBufferMemoryBarrier(
                *normals_[id], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            vk::createBufferMemoryBarrier(
                *faces_[id], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT))
        .copyBuffer(*vertices_[id], *verticesStaging_)
        .copyBuffer(*colors_[id], *colorsStaging_)
        .copyBuffer(*normals_[id], *normalsStaging_)
        .copyBuffer(*faces_[id], *facesStaging_)
#endif
        .end();

    computeQueue_.submit(cmdBuffer).waitIdle();

#ifdef DEBUG_BUFFERS
    static int imgCount = 0;
    const size_t vertexCount = sizeX_ * sizeY_;
    std::vector<glm::vec4> colors;
    colors.resize(vertexCount);

    colorStagingMem_->copyFromDevice<glm::vec4>(
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

    verticesStagingMem_->copyFromDevice<glm::vec3>(
        vertices.data(), verticesStaging_->getOffset(), vertexCount);
    normalsStagingMem_->copyFromDevice<glm::vec3>(
        normals.data(), normalsStaging_->getOffset(), vertexCount);
    facesStagingMem_->copyFromDevice<glm::uvec3>(
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
} // namespace cg

#ifdef DEBUG_BUFFERS
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