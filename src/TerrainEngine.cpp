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

namespace cg
{
TerrainEngine::TerrainEngine(
    GLFWwindow* window, const uint32_t initWidth, const uint32_t initHeight)
    : window_{window}
    , instance_{window}
    , device_{instance_}
    , width_{initWidth}
    , height_{initHeight}
    , generator_{device_}
{}

void TerrainEngine::prepare()
{
    graphicsCommandPool_.init(device_);
    graphicsQueue_.init(device_);
    presentQueue_.init(device_);

    initStorage();
    initGraphicsPipeline();

    allocateUBO(swapchain_.imageCount());
    allocateDescriptorPools(swapchain_.imageCount());
    allocateGraphicsCommandBuffers(swapchain_.imageCount());

    imageAvailableSemaphore_.init(device_);
    renderFinishedSemaphore_.init(device_);
    graphicsFence_.init(device_, true);
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
        generator_.generate(offsetX_, offsetY_, theta_);
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
    const auto sizeX = uint32_t((2.0f * d) / baseResolution_);
    const auto sizeY = uint32_t(farDistance_ / baseResolution_);
    generator_.initStorage(sizeX, sizeY);

    fprintf(stdout, "[DEBUG]\tsizeX = %u\n", sizeX);
    fprintf(stdout, "[DEBUG]\tsizeY = %u\n", sizeY);

    storageInitialized_ = true;
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

void TerrainEngine::recreateSwapchain()
{
    graphicsCommandBuffers_.clear();
    uboBuffers_.clear();
    uboMemory_.reset(nullptr);
    swapchain_.reCreate(width_, height_, colorFormat);

    const uint32_t imageCount = swapchain_.imageCount();
    allocateUBO(imageCount);
    allocateDescriptorPools(imageCount);
    allocateGraphicsCommandBuffers(imageCount);
    graphicsFence_ = vk::Fence(device_, true);
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
    graphicsPools_.clear();
    graphicsPools_.resize(imageCount);

    for(uint32_t i = 0; i < imageCount; ++i)
    {
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
        const uint32_t faceCount = generator_.faceCount();

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
            .bindVertexBuffer(0, generator_.vertices(), 0)
            .bindVertexBuffer(1, generator_.colors(), 0)
            .bindVertexBuffer(2, generator_.normals(), 0)
            .bindIndexBuffer(generator_.faces(), VK_INDEX_TYPE_UINT32)
            .drawIndexed(3 * faceCount, 1, 0, 0, 0)
            .endRenderPass()
            .end();
    }
}
} // namespace cg
