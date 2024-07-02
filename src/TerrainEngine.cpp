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
    , instance_{window_}
    , device_{instance_}
    , width_{initWidth}
    , height_{initHeight}
    , generator_{device_}
{
    const auto deviceFeatures = device_.getFeatures();
    const bool clipPlaneSupported = deviceFeatures.shaderClipDistance;
    if(!clipPlaneSupported)
    {
        throw std::runtime_error("Clip plane not supported on the selected device");
    }
}

void TerrainEngine::prepare()
{
    graphicsCommandPool_.init(device_);
    graphicsQueue_.init(device_);
    presentQueue_.init(device_);

    initStorage();
    initGraphicsPipeline();

    reflectionFrameBufferUpdatedEvent_.init(device_);
    refractionFrameBufferUpdatedEvent_.init(device_);

    allocateUBO(swapchain_.imageCount());
    allocateDescriptorPools(swapchain_.imageCount());
    allocateGraphicsCommandBuffers(swapchain_.imageCount());

    terrainGeneratedSemaphore_.init(device_);
    waterGeneratedSemaphore_.init(device_);
    imageAvailableSemaphore_.init(device_);
    renderFinishedSemaphore_.init(device_);

    graphicsFence_.init(device_, true);
}

void TerrainEngine::renderFrame()
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
    const glm::vec3 pos{0.0f, 0.0f, h};

    graphicsFence_.waitAndReset();
    uint32_t imageIndex;
    auto res = swapchain_.getNextImage(imageIndex, imageAvailableSemaphore_);
    if(res == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapchain();
        return;
    }

    MatrixBlock mvp{glm::mat4{1.0f}, glm::mat4{1.0f}};
    mvp.view = glm::lookAt(pos, pos + dir, glm::vec3{0.0f, 0.0f, 1.0f});
    mvp.proj
        = glm::perspective(glm::radians(fov_), float(width_) / float(height_), 0.1f, farDistance_);
    uboMemory_->copyFromHost<MatrixBlock>(&mvp, imageIndex * sizeof(MatrixBlock), 1);

    generator_.generate(
        offsetX_, offsetY_, theta_, terrainGeneratedSemaphore_, waterGeneratedSemaphore_);

    graphicsQueue_.submit(
        reflectionCommandBuffers_[imageIndex],
        {&terrainGeneratedSemaphore_},
        {VK_PIPELINE_STAGE_VERTEX_INPUT_BIT},
        {});
    graphicsQueue_.submit(
        refractionCommandBuffers_[imageIndex], {}, {VK_PIPELINE_STAGE_VERTEX_INPUT_BIT}, {});
    graphicsQueue_.submit(
        graphicsCommandBuffers_[imageIndex],
        {&imageAvailableSemaphore_, &waterGeneratedSemaphore_},
        {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT},
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

    fprintf(stdout, "[DEBUG]\tsizeX = %u\n", sizeX);
    fprintf(stdout, "[DEBUG]\tsizeY = %u\n", sizeY);
    generator_.initStorage(sizeX, sizeY);

    storageInitialized_ = true;
}

void TerrainEngine::initGraphicsPipeline()
{
    offscreenRenderpass_.init(device_);
    offscreenRenderpass_
        .addColorAttachment(
            colorFormat,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_SAMPLE_COUNT_1_BIT)
        .addDepthAttachment(
            depthStencilFormat,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_SAMPLE_COUNT_1_BIT)
        .addSubPass({0}, {1})
        .addSubpassDependency(
            VK_SUBPASS_EXTERNAL,
            0,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_NONE_KHR,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_DEPENDENCY_BY_REGION_BIT)
        .addSubpassDependency(
            0,
            VK_SUBPASS_EXTERNAL,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_DEPENDENCY_BY_REGION_BIT)
        .create();

    renderpass_.init(device_);
    renderpass_
        .addColorAttachment(
            colorFormat,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_SAMPLE_COUNT_1_BIT)
        .addDepthAttachment(
            depthStencilFormat,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
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

    graphicsLayout_.init(device_, 1);
    graphicsLayout_.getDescriptorSetlayoutInfo(0).addUniformBufferBinding(
        VK_SHADER_STAGE_VERTEX_BIT, 0, 1);
    terrainRenderConstantsOffset_ = graphicsLayout_.addPushConstantRange(
        VK_SHADER_STAGE_VERTEX_BIT, sizeof(TerrainRenderConstants));
    graphicsLayout_.create();

    // Offscreen renderings
    reflectionGraphicsPipeline_.init(device_);
    reflectionGraphicsPipeline_
        .addShaderStage(VK_SHADER_STAGE_VERTEX_BIT, "output/spv/terrainDisplay_vert.spv")
        .addShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, "output/spv/terrainDisplay_frag.spv")
        .addVertexBinding(0, sizeof(glm::vec3))
        .addVertexBinding(1, sizeof(glm::vec4))
        .addVertexBinding(2, sizeof(glm::vec3))
        .addVertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
        .addVertexAttribute(1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0)
        .addVertexAttribute(2, 2, VK_FORMAT_R32G32B32_SFLOAT, 0);
    reflectionGraphicsPipeline_.setViewport(0.0f, 0.0f, float(width_), float(height_))
        .setScissors(0, 0, width_, height_)
        .setPrimitiveType(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .cullFrontFaces(true)
        .createPipeline(offscreenRenderpass_, graphicsLayout_);
    refractionGraphicsPipeline_.init(device_);
    refractionGraphicsPipeline_
        .addShaderStage(VK_SHADER_STAGE_VERTEX_BIT, "output/spv/terrainDisplay_vert.spv")
        .addShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, "output/spv/terrainDisplay_frag.spv")
        .addVertexBinding(0, sizeof(glm::vec3))
        .addVertexBinding(1, sizeof(glm::vec4))
        .addVertexBinding(2, sizeof(glm::vec3))
        .addVertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
        .addVertexAttribute(1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0)
        .addVertexAttribute(2, 2, VK_FORMAT_R32G32B32_SFLOAT, 0);
    refractionGraphicsPipeline_.setViewport(0.0f, 0.0f, float(width_), float(height_))
        .setScissors(0, 0, width_, height_)
        .setPrimitiveType(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .cullFrontFaces(false)
        .createPipeline(offscreenRenderpass_, graphicsLayout_);

    // Terrain rendering
    terrainGraphicsPipeline_.init(device_);
    terrainGraphicsPipeline_
        .addShaderStage(VK_SHADER_STAGE_VERTEX_BIT, "output/spv/terrainDisplay_vert.spv")
        .addShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, "output/spv/terrainDisplay_frag.spv")
        .addVertexBinding(0, sizeof(glm::vec3))
        .addVertexBinding(1, sizeof(glm::vec4))
        .addVertexBinding(2, sizeof(glm::vec3))
        .addVertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
        .addVertexAttribute(1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0)
        .addVertexAttribute(2, 2, VK_FORMAT_R32G32B32_SFLOAT, 0);
    terrainGraphicsPipeline_.setViewport(0.0f, 0.0f, float(width_), float(height_))
        .setScissors(0, 0, width_, height_)
        .setPrimitiveType(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .createPipeline(renderpass_, graphicsLayout_);

    // Water rendering
    waterLayout_.init(device_, 1);
    waterLayout_.getDescriptorSetlayoutInfo(0)
        .addUniformBufferBinding(VK_SHADER_STAGE_VERTEX_BIT, 0, 1)
        .addSamplerImageBinding(VK_SHADER_STAGE_FRAGMENT_BIT, 1, 1)
        .addSamplerImageBinding(VK_SHADER_STAGE_FRAGMENT_BIT, 2, 1);
    waterRenderConstantsOffset_ = waterLayout_.addPushConstantRange(
        VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(WaterRenderConstants));
    waterLayout_.create();

    waterGraphicsPipeline_.init(device_);
    waterGraphicsPipeline_
        .addShaderStage(VK_SHADER_STAGE_VERTEX_BIT, "output/spv/waterDisplay_vert.spv")
        .addShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, "output/spv/waterDisplay_frag.spv")
        .addVertexBinding(0, sizeof(glm::vec3))
        .addVertexBinding(1, sizeof(glm::vec3))
        .addVertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
        .addVertexAttribute(1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0);
    waterGraphicsPipeline_.setViewport(0.0f, 0.0f, float(width_), float(height_))
        .setScissors(0, 0, width_, height_)
        .enableBlending(false)
        .setPrimitiveType(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .createPipeline(renderpass_, waterLayout_);

    swapchain_.init(instance_, device_, renderpass_, width_, height_, colorFormat);
    allocateFramebuffers(width_, height_);
}

void TerrainEngine::allocateFramebuffers(const uint32_t w, const uint32_t h)
{
    reflectionColorAttachment_ = vkw::ColorRenderTarget(device_, w, h, VK_FORMAT_B8G8R8A8_SRGB);
    reflectionDepthAttachment_ = vkw::DepthRenderTarget(device_, w, h, VK_FORMAT_D24_UNORM_S8_UINT);
    reflectionFramebuffer_ = vkw::Framebuffer(device_, offscreenRenderpass_, w, h);
    reflectionFramebuffer_.addAttachment(reflectionColorAttachment_)
        .addAttachment(reflectionDepthAttachment_)
        .create();

    refractionColorAttachment_ = vkw::ColorRenderTarget(device_, w, h, VK_FORMAT_B8G8R8A8_SRGB);
    refractionDepthAttachment_ = vkw::DepthRenderTarget(device_, w, h, VK_FORMAT_D24_UNORM_S8_UINT);
    refractionFramebuffer_ = vkw::Framebuffer(device_, offscreenRenderpass_, w, h);
    refractionFramebuffer_.addAttachment(refractionColorAttachment_)
        .addAttachment(refractionDepthAttachment_)
        .create();
}

void TerrainEngine::recreateSwapchain()
{
    graphicsCommandBuffers_.clear();
    uboBuffers_.clear();
    uboMemory_.reset(nullptr);
    swapchain_.reCreate(width_, height_);

    allocateFramebuffers(swapchain_.getExtent().width, swapchain_.getExtent().height);

    const uint32_t imageCount = swapchain_.imageCount();
    allocateUBO(imageCount);
    allocateDescriptorPools(imageCount);
    allocateGraphicsCommandBuffers(imageCount);
    graphicsFence_ = vkw::Fence(device_, true);
}

void TerrainEngine::allocateUBO(const uint32_t imageCount)
{
    if(!uboBuffers_.empty())
    {
        uboBuffers_.clear();
    }

    uboMemory_.reset(new vkw::Memory(
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
    terrainDescriptorPools_.clear();
    terrainDescriptorPools_.resize(imageCount);

    waterDescriptorPools_.clear();
    waterDescriptorPools_.resize(imageCount);

    for(uint32_t i = 0; i < imageCount; ++i)
    {
        // Descriptor pool for terrain
        terrainDescriptorPools_[i].init(
            device_, graphicsLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        terrainDescriptorPools_[i].bindUniformBuffer(0, 0, uboBuffers_[i]->getFullSizeInfo());

        // Descriptor pool for water
        waterDescriptorPools_[i].init(
            device_, waterLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        waterDescriptorPools_[i]
            .bindUniformBuffer(0, 0, uboBuffers_[i]->getFullSizeInfo())
            .bindSamplerImage(
                0,
                1,
                {reflectionColorAttachment_.sampler(),
                 reflectionColorAttachment_.imageView(),
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL})
            .bindSamplerImage(
                0,
                2,
                {refractionColorAttachment_.sampler(),
                 refractionColorAttachment_.imageView(),
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    }
}

void TerrainEngine::allocateGraphicsCommandBuffers(const uint32_t imageCount)
{
    static constexpr glm::vec4 clearColor{0.259f, 0.557f, 0.914f, 1.0f};

    reflectionCommandBuffers_.clear();
    reflectionCommandBuffers_ = graphicsCommandPool_.createCommandBuffers(imageCount);
    for(uint32_t i = 0; i < imageCount; ++i)
    {
        const uint32_t faceCount = generator_.faceCount();
        const float waterHeight
            = 0.25 + 2.0f * baseResolution_; // TODO : get this from terrain generator

        TerrainRenderConstants terrainRenderConstants;
        terrainRenderConstants.model = glm::transpose(glm::mat4{
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, -1.0f, 2.0f * waterHeight},
            {0.0f, 0.0f, 0.0f, 1.0f}});
        terrainRenderConstants.clipPlane = glm::vec4(0.0f, 0.0f, -1.0f, waterHeight);

        auto& cmdBuffer = reflectionCommandBuffers_[i];
        cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)
            .beginRenderPass(
                offscreenRenderpass_,
                reflectionFramebuffer_.getHandle(),
                VkOffset2D{0, 0},
                reflectionFramebuffer_.getExtent(),
                clearColor)
            .bindGraphicsPipeline(reflectionGraphicsPipeline_)
            .setViewport(0, 0, swapchain_.getExtent().width, swapchain_.getExtent().height)
            .setScissor({0, 0}, swapchain_.getExtent())
            .bindGraphicsDescriptorSets(graphicsLayout_, terrainDescriptorPools_[i])
            .pushConstants(
                graphicsLayout_,
                VK_SHADER_STAGE_VERTEX_BIT,
                terrainRenderConstantsOffset_,
                terrainRenderConstants)
            .bindVertexBuffer(0, generator_.vertices(), 0)
            .bindVertexBuffer(1, generator_.colors(), 0)
            .bindVertexBuffer(2, generator_.normals(), 0)
            .bindIndexBuffer(generator_.faces(), VK_INDEX_TYPE_UINT32)
            .drawIndexed(3 * faceCount, 1, 0, 0, 0)
            .endRenderPass()
            .setEvent(
                reflectionFrameBufferUpdatedEvent_, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
            .end();
    }

    refractionCommandBuffers_.clear();
    refractionCommandBuffers_ = graphicsCommandPool_.createCommandBuffers(imageCount);
    for(uint32_t i = 0; i < imageCount; ++i)
    {
        const uint32_t faceCount = generator_.faceCount();
        const float waterHeight
            = 0.25f + 2.0f * baseResolution_; // TODO : get this from terrain generator

        TerrainRenderConstants terrainRenderConstants;
        terrainRenderConstants.model = glm::transpose(glm::mat4{
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}});
        terrainRenderConstants.clipPlane = glm::vec4(0.0f, 0.0f, -1.0f, waterHeight);

        auto& cmdBuffer = refractionCommandBuffers_[i];
        cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)
            .beginRenderPass(
                offscreenRenderpass_,
                refractionFramebuffer_.getHandle(),
                VkOffset2D{0, 0},
                refractionFramebuffer_.getExtent(),
                glm::vec4(0.0f))
            .bindGraphicsPipeline(refractionGraphicsPipeline_)
            .setViewport(0, 0, swapchain_.getExtent().width, swapchain_.getExtent().height)
            .setScissor({0, 0}, swapchain_.getExtent())
            .bindGraphicsDescriptorSets(graphicsLayout_, terrainDescriptorPools_[i])
            .pushConstants(
                graphicsLayout_,
                VK_SHADER_STAGE_VERTEX_BIT,
                terrainRenderConstantsOffset_,
                terrainRenderConstants)
            .bindVertexBuffer(0, generator_.vertices(), 0)
            .bindVertexBuffer(1, generator_.colors(), 0)
            .bindVertexBuffer(2, generator_.normals(), 0)
            .bindIndexBuffer(generator_.faces(), VK_INDEX_TYPE_UINT32)
            .drawIndexed(3 * faceCount, 1, 0, 0, 0)
            .endRenderPass()
            .setEvent(
                refractionFrameBufferUpdatedEvent_, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
            .end();
    }

    graphicsCommandBuffers_.clear();
    graphicsCommandBuffers_ = graphicsCommandPool_.createCommandBuffers(imageCount);
    for(uint32_t i = 0; i < imageCount; ++i)
    {
        const uint32_t faceCount = generator_.faceCount();
        const uint32_t waterFaceCount = generator_.waterFacesCount();

        TerrainRenderConstants terrainRenderConstants;
        terrainRenderConstants.model = glm::mat4(1.0f);
        terrainRenderConstants.clipPlane = glm::vec4(0.0f);

        WaterRenderConstants waterRenderConstants;
        waterRenderConstants.width = float(swapchain_.getExtent().width);
        waterRenderConstants.height = float(swapchain_.getExtent().height);
        waterRenderConstants.view = glm::transpose(glm::mat4{
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}});

        auto& graphicsCmdBuffer = graphicsCommandBuffers_[i];
        graphicsCmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)
            .waitEvent(
                reflectionFrameBufferUpdatedEvent_,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                {},
                {},
                {})
            .waitEvent(
                refractionFrameBufferUpdatedEvent_,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                {},
                {},
                {})
            .beginRenderPass(
                renderpass_,
                swapchain_.getFramebuffer(i),
                VkOffset2D{0, 0},
                swapchain_.getExtent(),
                clearColor)
            // Terrain
            .bindGraphicsPipeline(terrainGraphicsPipeline_)
            .setViewport(0, 0, swapchain_.getExtent().width, swapchain_.getExtent().height)
            .setScissor({0, 0}, swapchain_.getExtent())
            .bindGraphicsDescriptorSets(graphicsLayout_, terrainDescriptorPools_[i])
            .pushConstants(
                graphicsLayout_,
                VK_SHADER_STAGE_VERTEX_BIT,
                terrainRenderConstantsOffset_,
                terrainRenderConstants)
            .bindVertexBuffer(0, generator_.vertices(), 0)
            .bindVertexBuffer(1, generator_.colors(), 0)
            .bindVertexBuffer(2, generator_.normals(), 0)
            .bindIndexBuffer(generator_.faces(), VK_INDEX_TYPE_UINT32)
            .drawIndexed(3 * faceCount, 1, 0, 0, 0)
            // Water
            .bindGraphicsPipeline(waterGraphicsPipeline_)
            .setViewport(0, 0, swapchain_.getExtent().width, swapchain_.getExtent().height)
            .setScissor({0, 0}, swapchain_.getExtent())
            .bindGraphicsDescriptorSets(waterLayout_, waterDescriptorPools_[i])
            .pushConstants(
                waterLayout_,
                VK_SHADER_STAGE_FRAGMENT_BIT,
                waterRenderConstantsOffset_,
                waterRenderConstants)
            .bindVertexBuffer(0, generator_.waterVertices(), 0)
            .bindVertexBuffer(1, generator_.waterNormals(), 0)
            .bindIndexBuffer(generator_.waterFaces(), VK_INDEX_TYPE_UINT32)
            .drawIndexed(3 * waterFaceCount, 1, 0, 0, 0)
            .endRenderPass()
            .end();
    }
}
} // namespace cg
