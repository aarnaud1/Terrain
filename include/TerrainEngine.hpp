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

#include "TerrainGeneratorGPU.hpp"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vkWrappers/wrappers.hpp>

namespace cg
{
class TerrainEngine
{
  public:
    TerrainEngine() = delete;
    TerrainEngine(GLFWwindow* window, const uint32_t initWidth, const uint32_t initHeight);

    TerrainEngine(const TerrainEngine&) = delete;
    TerrainEngine(TerrainEngine&&) = delete;

    TerrainEngine& operator=(const TerrainEngine&) = delete;
    TerrainEngine& operator=(TerrainEngine&&) = delete;

    ~TerrainEngine() { device_.waitIdle(); }

    void setRefDistance(const float dist) { generator_.setRefDistance(dist); }
    void setBaseResolution(const float res)
    {
        baseResolution_ = res;
        generator_.setBaseResolution(res);
        generator_.setWaterResolution(0.5f * res);
    }
    void setFarDistance(const float dist) { farDistance_ = dist; }
    void setFov(const float fov) { fov_ = fov; }

    void prepare();

    void renderFrame(const bool generateTerrain);

    void setOffset(const float offsetX, const float offsetY, const float theta)
    {
        offsetX_ = offsetX;
        offsetY_ = offsetY;
        theta_ = theta;
    }

  private:
    static constexpr VkFormat colorFormat = VK_FORMAT_B8G8R8A8_SRGB;
    static constexpr VkFormat depthStencilFormat = VK_FORMAT_D24_UNORM_S8_UINT;

    GLFWwindow* window_{nullptr};

    vkw::Instance instance_;
    vkw::Device device_;

    uint32_t width_{0};
    uint32_t height_{0};

    // Terrain generation data
    float baseResolution_{1.0f};
    float farDistance_{1.0f};
    float fov_{45.0f};

    bool storageInitialized_{false};

    TerrainGeneratorGPU generator_;

    struct MatrixBlock
    {
        glm::mat4 view;
        glm::mat4 proj;
    };
    std::unique_ptr<vkw::Memory> uboMemory_{nullptr};
    std::vector<vkw::Buffer<MatrixBlock>*> uboBuffers_{};

    // Command pools
    vkw::CommandPool<vkw::QueueFamilyType::GRAPHICS> graphicsCommandPool_{};

    // Device queues
    vkw::Queue<vkw::QueueFamilyType::GRAPHICS> graphicsQueue_{};
    vkw::Queue<vkw::QueueFamilyType::PRESENT> presentQueue_{};

    std::vector<vkw::CommandBuffer<vkw::QueueFamilyType::GRAPHICS>> graphicsCommandBuffers_{};
    std::vector<vkw::CommandBuffer<vkw::QueueFamilyType::GRAPHICS>> reflectionCommandBuffers_{};

    struct TerrainRenderConstants
    {
        glm::mat4 model;
        glm::vec4 clipPlane;
    };
    uint32_t terrainRenderConstantsOffset_;

    struct WaterRenderConstants
    {
        float width;
        float height;
    };
    uint32_t waterRenderConstantsOffset_;

    vkw::PipelineLayout graphicsLayout_{};
    vkw::GraphicsPipeline terrainGraphicsPipeline_{};
    vkw::GraphicsPipeline offScreenPipeline_{};
    std::vector<vkw::DescriptorPool> terrainDescriptorPools_{};
    std::vector<vkw::DescriptorPool> waterDescriptorPools_{};

    vkw::PipelineLayout waterLayout_{};
    vkw::GraphicsPipeline waterGraphicsPipeline_{};
    std::vector<vkw::DescriptorPool> waterPools_{};

    vkw::ColorRenderTarget reflectionColorAttachment_{};
    vkw::DepthRenderTarget reflectionDepthAttachment_{};
    vkw::Framebuffer reflectionFramebuffer_{};

    vkw::ColorRenderTarget refractionColorAttachment_{};
    vkw::DepthRenderTarget refractionDepthAttachment_{};
    vkw::Framebuffer refractionFramebuffer_{};

    vkw::RenderPass offscreenRenderpass_{};
    vkw::RenderPass renderpass_{};
    vkw::Swapchain swapchain_{};

    // Sync objects
    vkw::Semaphore imageAvailableSemaphore_{};
    vkw::Semaphore renderFinishedSemaphore_{};

    vkw::Fence graphicsFence_{};

    float offsetX_{0.0f};
    float offsetY_{0.0f};
    float theta_{0.0f};

    void initStorage();
    void initGraphicsPipeline();

    void allocateFramebuffers(const uint32_t w, const uint32_t h);
    void recreateSwapchain();
    void allocateUBO(const uint32_t imageCount);
    void allocateDescriptorPools(const uint32_t imageCount);
    void allocateGraphicsCommandBuffers(const uint32_t imageCount);

    void renderReflectionTexture(const uint32_t frameId);
    void renderRefractionTexture(const uint32_t frameId);
};
} // namespace cg