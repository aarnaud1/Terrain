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
    static constexpr uint32_t maxComputeBlockSize = 256;
    static constexpr VkFormat colorFormat = VK_FORMAT_B8G8R8A8_SRGB;
    static constexpr VkFormat depthStencilFormat = VK_FORMAT_D24_UNORM_S8_UINT;

    GLFWwindow* window_{nullptr};

    vk::Instance instance_;
    vk::Device device_;

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
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
    };
    std::unique_ptr<vk::Memory> uboMemory_{nullptr};
    std::vector<vk::Buffer<MatrixBlock>*> uboBuffers_{};

    // Command pools
    vk::CommandPool<vk::QueueFamilyType::GRAPHICS> graphicsCommandPool_{};

    // Device queues
    vk::Queue<vk::QueueFamilyType::GRAPHICS> graphicsQueue_{};
    vk::Queue<vk::QueueFamilyType::PRESENT> presentQueue_{};

    std::vector<vk::CommandBuffer<vk::QueueFamilyType::GRAPHICS>> graphicsCommandBuffers_{};

    vk::PipelineLayout graphicsLayout_{};
    vk::GraphicsPipeline graphicsPipeline_{};
    std::vector<vk::DescriptorPool> graphicsPools_{};

    vk::PipelineLayout waterLayout_{};
    vk::GraphicsPipeline waterGraphicsPipeline_{};
    std::vector<vk::DescriptorPool> waterPools_{};

    vk::RenderPass renderpass_{};
    vk::Swapchain swapchain_{};

    // Sync objects
    vk::Semaphore imageAvailableSemaphore_{};
    vk::Semaphore renderFinishedSemaphore_{};

    vk::Fence graphicsFence_{};

    float offsetX_{0.0f};
    float offsetY_{0.0f};
    float theta_{0.0f};

    void initStorage();
    void initGraphicsPipeline();

    void recreateSwapchain();
    void allocateUBO(const uint32_t imageCount);
    void allocateDescriptorPools(const uint32_t imageCount);
    void allocateGraphicsCommandBuffers(const uint32_t imageCount);
};
} // namespace cg