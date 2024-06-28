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

#include <GLFW/glfw3.h>
#include <cstdio>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

static void mainLoop(GLFWwindow* window, cg::TerrainEngine* engine);

static void errorCallback(int error, const char* msg);
static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
// static void mouseCallback(GLFWwindow* window, double xpos, double ypos);

float offsetX = 0.0f;
float offsetY = 0.0f;
float theta = 0.0f;

int main(int /*argc*/, char** /*argv*/)
{
    const uint32_t initWidth = 1024;
    const uint32_t initHeight = 768;

    if(!glfwInit())
    {
        fprintf(stderr, "Error initializing GLFW\n");
        return EXIT_FAILURE;
    }

    if(!glfwVulkanSupported())
    {
        fprintf(stderr, "Vulkan not supported\n");
        return EXIT_FAILURE;
    }
    glfwSetErrorCallback(errorCallback);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(initWidth, initHeight, "Terrain", nullptr, nullptr);
    if(!window)
    {
        fprintf(stderr, "Error creating window, terminating\n");
        return EXIT_FAILURE;
    }

    glfwSetKeyCallback(window, keyCallback);
    // glfwSetCursorPosCallback(window, mouseCallback);

    std::unique_ptr<cg::TerrainEngine> engine(new cg::TerrainEngine(window, initWidth, initHeight));
    engine->setRefDistance(40.0f);
    engine->setBaseResolution(0.02f);
    engine->setFarDistance(40.0f);
    engine->setFov(45.0f);
    engine->setVerticalScale(2.5f);
    engine->prepare();

    fprintf(stdout, "Starting the main loop\n");
    mainLoop(window, engine.get());
    engine.reset(nullptr); // Clear all Vulkan resources

    glfwDestroyWindow(window);
    glfwTerminate();

    return EXIT_SUCCESS;
}

// -------------------------------------------------------------------------------------------------

bool incrementX = false;
bool incrementY = false;
bool decrementX = false;
bool decrementY = false;
bool incrementTheta = false;

void mainLoop(GLFWwindow* window, cg::TerrainEngine* engine)
{
    engine->renderFrame(true);
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        if(incrementX)
        {
            offsetX += 0.1f;
        }
        if(incrementY)
        {
            offsetY += 0.1f;
        }

        if(decrementX)
        {
            offsetX -= 0.1f;
        }
        if(decrementY)
        {
            offsetY -= 0.1f;
        }

        engine->setOffset(offsetX, offsetY, theta);
        engine->renderFrame(incrementX || incrementY || decrementX || decrementY || incrementTheta);

        if(incrementTheta)
        {
            incrementTheta = false;
        }
    }
}

void errorCallback(int, const char* msg) { fprintf(stderr, "GLFW error : %s\n", msg); }

void keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    if(key == GLFW_KEY_RIGHT && (action == GLFW_PRESS))
    {
        incrementX = true;
    }
    if(key == GLFW_KEY_RIGHT && (action == GLFW_RELEASE))
    {
        incrementX = false;
    }

    if(key == GLFW_KEY_LEFT && (action == GLFW_PRESS))
    {
        decrementX = true;
    }
    if(key == GLFW_KEY_LEFT && (action == GLFW_RELEASE))
    {
        decrementX = false;
    }

    if(key == GLFW_KEY_UP && (action == GLFW_PRESS))
    {
        incrementY = true;
    }
    if(key == GLFW_KEY_UP && (action == GLFW_RELEASE))
    {
        incrementY = false;
    }

    if(key == GLFW_KEY_DOWN && (action == GLFW_PRESS))
    {
        decrementY = true;
    }
    if(key == GLFW_KEY_DOWN && (action == GLFW_RELEASE))
    {
        decrementY = false;
    }
}

// void mouseCallback(GLFWwindow* window, double xpos, double ypos) 
// {
//     const float thetaScale = xPos - 
// }
