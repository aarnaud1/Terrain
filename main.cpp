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
static void mouseCallback(GLFWwindow* window, double xpos, double ypos);

int main(int /*argc*/, char** /*argv*/)
{
    int32_t initWidth;
    int32_t initHeight;

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

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    initWidth = mode->width;
    initHeight = mode->height;

    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(
        static_cast<uint32_t>(initWidth),
        static_cast<uint32_t>(initHeight),
        "Terrain",
        monitor,
        nullptr);
    if(!window)
    {
        fprintf(stderr, "Error creating window, terminating\n");
        return EXIT_FAILURE;
    }

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mouseCallback);

    std::unique_ptr<cg::TerrainEngine> engine(new cg::TerrainEngine(window, initWidth, initHeight));
    engine->setRefDistance(40.0f);
    engine->setBaseResolution(0.025f);
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

static constexpr float thetaInc = 10.0f;
static constexpr float phiInc = 10.0f;

double prevX, prevY;
float theta = 0.0f;
float phi = 30.0f;

static constexpr float dx = 0.25f;
static constexpr float dy = 0.25f;
float offsetX = 0.0f;
float offsetY = 0.0f;

void mainLoop(GLFWwindow* window, cg::TerrainEngine* engine)
{
    engine->renderFrame();

    glfwGetCursorPos(window, &prevX, &prevY);
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        const float cosTheta = glm::cos(glm::radians(theta));
        const float sinTheta = glm::sin(glm::radians(theta));

        if(incrementX)
        {
            offsetX += dx * cosTheta;
            offsetY += dy * sinTheta;
        }
        if(incrementY)
        {
            offsetX += dx * -sinTheta;
            offsetY += dy * cosTheta;
        }

        if(decrementX)
        {
            offsetX -= dx * cosTheta;
            offsetY -= dy * sinTheta;
        }
        if(decrementY)
        {
            offsetX -= dx * -sinTheta;
            offsetY -= dy * cosTheta;
        }

        engine->setOffset(offsetX, offsetY, theta, phi);
        engine->renderFrame();
    }
}

void errorCallback(int, const char* msg) { fprintf(stderr, "GLFW error : %s\n", msg); }

void keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    if((key == GLFW_KEY_RIGHT || key == GLFW_KEY_D) && (action == GLFW_PRESS))
    {
        incrementX = true;
    }
    if((key == GLFW_KEY_RIGHT || key == GLFW_KEY_D) && (action == GLFW_RELEASE))
    {
        incrementX = false;
    }

    if((key == GLFW_KEY_LEFT || key == GLFW_KEY_A) && (action == GLFW_PRESS))
    {
        decrementX = true;
    }
    if((key == GLFW_KEY_LEFT || key == GLFW_KEY_A) && (action == GLFW_RELEASE))
    {
        decrementX = false;
    }

    if((key == GLFW_KEY_UP || key == GLFW_KEY_W) && (action == GLFW_PRESS))
    {
        incrementY = true;
    }
    if((key == GLFW_KEY_UP || key == GLFW_KEY_W) && (action == GLFW_RELEASE))
    {
        incrementY = false;
    }

    if((key == GLFW_KEY_DOWN || key == GLFW_KEY_S) && (action == GLFW_PRESS))
    {
        decrementY = true;
    }
    if((key == GLFW_KEY_DOWN || key == GLFW_KEY_S) && (action == GLFW_RELEASE))
    {
        decrementY = false;
    }
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    int w, h;
    glfwGetWindowSize(window, &w, &h);

    const double dx = (xpos - prevX) / (w / 2);
    const double dy = (ypos - prevY) / (h / 2);

    theta -= thetaInc * float(dx);
    phi = glm::clamp(phi + phiInc * float(dy), -45.0f, 45.0f);
    prevX = xpos;
    prevY = ypos;
}
