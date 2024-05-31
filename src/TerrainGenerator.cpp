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

#include "TerrainGenerator.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <tinyply.h>

namespace ply = tinyply;

static void saveSurface(
    const std::string& filename,
    const std::vector<glm::vec3>& xyz,
    const std::vector<glm::vec4>& rgb,
    const std::vector<glm::ivec3>& triangles);

namespace cg
{
void TerrainGenerator::generateTile(
    const size_t sizeX, const size_t sizeY, const glm::vec2& offset, const float theta)
{
    sizeX_ = sizeX;
    sizeY_ = sizeY;

    generateMaps(offset, glm::radians(theta));
    computeVertices(offset, glm::radians(theta));

    computeColors();
    computeFaces();

#ifdef DEBUG_GENERATION
    debugMaps(offset, theta);
    exportPLY("terrain", offset, theta);
#endif
}

void TerrainGenerator::generateTile(
    const glm::vec2& position, const float far, const float fov, const float theta)
{
    const float d = far * glm::tan(glm::radians(fov));
    // const float D = far / glm::cos(glm::radians(fov));

    sizeX_ = size_t((2.0f * d) / resolution_.x);
    sizeY_ = size_t(far / resolution_.y);
    fprintf(stdout, "[DEBUG]: sizeX_ = %zu\n", sizeX_);
    fprintf(stdout, "[DEBUG]: sizeY_ = %zu\n", sizeY_);

    generateMaps(position, glm::radians(theta));
    computeVertices(position, glm::radians(theta));

    computeColors();
    computeFaces();

#ifdef DEBUG_GENERATION
    debugMaps(position, theta);
    exportPLY("terrain", position, theta);
#endif
}

void TerrainGenerator::generateMaps(const glm::vec2& offset, const float theta)
{
    const double baseHeightWaveLength = heightWaveLength_;
    const double baseMoistureWaveLength = moistureWaveLength_;

    const double offX = offset.x / resolution_.x;
    const double offY = offset.y / resolution_.y;

    heightMap_.resize(sizeX_ * sizeY_);
    heightMapGenerator_.gen2DArray(
        heightMap_.data(), sizeX_, sizeY_, baseHeightWaveLength, offX, offY, theta);

    moistureMap_.resize(sizeX_ * sizeY_);
    moistureMapGenerator_.gen2DArray(
        moistureMap_.data(), sizeX_, sizeY_, baseMoistureWaveLength, offX, offY, theta);

    // Generate colors
    colorMap_.resize(sizeX_ * sizeY_);

    const auto biomeLookup = cv::imread("assets/biome.png", cv::IMREAD_ANYCOLOR);
    const auto* imgData = (ImgPixel*) biomeLookup.data;

    for(size_t i = 0; i < heightMap_.size(); ++i)
    {
        const float height = glm::clamp(heightMap_[i], 0.0, 1.0);
        const float moist = glm::clamp(moistureMap_[i], 0.0, 1.0);
        const int posY = glm::clamp(int(100.0f * height), 0, 99);
        const int posX = glm::clamp(int(100.0f * moist), 0, 99);
        const auto& px = imgData[(99 - posY) * 100 + posX];

        colorMap_[i] = px;
    }
}
void TerrainGenerator::reshapeMap(std::vector<double>& map)
{
    for(size_t i = 0; i < map.size(); ++i)
    {
        map[i] = std::pow(map[i], 1.3f);
    }
}

void TerrainGenerator::normalizeMap(std::vector<double>& map)
{
    double minValue = map[0];
    double maxValue = map[0];
    for(size_t i = 0; i < map.size(); ++i)
    {
        minValue = std::min(minValue, map[i]);
        maxValue = std::max(maxValue, map[i]);
    }
    const double range = maxValue - minValue;

    for(size_t i = 0; i < map.size(); ++i)
    {
        map[i] = (map[i] - minValue) / range;
    }
}

void TerrainGenerator::computeVertices(const glm::vec2& offset, const float theta)
{
    const glm::vec3 off = glm::vec3(offset.x, offset.y, 0.0f);
    const float cosTheta = glm::cos(-theta);
    const float sinTheta = glm::sin(-theta);

    vertices_.resize(sizeX_ * sizeY_);
#pragma omp parallel for
    for(int y = 0; y < int(sizeY_); ++y)
    {
        for(int x = 0; x < int(sizeX_); ++x)
        {
            const float x0 = resolution_.x * float(x - int(sizeX_ / 2));
            const float y0 = resolution_.y * float(y - int(sizeY_ / 2));
            const float z = heightMap_[y * sizeX_ + x] / resolution_.z;
            const auto pose
                = glm::vec3(cosTheta * x0 + sinTheta * y0, -sinTheta * x0 + cosTheta * y0, z);
            vertices_[y * sizeX_ + x] = pose + off;
        }
    }
}

void TerrainGenerator::computeColors()
{
    colors_.resize(sizeX_ * sizeY_);

#pragma omp parallel for
    for(size_t y = 0; y < sizeY_; ++y)
    {
        for(size_t x = 0; x < sizeX_; ++x)
        {
            const auto& col = colorMap_[y * sizeX_ + x];
            colors_[y * sizeX_ + x] = glm::vec4(
                float(col.r) / 255.0f, float(col.g) / 255.0f, float(col.b) / 255.0f, 255.0f);
        }
    }
}

void TerrainGenerator::computeNormals()
{
    // TODO : not implemented yet
}

void TerrainGenerator::computeFaces()
{
    const size_t dimX = sizeX_ - 1;
    const size_t dimY = sizeY_ - 1;
    const size_t faceCount = 2 * dimX * dimY;
    faces_.resize(faceCount);

#pragma omp parallel for
    for(size_t i = 0; i < dimY; ++i)
    {
        for(size_t j = 0; j < dimX; ++j)
        {
            const size_t baseIndex = i * sizeX_ + j;
            const glm::ivec3 face0{baseIndex, baseIndex + sizeX_ + 1, baseIndex + sizeX_};
            const glm::ivec3 face1{baseIndex, baseIndex + 1, baseIndex + sizeX_ + 1};

            const size_t faceOffset = 2 * (i * dimX + j);
            faces_[faceOffset + 0] = face0;
            faces_[faceOffset + 1] = face1;
        }
    }
}

#ifdef DEBUG_GENERATION
void TerrainGenerator::debugMaps(const glm::vec2& offset, const float theta)
{
    std::vector<float> heightImgData(heightMap_.size());
    std::vector<float> moistImgData(moistureMap_.size());
    std::vector<ImgPixel> colorImgData(heightMap_.size());

    for(size_t i = 0; i < sizeY_; ++i)
    {
        for(size_t j = 0; j < sizeX_; ++j)
        {
            heightImgData[(sizeY_ - i - 1) * sizeX_ + j] = 255.0f * heightMap_[i * sizeX_ + j];
            moistImgData[(sizeY_ - i - 1) * sizeX_ + j] = 255.0f * moistureMap_[i * sizeX_ + j];
            colorImgData[(sizeY_ - i - 1) * sizeX_ + j] = colorMap_[i * sizeX_ + j];
        }
    }

    cv::Mat heightImg(sizeY_, sizeX_, CV_32FC1, heightImgData.data());
    cv::Mat moistImg(sizeY_, sizeX_, CV_32FC1, moistImgData.data());
    cv::Mat colorImg(sizeY_, sizeX_, CV_8UC3, colorImgData.data());

    char heightMapName[512];
    snprintf(heightMapName, 512, "height_map_%f_%f_%f.png", offset.x, offset.y, theta);

    char moistureMapName[512];
    snprintf(moistureMapName, 512, "moisture_map_%f_%f_%f.png", offset.x, offset.y, theta);

    char colorMapName[512];
    snprintf(colorMapName, 512, "color_map_%f_%f_%f.png", offset.x, offset.y, theta);

    cv::imwrite(heightMapName, heightImg);
    cv::imwrite(moistureMapName, moistImg);
    cv::imwrite(colorMapName, colorImg);
}

void TerrainGenerator::exportPLY(const char* filename, const glm::vec2& offset, const float theta)
{
    char plyName[512];
    snprintf(plyName, 512, "%s_%f_%f_%f.ply", filename, offset.x, offset.y, theta);

    fprintf(stdout, "[DEBUG]: Exporting %s...\n", plyName);
    saveSurface(plyName, vertices_, colors_, faces_);
    fprintf(stdout, "[DEBUG]: Exporting %s done.\n", plyName);
}
#endif
} // namespace cg

void saveSurface(
    const std::string& filename,
    const std::vector<glm::vec3>& xyz,
    const std::vector<glm::vec4>& rgb,
    const std::vector<glm::ivec3>& triangles)
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