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

#include <cstdlib>
#include <glm/glm.hpp>
#include <vector>

namespace cg
{
namespace value_noise
{
    void genRandom2DArray(
        double* __restrict__ values,
        const size_t w,
        const size_t h,
        const double offX,
        const double offY,
        const double theta,
        const double baseWaveLength,
        const double randomSeed = 0,
        const size_t octaves = 10);

    void genRandom2DArray(
        const glm::vec2* coords,
        double* values,
        const size_t count,
        const double baseWaveLength,
        const double randomSeed = 0,
        const size_t octaves = 10);
} // namespace value_noise

class ValueNoiseGenerator
{
  public:
    ValueNoiseGenerator() = default;
    ValueNoiseGenerator(const double randomSeed) : randomSeed_{randomSeed} {}

    ~ValueNoiseGenerator() = default;

    inline void setOctaveCount(const size_t octaves) { octaves_ = octaves; }
    inline void setSeed(const double randomSeed) { randomSeed_ = randomSeed; }

    std::vector<double> gen2DArray(
        const size_t w,
        const size_t h,
        const double baseWaveLength,
        const double offX = 0.0,
        const double offY = 0.0,
        const double theta = 0.0);
    void gen2DArray(
        double* data,
        const size_t w,
        const size_t h,
        const double baseWaveLength,
        const double offX = 0.0,
        const double offY = 0.0,
        const double theta = 0.0);

  private:
    size_t octaves_{10};
    double randomSeed_{0.0};
};
} // namespace cg