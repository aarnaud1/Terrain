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

#include "ValueNoiseGenerator.hpp"

#include <bits/stdc++.h>
#include <omp.h>

#define MAX_PRIME_INDEX 10

#define INTERPOLATION_TYPE_LINEAR 0
#define INTERPOLATION_TYPE_COSINE 1

// Inspired by :
// https://web.archive.org/web/20160530124230/http://freespace.virgin.net/hugo.elias/models/m_perlin.htm

static int primes[MAX_PRIME_INDEX][3]
    = {{995615039, 600173719, 701464987},
       {831731269, 162318869, 136250887},
       {174329291, 946737083, 245679977},
       {362489573, 795918041, 350777237},
       {457025711, 880830799, 909678923},
       {787070341, 177340217, 593320781},
       {405493717, 291031019, 391950901},
       {458904767, 676625681, 424452397},
       {531736441, 939683957, 810651871},
       {997169939, 842027887, 423882827}};

#define INTERPOLATION_TYPE INTERPOLATION_TYPE_COSINE
#if(defined INTERPOLATION_TYPE && INTERPOLATION_TYPE == INTERPOLATION_TYPE_COSINE)
static inline double interpolate(double a, double b, double x)
{
    const double ft = x * 3.1415927;
    const double f = (1 - cos(ft)) * 0.5;
    return a * (1 - f) + b * f;
}
#else
static inline double interpolate(const double a, const double b, const double alpha)
{
    return a * (1.0f - alpha) + b * alpha;
}
#endif

static inline double noise(const int x, const int y, const size_t i)
{
    // int n = x + y * 57;
    int n = x + y * 512;
    n = (n << 13) ^ n;
    return (
        1.0f
        - ((n * (n * n * primes[i][0] + primes[i][1]) + primes[i][2]) & 0x7fffffff)
              / 1073741824.0f);
}

static inline double smoothNoise(const int x, const int y, const size_t i)
{
    return 0.0625f
               * (noise(x - 1, y - 1, i) + noise(x + 1, y - 1, i) + noise(x + 1, y + 1, i)
                  + noise(x - 1, y + 1, i))
           + 0.125f
                 * (noise(x + 1, y, i) + noise(x - 1, y, i) + noise(x, y - 1, i)
                    + noise(x, y + 1, i))
           + 0.25f * noise(x, y, i);
}

static inline double interpolateNoise(const double x, const double y, const size_t i)
{
    const int intX = floor(x);
    const double fracX = x - intX;

    const int intY = floor(y);
    const double fracY = y - intY;

    const double c00 = smoothNoise(intX, intY, i);
    const double c10 = smoothNoise(intX + 1, intY, i);
    const double c01 = smoothNoise(intX, intY + 1, i);
    const double c11 = smoothNoise(intX + 1, intY + 1, i);

    const double c0 = interpolate(c00, c10, fracX);
    const double c1 = interpolate(c01, c11, fracX);

    return interpolate(c0, c1, fracY);
}

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
        const double randomSeed,
        const size_t octaves)
    {
        const double cosTheta = glm::cos(-theta);
        const double sinTheta = glm::sin(-theta);

        memset(values, 0, w * h * sizeof(double));

        for(int i = 0; i < int(h); ++i)
        {
            double freq = 1.0f / baseWaveLength;
            double amp = 1.0f;

            // const double y = double(i) + offY + randomSeed;
            for(size_t octave = 0; octave < octaves; ++octave)
            {
                for(int j = 0; j < int(w); ++j)
                {
                    const double x = double(j - int(w / 2)); //  + offX;
                    const double y = double(i - int(h / 2)); //  + offY;

                    const double X = cosTheta * x + sinTheta * y + offX;
                    const double Y = -sinTheta * x + cosTheta * y + offY + randomSeed;

                    const double tmp
                        = interpolateNoise(X * freq, Y * freq, octave % MAX_PRIME_INDEX);
                    values[i * w + j] += amp * tmp;
                }
                freq *= 2.0f;
                amp *= 0.5f;
            }

            for(size_t j = 0; j < w; ++j)
            {
                values[i * w + j] += 0.5;
            }
        }
    }

    void genRandom2DArray(
        const glm::vec2* coords,
        double* values,
        const size_t count,
        const double baseWaveLength,
        const double randomSeed,
        const size_t octaves)
    {
        memset(values, 0, count * sizeof(double));
        for(size_t octave = 0; octave < octaves; ++octave)
        {
            double freq = 1.0f / baseWaveLength;
            double amp = 1.0f;

#pragma omp parallel for
            for(size_t i = 0; i < count; ++i)
            {
                const double x = coords[i].x;
                const double y = coords[i].y + randomSeed;
                const double tmp = interpolateNoise(x * freq, y * freq, octave % MAX_PRIME_INDEX);
                values[i] += amp * tmp;
            }
            freq *= 2.0f;
            amp *= 0.5f;
        }

#pragma omp parallel for
        for(size_t i = 0; i < count; ++i)
        {
            values[i] += 0.5;
        }
    }
} // namespace value_noise

std::vector<double> ValueNoiseGenerator::gen2DArray(
    const size_t w,
    const size_t h,
    const double baseWaveLength,
    const double offX,
    const double offY,
    const double theta)
{
    std::vector<double> ret;
    ret.resize(w * h);
    value_noise::genRandom2DArray(
        ret.data(), w, h, offX, offY, theta, baseWaveLength, randomSeed_, octaves_);
    return ret;
}

void ValueNoiseGenerator::gen2DArray(
    double* data,
    const size_t w,
    const size_t h,
    const double baseWaveLength,
    const double offX,
    const double offY,
    const double theta)
{
    value_noise::genRandom2DArray(
        data, w, h, offX, offY, theta, baseWaveLength, randomSeed_, octaves_);
}
} // namespace cg