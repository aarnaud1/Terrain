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

#ifndef NOISE_INL
#define NOISE_INL

#define MAX_PRIME_INDEX 10

#define INTERPOLATION_TYPE_LINEAR 0
#define INTERPOLATION_TYPE_COSINE 1

int primes[MAX_PRIME_INDEX][3]
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
float interpolate(float a, float b, float x)
{
    const float ft = x * 3.1415927f;
    const float f = (1.0f - cos(ft)) * 0.5f;
    return a * (1 - f) + b * f;
}
#else
float interpolate(const float a, const float b, const float alpha)
{
    return a * (1.0f - alpha) + b * alpha;
}
#endif

float noise(const int x, const int y, const uint i)
{
    int n = x + y * 512;
    n = (n << 13) ^ n;
    return (
        1.0f
        - ((n * (n * n * primes[i][0] + primes[i][1]) + primes[i][2]) & 0x7fffffff)
              / 1073741824.0f);
}

float smoothNoise(const int x, const int y, const uint i)
{
    return 0.0625f
               * (noise(x - 1, y - 1, i) + noise(x + 1, y - 1, i) + noise(x + 1, y + 1, i)
                  + noise(x - 1, y + 1, i))
           + 0.125f
                 * (noise(x + 1, y, i) + noise(x - 1, y, i) + noise(x, y - 1, i)
                    + noise(x, y + 1, i))
           + 0.25f * noise(x, y, i);
}

float interpolateNoise(const float x, const float y, const uint i)
{
    const int intX = int(floor(x));
    const float fracX = x - intX;

    const int intY = int(floor(y));
    const float fracY = y - intY;

    const float c00 = smoothNoise(intX, intY, i);
    const float c10 = smoothNoise(intX + 1, intY, i);
    const float c01 = smoothNoise(intX, intY + 1, i);
    const float c11 = smoothNoise(intX + 1, intY + 1, i);

    const float c0 = interpolate(c00, c10, fracX);
    const float c1 = interpolate(c01, c11, fracX);

    return interpolate(c0, c1, fracY);
}
#endif // NOISE_INL