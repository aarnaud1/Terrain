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

#ifndef BIOME_INL
#define BIOME_INL

#define OCEAN uvec3(70, 67, 122)
#define BEACH uvec3(162, 142, 117)
#define SUBTROPICAL_DESERT uvec3(211, 184, 139)
#define TEMPERATE_DESERT uvec3(201, 210, 155)
#define SHRUBLAND uvec3(136, 153, 119)
#define TAIGA uvec3(153, 170, 116)
#define SCORCHED uvec3(85, 85, 85)
#define BARE uvec3(136, 136, 136)
#define TUNDRA uvec3(186, 187, 169)
#define SNOW uvec3(221, 222, 227)
#define GRASSLAND uvec3(136, 170, 84)
#define TEMPERATE_DECIDUOUS_FOREST uvec3(103, 148, 89)
#define TEMPERATE_RAIN_FOREST uvec3(69, 135, 87)
#define TROPICAL_SEASONAL_FOREST uvec3(85, 154, 65)
#define TROPICAL_RAIN_FOREST uvec3(52, 120, 83)

uvec3 getColor(const float e, const float m)
{
    if(e < 0.1f)
    {
        return OCEAN;
    }
    if(e < 0.12f)
    {
        return BEACH;
    }

    if(e > 0.8f)
    {
        if(m < 0.1f)
        {
            return SCORCHED;
        }
        if(m < 0.2f)
        {
            return BARE;
        }
        if(m < 0.5f)
        {
            return TUNDRA;
        }
        return SNOW;
    }

    if(e > 0.6f)
    {
        if(m < 0.33f)
        {
            return TEMPERATE_DESERT;
        }
        if(m < 0.66f)
        {
            return SHRUBLAND;
        }
        return TAIGA;
    }

    if(e > 0.3f)
    {
        if(m < 0.16f)
        {
            return TEMPERATE_DESERT;
        }
        if(m < 0.50f)
        {
            return GRASSLAND;
        }
        if(m < 0.83f)
        {
            return TEMPERATE_DECIDUOUS_FOREST;
        }
        return TEMPERATE_RAIN_FOREST;
    }

    if(m < 0.16f)
    {
        return SUBTROPICAL_DESERT;
    }
    if(m < 0.33f)
    {
        return GRASSLAND;
    }
    if(m < 0.66f)
    {
        return TROPICAL_SEASONAL_FOREST;
    }
    return TROPICAL_RAIN_FOREST;
}
#endif