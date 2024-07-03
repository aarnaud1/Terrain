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

#ifndef LIGHTING_INL
#define LIGHTING_INL

// const float blurDist = 35.0f;
const vec3 horizonColor = vec3(0.259f, 0.557f, 0.914f);

float sigm(const float x, const float alpha) { return 1.0f / (1.0f + exp(-alpha * x)); }

#endif // LIGHTING_INL