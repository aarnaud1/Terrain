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

#version 450 core

layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec3 vertexNormal;

layout(location = 0) out vec4 fragColor;

layout(binding = 1) uniform Matrices
{
    mat4 model;
    mat4 view;
    mat4 proj;
}
mvp;

// TODO : try to use spec constants
const float blurDist = 25.0f;
const vec3 horizonColor = vec3(0.259f, 0.557f, 0.914f);

float sigm(const float x, const float alpha) { return 1.0f / (1.0f + exp(-alpha * x)); }

void main()
{
    const vec3 color = vec3(0.004f, 0.137f, 0.271f);
    const float blurFact = sigm(vertexPos.z - blurDist, 1.0f);
    fragColor = vec4(mix(color, horizonColor, blurFact), 0.8f);
}