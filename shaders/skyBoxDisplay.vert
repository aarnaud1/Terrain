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

layout(location = 0) in vec3 position;

out gl_PerVertex
{
    vec4 gl_Position;
    float gl_ClipDistance[1];
};

layout(location = 0) out vec3 vertexPos;

layout(binding = 0) uniform Matrices
{
    mat4 view;
    mat4 proj;
    mat4 invView;
}
mvp;

layout(push_constant) uniform PushConstants
{
    mat4 model;
    vec4 clipPlane;
    vec4 offset;
}
pcs;

void main()
{
    const vec3 worldPos =position + pcs.offset.xyz;
    gl_ClipDistance[0] = dot(pcs.model * vec4(worldPos, 1.0f), pcs.clipPlane);
    gl_Position = mvp.proj * mvp.view * pcs.model * vec4(worldPos, 1.0f);
    vertexPos = position;
}