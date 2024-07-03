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
#include "lighting.inl"

layout(location = 0) in vec3 vertexPos;    // In world coordinates
layout(location = 1) in vec3 vertexNormal; // In world coordinates
layout(location = 2) in vec4 vertexColor;

layout(location = 0) out vec4 fragColor;

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
    vec4 lightPos;
    float farDist;
}
pcs;

void main()
{
    const float shininess = 10.0f;

    const vec3 org = mvp.invView[3].xyz;
    const vec3 lightPos = vec3(org.xy, 0.0f) + pcs.farDist * pcs.lightPos.xyz;
    const vec3 v = vec3(mvp.invView * vec4(vertexPos, 1.0f));

    const vec3 N = normalize(vec3(mvp.invView * vec4(vertexNormal, 0.0f)));
    const vec3 L = normalize(lightPos - v);
    const vec3 R = reflect(L, N);
    const vec3 V = normalize(org - v);
    // Little hack : put -dot to simulate some sun reflection
    const float alpha = max(-dot(V, R), 0.0f);
    const float specular = pow(alpha, shininess);
    const float lambertian = max(dot(N, L), 0.0f);

    const vec3 ambiantColor = 0.2f * vertexColor.xyz;
    const vec3 diffuseColor = 0.8f * vertexColor.xyz;
    const vec3 specularColor = 0.1f * vertexColor.xyz;
    const vec3 color = ambiantColor + lambertian * diffuseColor + specular * specularColor;

    const float blurFact = sigm(length(org - v) - pcs.farDist, 1.0f);
    fragColor = vec4(mix(color, horizonColor, blurFact), 1.0f);
}