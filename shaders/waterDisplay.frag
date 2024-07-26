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

layout(location = 0) in vec3 vertexPos;    // In camera coordinates
layout(location = 1) in vec3 vertexNormal; // In camera coordinates

layout(location = 0) out vec4 fragColor;

layout(binding = 0) uniform Matrices
{
    mat4 view;
    mat4 proj;
    mat4 invView;
}
mvp;
layout(binding = 1) uniform sampler2D reflectionSampler;
layout(binding = 2) uniform sampler2D refractionSampler;

layout(push_constant) uniform PushConstants
{
    float width;
    float height;
    float farDist;
    vec4 lightPos;
}
pcs;

void main()
{
    const float shininess = 100.0f;

    const vec3 reflectColor
        = texture(reflectionSampler, vec2(gl_FragCoord.x / pcs.width, gl_FragCoord.y / pcs.height))
              .rgb;
    const vec3 refractColor
        = texture(refractionSampler, vec2(gl_FragCoord.x / pcs.width, gl_FragCoord.y / pcs.height))
              .rgb;

    const vec3 org = vec3(mvp.invView[3].xyz);
    const vec3 lightPos = vec3(org.xy, 0.0f) + pcs.farDist * pcs.lightPos.xyz;
    const vec3 v = vec3(mvp.invView * vec4(vertexPos, 1.0f));

    const vec3 N = normalize(vec3(mvp.invView * vec4(vertexNormal, 0.0f)));
    const vec3 L = normalize(lightPos - v);
    const vec3 R = reflect(L, N);
    const vec3 V = normalize(org - v);
    // Little hack : put -dot to simulate some sun reflection
    const float alpha = max(-dot(V, R), 0.0f);
    const float specular = pow(alpha, shininess);

    const float fact = clamp(2.0f * dot(V, N), 0.0f, 1.0f);
    const vec3 baseColor = mix(reflectColor, refractColor, fact);
    const vec3 specularColor = vec3(1.0f, 1.0f, 1.0f);
    const vec3 color = baseColor + specular * specularColor;

    const float blurFact = sigm(length(org - v) - pcs.farDist, 1.0f);
    fragColor = vec4(mix(color, horizonColor, blurFact), 1.0f);
}