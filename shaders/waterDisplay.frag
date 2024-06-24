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

layout(binding = 1) uniform sampler2D colorSampler;

// TODO : try to use spec constants
const float blurDist = 25.0f;
const vec3 horizonColor = vec3(0.259f, 0.557f, 0.914f);
const vec3 baseWaterColor = vec3(0.0f, 0.1f, 0.2);

float sigm(const float x, const float alpha) { return 1.0f / (1.0f + exp(-alpha * x)); }

layout(push_constant) uniform PushConstants
{
    float width;
    float height;
}
pcs;

void main()
{
    const float shininess = 50.0f;
    const vec3 L = normalize(vec3(1.0f, 0.0f, 1.0f));
    const vec3 N = normalize(vertexNormal);
    const vec3 R = reflect(L, N);
    const vec3 V = normalize(-vertexPos);
    float specAngle = max(dot(R, V), 0.0);
    const float specular = pow(specAngle, shininess);

    const vec3 reflectColor
        = texture(colorSampler, vec2(gl_FragCoord.x / pcs.width, gl_FragCoord.y / pcs.height)).rgb;
    const vec3 waterColor = 0.6f * baseWaterColor + 0.4f * reflectColor;
    const float lambertian = max(dot(N, L), 0.0f);
    const vec3 ambiantColor = 0.1f * waterColor.xyz;
    const vec3 diffuseColor = 0.9f * waterColor.xyz;
    const vec3 specularColor = 0.2f * waterColor.xyz;
    const vec3 color = ambiantColor + lambertian * diffuseColor + specular * specularColor;

    const float blurFact = sigm(vertexPos.z - blurDist, 1.0f);
    // fragColor = vec4(mix(color, horizonColor, blurFact), 0.8f);
    fragColor = vec4(mix(color, horizonColor, blurFact), 1.0f);
}