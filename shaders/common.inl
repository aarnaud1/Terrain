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

vec3 getNormal(const vec3 p, const vec3 px0, const vec3 px1, const vec3 py0, const vec3 py1)
{
    const vec3 n0 = cross(px1, py1);
    const vec3 n1 = cross(py1, px0);
    const vec3 n2 = cross(px0, py0);
    const vec3 n3 = cross(py0, px1);

    return normalize(n0 + n1 + n2 + n3);
}