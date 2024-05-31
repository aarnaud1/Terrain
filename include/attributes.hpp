/*
 * Copyright (C) 2024  Adrien ARNAUD
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

#pragma once

#ifdef __NVCC__
#    define ATTR_INL __forceinline__
#    define ATTR_DEV __device__
#    define ATTR_HOST __host__
#    define ATTR_HOST_DEV __host__ __device__
#    define ATTR_HOST_INL __host__ __forceinline__
#    define ATTR_DEV_INL __device__ __forceinline__
#    define ATTR_HOST_DEV_INL __host__ __device__ __forceinline__
#else
#    define ATTR_INL inline
#    define ATTR_DEV
#    define ATTR_HOST
#    define ATTR_HOST_DEV
#    define ATTR_HOST_INL inline
#    define ATTR_DEV_INL inline
#    define ATTR_HOST_DEV_INL inline
#endif