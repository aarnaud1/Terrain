 # Copyright (C) 2024 Adrien ARNAUD
 #
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 #
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program. If not, see <https://www.gnu.org/licenses/>.

CXX       := g++ -W -Wall -Wextra -std=c++17
CXX_FLAGS := -O3 -g --pedantic -ffast-math -ftree-vectorize -march=native -mavx2 -fopenmp
DEFINES   := -DGLM_FORCE_RADIANS
IFLAGS    := -I./include -I./Vulkan/include/ -I/usr/include/opencv4
LFLAGS    := -L./Vulkan/output/lib -lVkWrappers \
             -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -ltinyply

all: submodules main

main : main.cpp src/TerrainGenerator.cpp src/ValueNoiseGenerator.cpp
		$(CXX) $(DEFINES) $(CXX_FLAGS) -o $@ $(IFLAGS) $^ $(LFLAGS)

submodules:
	make -C Vulkan lib -j8

clean:
	rm -f main
	make -C Vulkan clean