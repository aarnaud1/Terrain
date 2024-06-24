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
DEFINES   := -DGLM_FORCE_RADIANS -DGLM_FORCE_DEPTH_ZERO_TO_ONE
IFLAGS    := -I./include -I./Vulkan/include/ -I/usr/include/opencv4
LFLAGS    := -L./Vulkan/output/lib -Wl,-rpath,./Vulkan/output/lib -lVkWrappers \
             -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc \
			 -ltinyply -lglfw -lvulkan

SHADERS_SPV := $(patsubst shaders/%.comp,output/spv/%_comp.spv,$(wildcard shaders/*.comp)) \
               $(patsubst shaders/%.vert,output/spv/%_vert.spv,$(wildcard shaders/*.vert)) \
			   $(patsubst shaders/%.frag,output/spv/%_frag.spv,$(wildcard shaders/*.frag))

SRC_FILES := main.cpp \
             src/TerrainGenerator.cpp \
             src/ValueNoiseGenerator.cpp \
	         src/TerrainEngine.cpp \
	         src/TerrainGeneratorGPU.cpp

all: dir $(SHADERS_SPV) submodules main

main : $(SRC_FILES)
		$(CXX) $(DEFINES) $(CXX_FLAGS) -o $@ $(IFLAGS) $^ $(LFLAGS)

output/spv/%_comp.spv: shaders/%.comp
	glslc -std=450core -fshader-stage=compute -o $@ $^
output/spv/%_vert.spv: shaders/%.vert
	glslc -std=450core -fshader-stage=vertex -o $@ $^
output/spv/%_frag.spv: shaders/%.frag
	glslc -std=450core -fshader-stage=fragment -o $@ $^

submodules:
	make -C Vulkan lib -j8

dir:
	mkdir -p output/spv

clean:
	rm -f main
	rm -rfd output
	make -C Vulkan clean