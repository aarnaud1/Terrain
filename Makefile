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
IFLAGS    := -I./include -I/usr/include/opencv4
LFLAGS    := -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -ltinyply

all: main

main : main.cpp src/TerrainGenerator.cpp src/ValueNoiseGenerator.cpp
		$(CXX) $(DEFINES) $(CXX_FLAGS) -o $@ $(IFLAGS) $^ $(LFLAGS)

clean:
	rm -f main