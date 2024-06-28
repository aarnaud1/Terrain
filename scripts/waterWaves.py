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

#!/usr/bin/python3

import argparse
import sys 
import os 

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QSlider, QWidget)

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

maxOctaves = 4
gridSizeX = 1024
gridSizeY = 1024

x0 = np.linspace(-10.0, 10.0, num=gridSizeX)
y0 = np.linspace(-10.0, 10.0, num=gridSizeY)
x, y = np.meshgrid(x0, y0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

EARTH_GRAVITY = 9.81
L = [1.0, 1.0, 1.0, 1.0]
S = [1.0, 1.0, 1.0, 1.0]
A = [1.0, 1.0, 1.0, 1.0]
Q = [0.0, 1.0, 1.0, 1.0]
alpha = [0.0, 0.0, 0.0, 0.0]

def wave(x, y):
    i = 0

    retX = x
    retY = y
    retZ = np.zeros_like(x)

    alphai = alpha[i]
    Li = L[i]
    Si = S[i]
    Ai = A[i]
    Qi = Q[i]

    omegai = np.sqrt(2.0 * np.pi * EARTH_GRAVITY / Li)
    phii = 2.0 * Si / Li

    Dx = np.cos(alphai)
    Dy = np.sin(alphai)
    dp = Dx * x + Dy * y

    thetai = omegai * dp + phii
    cosTheta = Qi * Ai * np.cos(thetai)
    sinTheta = Ai * np.sin(thetai)
    retX = retX + Dx * cosTheta
    retY = retY + Dy * cosTheta
    retZ +=  retZ + sinTheta

    return retX, retY, retZ

X, Y, Z = wave(x, y)
ax.set_zlim([-10.0, 10.0])
ax.plot_surface(X, Y, Z)

def updateL0(val):
    L[0] = val
    X, Y, Z = wave(x, y)
    ax.cla()
    ax.set_zlim([-10.0, 10.0])
    ax.plot_surface(X, Y, Z)
def updateQ0(val):
    Q[0] = val
    X, Y, Z = wave(x, y)
    ax.cla()
    ax.set_zlim([-10.0, 10.0])
    ax.plot_surface(X, Y, Z)

class Window(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
    
    Window(QWidget perent = none)

    

if __name__ == '__main__':
    sL0 = Slider(
        ax=plt.axes([0.2, 0.05, 0.65, 0.03]),
        label='L0',
        valmin=0.0,
        valmax=10.0,
        valinit=1.0,
        valstep=0.01
    )
    sL0.on_changed(updateL0)
    sQ0 = Slider(
        ax=plt.axes([0.2, 0.1, 0.65, 0.03]),
        label='Q0',
        valmin=0.0,
        valmax=2.0,
        valinit=0.0,
        valstep=0.001
    )
    sQ0.on_changed(updateQ0)

    plt.show()
    