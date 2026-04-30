# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import PhysicsNeMo library
from sympy import Symbol
import numpy as np
from stl import mesh as np_mesh
from physicsnemo.sym.geometry.primitives_3d import Box, Channel, Plane
from physicsnemo.sym.geometry import Parameterization, Parameter
from physicsnemo.sym.geometry.tessellation import Tessellation


class LimeRock(object):
    def __init__(self):
        scale_value = 100
        self.scale = 1 / scale_value
        self.translate = (80, 5, 160)  # mm, shifts STL_y so inlet starts at 0

        # make solids
        self.copper = None

        # parse file
        print("parsing stl file...")
        self._parse_file("../stl_files/Placement_m_ascii.stl")
        print("finished parsing")

        # inlet area
        self.inlet_area = (325*self.scale - 245*self.scale) * (
            self.geo_bounds_upper[1] - self.geo_bounds_lower[1]
        )

        # geo
        self.heat_sink_bounds = (15*self.scale, 115*self.scale)  
        self.geo = self.channel-self.copper
        self.geo_solid = self.copper

        self.geo_bounds = {
            Symbol("x"): (self.geo_bounds_lower[0], self.geo_bounds_upper[0]),
            Symbol("y"): (self.geo_bounds_lower[1], self.geo_bounds_upper[1]),
            Symbol("z"): (self.geo_bounds_lower[2], self.geo_bounds_upper[2]),
        }
        self.geo_hr_bounds = {
            Symbol("x"): self.heat_sink_bounds,
            Symbol("y"): (self.geo_bounds_lower[1], self.geo_bounds_upper[1]),
            Symbol("z"): (self.geo_bounds_lower[2], self.geo_bounds_upper[2]),
        }

        # integral plane
        x_pos = Parameter("x_pos")
        self.integral_plane = Plane(
            (x_pos, self.geo_bounds_lower[1], self.geo_bounds_lower[2]),
            (x_pos, self.geo_bounds_upper[1], self.geo_bounds_upper[2]),
            1,
            parameterization=Parameterization({x_pos: self.heat_sink_bounds}),
        )

    def solid_names(self):
        return list(self.solids.keys())

    def _parse_file(self, filename):
        # Read file
        reader = open(filename)
        solid_bounds_list = []
        while True:
            line = reader.readline()
            if "solid" == line.split(" ")[0]:
                bounds_lower, bounds_upper = self.read_solid(reader)
                solid_bounds_list.append((bounds_lower, bounds_upper))
            else:
                break

        # compute solid bounds from STL
        solid_lower = (
            min(b[0][0] for b in solid_bounds_list),
            min(b[0][1] for b in solid_bounds_list),
            min(b[0][2] for b in solid_bounds_list),
        )
        solid_upper = (
            max(b[1][0] for b in solid_bounds_list),
            max(b[1][1] for b in solid_bounds_list),
            max(b[1][2] for b in solid_bounds_list),
        )

        # STL is hollow: its bounding box IS the fluid domain (channel)
        self.geo_bounds_lower = solid_lower
        self.geo_bounds_upper = solid_upper

        # build copper Tessellation from STL with axis remap + scale + translate
        mesh = np_mesh.Mesh.from_file(filename)
        orig = mesh.vectors.copy()
        # sim_x = STL_y, sim_y = STL_z, sim_z = STL_x
        mesh.vectors[:, :, 0] = (orig[:, :, 1] + self.translate[0]) * self.scale
        mesh.vectors[:, :, 1] = (orig[:, :, 2] + self.translate[1]) * self.scale
        mesh.vectors[:, :, 2] = (orig[:, :, 0] + self.translate[2]) * self.scale
        self.copper = Tessellation(mesh, airtight=True)

        # actual bottom y of copper tessellation (may differ from channel geo_bounds_lower[1])
        self.copper_base_y = float(mesh.vectors[:, :, 1].min())

        # inlet at x_min face, outlet at x_max face, spanning full channel height
        # self.inlet = Plane(
        #     (self.geo_bounds_lower[0], 245*self.scale, self.geo_bounds_lower[2]),
        #     (self.geo_bounds_lower[0], 325*self.scale, self.geo_bounds_upper[2]),
        #     -1,
        # )
        # self.outlet = Plane(
        #     (self.geo_bounds_upper[0], 15*self.scale, self.geo_bounds_lower[2]),
        #     (self.geo_bounds_upper[0], 115*self.scale, self.geo_bounds_upper[2]),
        #     1,
        # )
        self.inlet = Plane(
            (self.geo_bounds_lower[0], self.geo_bounds_lower[1], 245*self.scale),
            (self.geo_bounds_lower[0], self.geo_bounds_upper[1], 325*self.scale),
            -1,
        )
        self.outlet = Plane(
            (self.geo_bounds_upper[0], self.geo_bounds_lower[1], 15*self.scale),
            (self.geo_bounds_upper[0], self.geo_bounds_upper[1], 115*self.scale),
            1,
        )
        self.channel = Channel(self.geo_bounds_lower, self.geo_bounds_upper)


    def read_solid(self, reader):
        # solid pieces
        faces = []
        while True:
            line = reader.readline()
            split_line = line.split()  # split() handles leading whitespace and newlines
            if len(split_line) == 0:   # EOF returns "", split() gives [] → break
                break
            elif "endsolid" == split_line[0]:
                break
            elif "facet" == split_line[0]:
                # read outer loop line
                _ = reader.readline()
                # read 3 vertices
                a_0 = [float(x) for x in reader.readline().split()[-3:]]
                a_1 = [float(x) for x in reader.readline().split()[-3:]]
                a_2 = [float(x) for x in reader.readline().split()[-3:]]
                faces.append([a_0, a_1, a_2])
                # read end loop/end facet
                _ = reader.readline()
                _ = reader.readline()
        faces = np.array(faces)
        bounds_lower = (
            np.min(faces[..., 1]),  # sim_x = STL_y (flow direction)
            np.min(faces[..., 2]),  # sim_y = STL_z
            np.min(faces[..., 0]),  # sim_z = STL_x
        )
        bounds_upper = (
            np.max(faces[..., 1]),  # sim_x = STL_y (flow direction)
            np.max(faces[..., 2]),  # sim_y = STL_z
            np.max(faces[..., 0]),  # sim_z = STL_x
        )
        bounds_lower = tuple(
            [self.scale * (x + t) for x, t in zip(bounds_lower, self.translate)]
        )
        bounds_upper = tuple(
            [self.scale * (x + t) for x, t in zip(bounds_upper, self.translate)]
        )
        return bounds_lower, bounds_upper


# def _center(bounds_lower, bounds_upper):
#     center_x = bounds_lower[0] + (bounds_upper[0] - bounds_lower[0]) / 2
#     center_y = bounds_lower[1] + (bounds_upper[1] - bounds_lower[1]) / 2
#     center_z = bounds_lower[2] + (bounds_upper[2] - bounds_lower[2]) / 2
#     return center_x, center_y, center_z
