import math
import os
import struct
import sys

import glm
import moderngl
import pygame
from objloader import Obj
from PIL import Image
import numpy as np
import math
import time

import PIL

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((2560, 1440), flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN, vsync=True)


def init_includes():
    
    ctx = moderngl.get_context()

    ctx.includes['uniform_buffer_a'] = '''
        layout (std430, binding = 0) buffer Points_a {
            Point points[2560 * 1440];
        } Data_in;
    '''
    
    ctx.includes['uniform_buffer_b'] = '''
        struct Point {
            vec4 position;
            vec4 prev_position;
        };

        layout (std430, binding = 1) buffer Points_b {
            Point points[2560 * 1440];
        } Data_out;
    '''
    
class Compute:
    def __init__(self, batch_size):
        self.ctx = moderngl.get_context()
        self.program = self.ctx.compute_shader(
            '''
                #version 430
                #include "uniform_buffer_b"
                #include "uniform_buffer_a"
                
                layout (local_size_x = 2, local_size_y = 2) in;
                uniform float C2;
                uniform int width;
                uniform int height;

                void main() {                 
                    uint id = uint(min(gl_GlobalInvocationID.x + 1,width - 1) + int(min(gl_GlobalInvocationID.y + 1,height - 1)) * width);
                    float pos = Data_in.points[id].position.z;
                    float prevPos = Data_in.points[id].prev_position.z;
                    float posLeft = Data_in.points[id - 1].position.z;
                    float posRight = Data_in.points[id + 1].position.z;
                    float posTop = Data_in.points[id - width].position.z;
                    float posBottom = Data_in.points[id + width].position.z;
                    
                    float u = 2*pos - prevPos + C2*(posRight + posLeft + posTop + posBottom - 4*pos);
                    
                    Data_out.points[id].prev_position = Data_in.points[id].position;
                    Data_out.points[id].position = Data_in.points[id].position;
                    Data_out.points[id].position.z = u;
                    
                    // Data.points[gl_GlobalInvocationID.x + 1 + int(gl_GlobalInvocationID.y + 1) * width].prev_position = vec4(0,0,int(gl_GlobalInvocationID.x + 1)/width,0);
                }
            '''.replace("%COMPUTE_SIZE%", str(batch_size))
        )
    
    def run(self, screen_dimensions, courant):
        self.program['C2'] = courant**2
        self.program['width'] = screen_dimensions[0]
        self.program['height'] = screen_dimensions[1]
        self.program.run(screen_dimensions[0] - 2, screen_dimensions[1] - 2)
    
class Visualisation:
    def __init__(self):
        self.ctx = moderngl.get_context()
        self.program = self.ctx.program(
            vertex_shader='''
                #version 330 core

                vec2 positions[3] = vec2[](
                    vec2(-3.0, -1.0 ),
                    vec2(3.0, -1.0),
                    vec2(0.0, 2.0)
                );

                void main() {
                    gl_Position = vec4(positions[gl_VertexID], 1.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 430 core
                
                #include "uniform_buffer_b"
                
                uniform int width;
                // uniform int height;
                out vec4 out_color;

                void main() {
                    uint id = uint( gl_FragCoord.x + width * int(gl_FragCoord.y));
                    float pos = Data_out.points[id].position.z;
                    // vec3 col = Data.points[id].position.xyz / vec3(width, height, 1);
                    
                    float null = 0;
                    // if (pos == 0) {
                    //     null = 1;
                    // }
                    
                    out_color = vec4(null,-pos,pos,0);
                }
            ''',
        )
        self.vao = self.ctx.vertex_array(self.program, [])
        self.vao.vertices = 3

    def render(self,dimensions):
        self.ctx.enable_only(self.ctx.NOTHING)
        self.program['width'] = dimensions[0]
        # self.program['height'] = dimensions[1]
        self.vao.render()
        
class UniformBuffer:
    def __init__(self,screen_dimensions, STRUCT_SIZE=32):
        self.ctx = moderngl.get_context()
        self.data = bytearray(math.ceil(screen_dimensions[0] * screen_dimensions[1] * STRUCT_SIZE))
        self.ubo = self.ctx.buffer(self.data)
    
    def set_point(self, point_index, prev_position, position):
        offset = point_index * 32
        # add padding to make it 16 bytes, because of fucking reasons???
        self.data[offset + 0 : offset + 16] = struct.pack('4f', *position, 0.0)
        self.data[offset + 16 : offset + 32] = struct.pack('4f', *prev_position, 0.0)

    def use(self):
        self.ubo.write(self.data)
        self.ubo.bind_to_storage_buffer(0)
        
    def push(self):
        self.ubo.write(self.data)
        
    def bind(self, binding_point):
        self.ubo.bind_to_storage_buffer(binding_point)
        
    def update(self, point_index, prev_position, position):
        self.set_point(point_index, prev_position, position)
        offset = point_index * 32
        self.ubo.write(offset, self.data[offset : offset + 32])


def init_state(x,y,dimensions):
    theta = np.sqrt((x - 500)**2 + (y - 600)**2) * 1/32
    b1 = -math.pi
    b2 = math.pi
    g = 3.5
    if theta > b1 and theta < b2:
        return ((pow(g, math.sin(theta + math.pi/2)) - pow(g, -1)) / (g - 1/g)) * 1
    else:
       return 0
        
    # centrered circle with 1/r fallof
    # return 1/r if r < 300 else 0

class Scene:
    def __init__(self):
        print("initialising state, this could take a while...")
        self.ctx = moderngl.get_context()
        
        self.screen_width, self.screen_height = pygame.display.get_surface().get_size()
        self.STRUCT_SIZE = 4 * 4 + 4 * 4
        self.count = self.screen_width * self.screen_height
        self.inner_count = (self.screen_width - 2) * (self.screen_height - 2)
        
        self.courant = WAVE_VEL * DELTA_T * 1
        
        print(f"screen size: {self.screen_width}x{self.screen_height}")
        print(f"count: {self.count}")
        print(f"courant: {self.courant}")

        self.uniform_buffer_a = UniformBuffer((self.screen_width, self.screen_height), self.STRUCT_SIZE)
        self.uniform_buffer_b = UniformBuffer((self.screen_width, self.screen_height), self.STRUCT_SIZE)
        self.computation = Compute(batch_size=self.STRUCT_SIZE)

        self.visualisation = Visualisation()
        
        x = np.linspace(-self.screen_width/2,self.screen_width/2 - 1,self.screen_width)
        y = np.linspace(-self.screen_height/2,self.screen_height/2 - 1,self.screen_height)
        z = np.ndarray((self.screen_width, self.screen_height))
        # begin clock
        tic = time.perf_counter()
        # set initial state
        for ix,iy in np.ndindex(z.shape):
            # print(f"({ix},{iy})")
            z[ix,iy] = init_state(x[ix],y[iy],(self.screen_width,self.screen_height))
            
        # compute first time step
        for iy in range(1, self.screen_height - 1):
            for ix in range(1, self.screen_width - 1):
                p_z =  z[ix, iy]
                p_z_top = z[ix, iy - 1]
                p_z_bottom = z[ix, iy + 1]
                p_z_left = z[ix - 1, iy]
                p_z_right = z[ix + 1, iy]
                
                u = p_z + 1/2 * self.courant**2 * (p_z_top + p_z_bottom + p_z_left + p_z_right - 4 * p_z)
                
                position = (x[ix], y[iy],u)
                
                self.uniform_buffer_a.set_point(ix + iy * self.screen_width, (x[ix], y[iy],z[ix, iy]), position)
                
                # z[ix, iy] = u
            
        # insert boundary conditions
        
        for i in range(self.screen_width):
            z[i, 0] = 0
            self.uniform_buffer_a.set_point(i, (x[i], y[0],z[i, 0]), (x[i], y[0],0))
            z[i, self.screen_height - 1] = 0
            self.uniform_buffer_a.set_point(i + (self.screen_height - 1) * self.screen_width, (x[i], y[self.screen_height - 1],z[i, self.screen_height - 1]), (x[i], y[self.screen_height - 1],0))
        
        for i in range(self.screen_height):
            z[0, i] = 0
            self.uniform_buffer_a.set_point(i * self.screen_width, (x[0], y[i],z[0, i]), (x[0], y[i],0))
            z[self.screen_width - 1, i] = 0
            self.uniform_buffer_a.set_point(i * self.screen_width + (self.screen_width - 1), (x[self.screen_width - 1], y[i],z[self.screen_width - 1, i]), (x[self.screen_width - 1], y[i],0))
            
        self.uniform_buffer_a.push()
        self.uniform_buffer_b.push()
            
        toc = time.perf_counter()
        print(f"Elapsed time: {toc - tic:0.4f} seconds")
        

    def render(self,t,zoom):
        self.uniform_buffer_a.bind(0)
        self.uniform_buffer_b.bind(1)

        # Swap the buffers and vertex arrays around for next frame
        # enabling will absolutely fry performance, hard cpu bottleneck
        # self.uniform_buffer.use()
        
        # compute internal points
        self.computation.run((self.screen_width, self.screen_height), self.courant)
        self.visualisation.render((self.screen_width, self.screen_height))
        
        self.uniform_buffer_a, self.uniform_buffer_b = self.uniform_buffer_b, self.uniform_buffer_a

init_includes()

DELTA_T = 0.1
WAVE_VEL = 1
global_t = 0

zoom = 1

TELEMETRY = False

scene = Scene()

clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.MOUSEWHEEL:
            if event.y > 0:
                zoom *= 1.1
            if event.y < 0:
                zoom /= 1.1
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    clock.tick()
    if TELEMETRY:
        print(clock.get_fps())
      
    
    scene.render(t = global_t,zoom = zoom)
    global_t += DELTA_T

    pygame.display.flip()
    # clear screen
    
# TODO: switch buffers on every frame to prevent interference and multi-write