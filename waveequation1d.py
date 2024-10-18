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

import PIL

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((2560, 1440), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

screen_resolution_multiplier = 1


def init_includes():
    
    ctx = moderngl.get_context()

    ctx.includes['uniform_buffer'] = '''
        struct Point {
            vec4 position;
            vec4 prev_position;
        };

        layout (std430, binding = 0) buffer Points {
            Point points[2560];
        } Data;
    '''

    ctx.includes['srgb'] = '''
        vec3 srgb_to_linear(vec3 color) {
            return pow(color, vec3(2.2));
        }
        vec3 linear_to_srgb(vec3 color) {
            return pow(color, vec3(1.0 / 2.2));
        }
    '''

                    # vec2(-1.0, -1.0),
                    # vec2(3.0, -1.0),
                    # vec2(-1.0, 3.0)
                
class Compute:
    def __init__(self):
        self.ctx = moderngl.get_context()
        self.compute_program = self.ctx.compute_shader(
            compute_worker_shader_code='''
            
                layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
                
                // Shared memory
                // layout (shared) buffer SharedMemory {
                //     float points[2560];
                // } sharedMemory;
                
                // uniform float courant;
                uniform float c2;
                
                // uint index = uint(gl_GlobalInvocationID.x);
                // sharedMemory.points[index] = points[index].position.y;
                // barrier();
                // points[index].position.y = sharedMemory.points[index];
                // barrier();
                
                void main() {
                    uint index = uint(gl_GlobalInvocationID.x);
                    vec2 prev_position = points[index].prev_position.xy;
                    vec2 position =  [prev_position.x, points[index].position.y];
                    vec2 lposition = points[uint(gl_GlobalInvocationID.x - 1)].position.xy;
                    vec2 rposition = points[uint(gl_GlobalInvocationID.x + 1)].position.xy;
                    
                    // animation step
                    float u = -prev_position.y + 2*position.y + c2*(rposition.x - 2*position.x + lposition.y);
                    
                    // atomicExch(points[index].prev_position.y, points[index].position.y);
                    // atomicExch(points[index].position.y, u);
                    
                    // points[index].prev_position = points[index].position;
                    // points[index].position.y = u;
                    
                    points[index].position.y = u;
                }
            '''
        )    
        self.vao = self.ctx.vertex_array(self.compute_program, [])
        
    def compute(args):
        pass
                    
class Visualisation:
    def __init__(self, texture):
        self.ctx = moderngl.get_context()
        self.program = self.ctx.program(
            vertex_shader='''
                #version 430 core

                #include "uniform_buffer"
                
                uniform float courant;
                uniform float t;
                uniform float dt;
                uniform float dx;
                uniform float tx2;
                // uniform float c2;
                
                float c(float x) {
                    float a = courant;
                    // if (x > -0.9){
                    //     a *= 0.1;
                    // }
                    // if(x > -1000){
                    //     float difference = (x + 1000)/100;
                    //     a *= 1 - difference;
                    // }
                    return a * a;
                }
                
                float wave_source(float x,float t) {    
                    // return sin(t);          
                    if (t < 3.14){
                        if (x == 0){
                            return sin(t * 10 * sin(t * 0.5)) * 100000;
                        }
                        else{ return 0;}
                    }
                    else{ return 0;}
                }


                void main() {
                    uint index = uint(gl_VertexID + 1);
                    vec2 prev_position = Data.points[index].prev_position.xy;
                    vec2 position = Data.points[index].position.xy;
                    vec2 lposition = Data.points[uint(gl_VertexID)].position.xy;
                    vec2 rposition = Data.points[uint(gl_VertexID + 2)].position.xy;
                    
                    // animation step
                    // float u =  2*position.y - prev_position.y + c2*(rposition.y - 2*position.y + lposition.y) + dt * dt * wave_source(position.x, t);
                    float u =  2*position.y - prev_position.y + c(position.x)*(rposition.y - 2*position.y + lposition.y);
                    // float u =  2*position.y - prev_position.y + tx2*((c(position.x + 0.5*dx) * (rposition.y - position.y)) - (c(position.x - 0.5*dx) * (position.y - lposition.y)));
                    
                    // atomicExch(Data.points[index].prev_position.y, Data.points[index].position.y);
                    // atomicExch(Data.points[index].position.y, u);
                    
                    Data.points[index].prev_position = Data.points[index].position;
                    Data.points[index].position.y = u;
                    
                    
                    vec2 dposition = Data.points[index].position.xy / vec2(2560.0 , 1440.0) * 2;
                    gl_Position = vec4(dposition, 1.0, 1.0);
                    
                }
            ''',
            fragment_shader='''
                #version 430 core

                layout (location = 0) out vec4 out_color;
                

                void main() {
                    out_color = vec4(1.0, 1.0, 1.0, 1.0);
                }
            '''
        )
        self.vao = self.ctx.vertex_array(self.program, [])
        self.vao.vertices = 2560 * screen_resolution_multiplier - 2
        
    def render(self,t):
        '''
        Renders the wave equation simulation at time t.

        Parameters
        ----------
        t : float
            The current time in seconds.

        '''

        courant = WAVE_VEL * DELTA_T * screen_resolution_multiplier * 2560
        self.program['courant'] = courant
        # self.program['dx'] = 1/(2560 * screen_resolution_multiplier)
        # self.program['tx2'] = DELTA_T * DELTA_T * (2560 * screen_resolution_multiplier)**2
        # self.program['dt'] = DELTA_T
        # self.program['c2'] = courant**2
        # self.program['t'] = t
        self.ctx.enable_only(self.ctx.NOTHING)
        self.vao.render(self.ctx.LINE_STRIP)
                    

class Simulation:
    def __init__(self, texture,buffer):
        self.ctx = moderngl.get_context()
        self.program = self.ctx.program(
            vertex_shader='''
                #version 330 core
                
                // in vec2 in_pos;
                // in vec3 in_col;
                
                // out vec2 pos;
                // out vec3 col;


                vec2 positions[3] = vec2[](
                    vec2(-3.0, -1.0 ),
                    vec2(3.0, -1.0),
                    vec2(0.0, 2.0)
                );

                void main() {
                    gl_Position = vec4(positions[gl_VertexID], 1.0, 1.0);
                    // pos = in_pos;
                    // col = in_col;
                }
            ''',
            fragment_shader='''
                #version 330 core
                
                #include "uniform_buffer"

                uniform sampler2D Texture;
                uniform float t;
                uniform float dt;
                uniform float zoom;
                
                // in vec2 pos;
                // in vec3 col;

                layout (location = 0) out vec4 out_color;
                
                float distToSegment(vec2 p, vec2 v, vec2 w) {
                    vec2 line = w - v;
                    float lenSq = dot(line, line);  // Length squared of the line segment

                    // Project the fragment onto the line segment, clamping to the segment
                    float t = clamp(dot(p - v, line) / lenSq, 0.0, 1.0);
                    vec2 projection = v + t * line;  // The point on the line segment closest to 'p'
                    
                    // Return the distance from the fragment to the closest point on the segment
                    return length(p - projection);
                }
                
                
                void main() {
                    float bullshit = t + dt + zoom;
                    float minDist = 1e10;
                    
                    vec2 pc = (ivec2(gl_FragCoord.xy) - vec2(2560,1440)/2);
                    
                    int i = 0;
                    for (i = 0; i < 2560 - 1; i++) {
                        vec2 ppos1 = points[i].position.xy;
                        vec2 ppos2 = points[i+1].position.xy;
                        // this triples the frame rate, you're welcome
                        // if (dot(pc - ppos1, pc - ppos1) > dot(ppos2 - ppos1, ppos2 - ppos1)) {
                        // if (dot(pc - ppos1, pc - ppos1) > 200) {
                        //     continue;
                        // }
                        // float d = distToSegment(pc, ppos1, ppos2);
                        // minDist = min(minDist, d);
                        
                        
                        // for circular display:
                        vec2 dist = pc - ppos1;
                        if (dot(dist, dist) < 4.0) {
                            out_color = vec4(points[i].color.xyz,0);
                        }
                    }
                    
                    // if (minDist < 1) {
                    //     out_color = vec4(1.0, 1.0, 1.0, 1.0);  // Line color, e.g., white
                    // } else {
                    //     out_color = vec4(0.0, 0.0, 0.0, 1.0);  // Background color, e.g., black
                    // }
                    
                    // float lineHeight = 10.0;
                    // if (pc.y - lineHeight < points[uint(gl_FragCoord.x)].position.y && pc.y + lineHeight > points[uint(gl_FragCoord.x)].position.y) {
                    //     out_color = vec4(points[uint(gl_FragCoord.x)].color.xyz,0);
                    // }
                    // uint index = uint(gl_FragCoord.x);
                    // 
                    // float x = points[index].position.y;
                    // out_color = vec4(x,0,0,0);
                }
            ''',
        )

        self.sampler = self.ctx.sampler(texture=texture)
        # self.vao = self.ctx.vertex_array(self.program, [(buffer, '2f 3f', 'in_pos', 'in_col')])
        self.vao = self.ctx.vertex_array(self.program, [])
        self.vao.vertices = 3

    def render(self, now,zoom):
        self.ctx.enable_only(self.ctx.NOTHING)
        self.sampler.use()
        self.program['t'] = now
        self.program['dt'] = DELTA_T
        self.program['zoom'] = zoom
        self.vao.render()


def wave_source(x,t):
    is_early = t < math.pi
    if is_early:
        if x == 0:
            print("SIN")
            return math.sin(t)
        else:
            return 0
    else:
        return 0

def variable_wave_vel(x,t):
    return WAVE_VEL * x <= 0.5

class UniformBuffer:
    def __init__(self,screen_dimensions):
        self.ctx = moderngl.get_context()
        self.data = bytearray(math.ceil(screen_dimensions[0] * 32 * screen_resolution_multiplier))
        self.ubo = self.ctx.buffer(self.data)
    
    def set_point(self, point_index, prev_position, position):
        offset = point_index * 32
        # add padding to make it 16 bytes, because of fucking reasons???
        self.data[offset + 0 : offset + 16] = struct.pack('4f', *position, 0.0, 0.0)
        self.data[offset + 16 : offset + 32] = struct.pack('4f', *prev_position, 0.0, 0.0)

    def use(self):
        self.ubo.write(self.data)
        self.ubo.bind_to_storage_buffer(0)
        
        
class PostProcessVisuals:
    def __init__(self):
        self.program = self.ctx.program(
            vertex_shader='''
                #version 330

                in vec2 in_vert;
                out vec2 vert;

                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    vert = in_vert;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform sampler2D screen;
                in vec2 vert;
                out vec4 frag_color;

                void main() {
                    frag_color = texture(screen, vert);
                }
            ''',
        )

    def render(self, now):
        self.program['screen'] = self.screen
        self.vao.render(moderngl.TRIANGLE_STRIP)


class Scene:
    def __init__(self):
        self.ctx = moderngl.get_context()

        size = pygame.display.get_window_size()
        self.screen = self.ctx.texture(size, 4)
        self.depth = self.ctx.depth_texture(size)

        self.framebuffer = self.ctx.framebuffer(
            color_attachments=[self.screen],
            depth_attachment=self.depth,
        )
        
        courant = WAVE_VEL * DELTA_T * screen_resolution_multiplier * 2560
        print(courant)
        c2 = courant**2

        self.uniform_buffer = UniformBuffer(size)
        x = np.linspace(-size[0]/2,size[0]/2 - 1,math.ceil(size[0] * screen_resolution_multiplier))
        y = np.ndarray(math.ceil(size[0] * screen_resolution_multiplier))
        # math.radians(x/20)
        # y = np.sin(math.radians(x/20),y) * size[1]/32
        # set inital condition
        for i in range(len(x)):
            # y[i] = -math.sin(min(x[i] * math.pi * (2/size[0]),0)) * size[1]/16
            # theta = (x[i] - size[0]/2 ) * math.pi * (2/size[0])
            theta = (x[i] + size[0]/2) * 1/64
            # theta = max(theta, 0)
            # y[i] = math.sin(theta * 8 * (theta * 8 < math.pi)) * size[1]/16
            # y[i] = 0
            # y[i] = max(100 - abs(x[i]),0)**2 * 1/32
            # y[i] = ((x[i] * (abs(x[i]) < 150))**2 - 150 * 150 * (abs(x[i]) < 150)) * -1/64
            b1 = -math.pi
            b2 = math.pi
            g = 3.5
            if theta > b1 and theta < b2:
                y[i] = (pow(g, math.sin(theta + math.pi/2)) - pow(g, -1)) * size[1]/16
            else:
                y[i] = 0
            
        for i in range(1, math.ceil(size[0] * screen_resolution_multiplier) - 1):
            prev_position = (x[i], y[i])
            # compute first step on cpu:
            # unnecessary amount of brackets because i dont trust the cpu
            position =  [x[i], y[i]]
            # position[1] = y[i] - 0.5*c2*(y[i + 1] - 2 * y[i] + y[i - 1]) + 0.5*DELTA_T*DELTA_T * wave_source(x[i], DELTA_T)
            position[1] = y[i] - 0.5*c2*(y[i + 1] - 2 * y[i] + y[i - 1])
            
            # du = position[1] - prev_position[1]
            
            self.uniform_buffer.set_point(i, prev_position, position)
            
        # insert boundary conditions
        self.uniform_buffer.set_point(0, (x[0], 0), (x[0], 0))
        self.uniform_buffer.set_point(math.ceil(size[0] * screen_resolution_multiplier) - 1, (x[-1], 0), (x[-1], 0))            
        self.uniform_buffer.use()

        
            
        # vertices = np.dstack([x, y, r, g, b])
        # self.uniform_buffer.data = vertices.astype("f4").tobytes()
        
        # vbo = self.ctx.buffer(vertices.astype("f4").tobytes())
        
        # self.simulation = Simulation(self.screen,None)
        
        # self.compute_shader = Compute()
        
        self.visualisation = Visualisation(self.screen)
        
        self.frame_count = 0
        


    def render(self,t,zoom):
        # now = pygame.time.get_ticks() / 1000.0
        size = pygame.display.get_window_size()
        
        now = t

        eye = (math.cos(now), math.sin(now), 0.5)


        
        
        
        self.ctx.screen.use()
        
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.enable(self.ctx.DEPTH_TEST)
        
        # this works lol, no convoluted math needed
        
        # insert neumann condition on left boundary
        # read x1 position from uniform buffer and write it to x0
        self.uniform_buffer.ubo.read_into(self.uniform_buffer.data)
        # x = self.uniform_buffer.ubo.read(16,offset=0)
        y = self.uniform_buffer.data[32 + 4:32+8]
        self.uniform_buffer.data[0 + 4:0 + 8] = y
        
        # p = int(math.ceil(size[0] * screen_resolution_multiplier)/2 -1)

        # y = self.uniform_buffer.data[32 * (p - 1) + 8:32 * (p - 1) +12]
        # self.uniform_buffer.data[32 * p + 4:32 * p + 8] = y
        
        self.uniform_buffer.ubo.write(self.uniform_buffer.data)
        
        
        #  execute black magic on the gpu, PARRALLELISE IT!
        self.visualisation.render(now)
        
        # self.uniform_buffer.use()

        # self.simulation.render(now,zoom)
        
        
        # save frame as image
        # self.framebuffer.use()
        
        # self.visualisation.render(now)

        # bytes_data = self.framebuffer.read(components=4)
        # image = Image.frombytes("RGBA", size, bytes_data)
        # image.save("./frames/frame_{:04d}.png".format(self.frame_count))
        # self.frame_count += 1


init_includes()

DELTA_T = 0.0039
WAVE_VEL = 0.1
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
    
# TODO: implement fullscreen
# TODO: implement proper scaling and distance calculations
# DONE: implement neumann boundary conditions