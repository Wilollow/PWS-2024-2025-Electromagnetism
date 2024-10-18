# import moderngl_window as glw
# import moderngl as gl

# import numpy as np


# window_cls = glw.get_local_window_cls('pyglet')

# window = window_cls(

#     size=(512, 512), fullscreen=False, title='ModernGL Window',

#     resizable=False, vsync=True, gl_version=(3, 3)

# )

# ctx = window.ctx

# glw.activate_context(window, ctx=ctx)

# window.clear()

# window.swap_buffers()


# prog = ctx.program(

#     vertex_shader="""

#         #version 330


#         in vec2 in_vert;

#         in vec3 in_color;


#         out vec3 v_color;


#         void main() {

#             v_color = in_color;

#             gl_Position = vec4(in_vert, 0.0, 1.0);

#         }

#     """,

#     fragment_shader="""

#         #version 330


#         in vec3 v_color;


#         out vec3 f_color;


#         void main() {

#             f_color = v_color;

#         }

#     """,

# )


# x = np.linspace(-1.0, 1.0, 50)

# y = np.random.rand(50) - 0.5

# r = np.zeros(50)

# g = np.ones(50)

# b = np.zeros(50)


# vertices = np.dstack([x, y, r, g, b])


# vbo = ctx.buffer(vertices.astype("f4").tobytes())

# vao = ctx.vertex_array(prog, vbo, "in_vert", "in_color")


# fbo = ctx.framebuffer(

#     color_attachments=[ctx.texture((512, 512), 3)]

# )



# while not window.is_closing:

#     fbo.use()

#     fbo.clear(0.0, 0.0, 0.0, 1.0)

#     vao.render(gl.LINE_STRIP)


#     ctx.copy_framebuffer(window.fbo, fbo)


#     window.swap_buffers()

# import moderngl
# import pygame

# pygame.init()
# pygame.display.set_mode((800, 800), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

# ctx = moderngl.get_context()


# pygame.display.flip()


# while screen and stuff:
#     render with fixed dt at very low value
    
#     animprogresstime = pygame.time.get_ticks() / 1000
    
#     # if realtime, approach would be
#     # dt fixed to 1/fps
#     # if pygame.time.get_ticks() / 1000 <= animprogresstime:
#     #     do useless stuff to pass time
        
#     if pygame.time.get_ticks() / 1000 >= animprogresstime:
#         render(1/camera.fps)
        
        
# def render(dt):
#     self.increment_time(dt)
    
#     # not going to be realtime, hence either save to some kind of video, or slow dt down a lot
#     # chose not realtime, as one is then bound by dt of 1/fps for accuracy, not by the time it takes to render, accuracy too low.

import math
import os
import struct
import sys

import glm
import moderngl
import pygame
from objloader import Obj
from PIL import Image

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((2560, 1440), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)


def init_includes():
    ctx = moderngl.get_context()

    # ctx.includes['uniform_buffer'] = '''
    #     struct Light {
    #         vec4 light_position;
    #         vec4 light_color;
    #         float light_power;
    #     };

    #     layout (std140) uniform Common {
    #         mat4 camera;
    #         vec4 camera_position;
    #         Light lights[2];
    #     };
    # '''

    # ctx.includes['blinn_phong'] = '''
    #     vec3 blinn_phong(
    #             vec3 vertex, vec3 normal, vec3 camera_position, vec3 light_position, float shininess, vec3 ambient_color,
    #             vec3 diffuse_color, vec3 light_color, vec3 spec_color, float light_power) {

    #         vec3 light_dir = light_position - vertex;
    #         float light_distance = length(light_dir);
    #         light_distance = light_distance * light_distance;
    #         light_dir = normalize(light_dir);

    #         float lambertian = max(dot(light_dir, normal), 0.0);
    #         float specular = 0.0;

    #         if (lambertian > 0.0) {
    #             vec3 view_dir = normalize(camera_position - vertex);
    #             vec3 half_dir = normalize(light_dir + view_dir);
    #             float spec_angle = max(dot(half_dir, normal), 0.0);
    #             specular = pow(spec_angle, shininess);
    #         }

    #         vec3 color_linear = ambient_color +
    #             diffuse_color * lambertian * light_color * light_power / light_distance +
    #             spec_color * specular * light_color * light_power / light_distance;

    #         return color_linear;
    #     }
    # '''

    # ctx.includes['calculate_lights'] = '''
    #     vec3 calculate_lights(vec3 vertex, vec3 normal, vec3 color, vec3 camera_position) {
    #         vec3 result = vec3(0.0);
    #         for (int i = 0; i < 2; ++i) {
    #             result += blinn_phong(
    #                 vertex, normal, camera_position, lights[i].light_position.xyz, 16.0, color * 0.05,
    #                 color, lights[i].light_color.rgb, vec3(1.0, 1.0, 1.0), lights[i].light_power
    #             );
    #         }
    #         return result;
    #     }
    # '''

    # ctx.includes['srgb'] = '''
    #     vec3 srgb_to_linear(vec3 color) {
    #         return pow(color, vec3(2.2));
    #     }
    #     vec3 linear_to_srgb(vec3 color) {
    #         return pow(color, vec3(1.0 / 2.2));
    #     }
    # '''

    # ctx.includes['hash13'] = '''
    #     float hash13(vec3 p3) {
    #         p3 = fract(p3 * 0.1031);
    #         p3 += dot(p3, p3.zyx + 31.32);
    #         return fract((p3.x + p3.y) * p3.z);
    #     }
    # '''

                    # vec2(-1.0, -1.0),
                    # vec2(3.0, -1.0),
                    # vec2(-1.0, 3.0)

class Simulation:
    def __init__(self, texture):
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
                #version 330 core

                uniform sampler2D Texture;
                uniform float t;
                uniform float zoom;
                
                bool auxiliary = false;
                

                layout (location = 0) out vec4 out_color;
                
                float propagationspeed = 20;
                
                vec2 chargestartpos = vec2(0,0);
                
                vec2 position(float t){
                    if (t < 0){
                        return chargestartpos;
                    }
                    return chargestartpos + vec2(0,10 * sin(1 * t));
                }
                
                vec2 velocity(float t){
                    return vec2(0,cos(t) * 10);
                }
                
                vec2 acceleration(float t){
                    return vec2(-sin(t) * 10);
                }
                
                vec2 force(float t,vec2 pos){
                    // vec2 origin = position(t);
                    // vec2 origintopos = pos - origin;
                    // float dist = length(origintopos);
                    // float dt = dist / propagationspeed;
                    // float dtcenter = length(pos - vec2(2560,1440)/2) / propagationspeed;
                    // float modt = t - (40 * floor((t)/40));
                    
                    // __________________________________________
                    float dt = length(pos - position(0)) / propagationspeed;
                    

                    if ((t - dt) - (40 * floor((t - dt)/40)) < 0.2 / zoom && (t - dt) > 0 && auxiliary){
                        return vec2(0,0);
                    }
                    
                    // int emissiondisplayinterval = 1;
                    // 
                    // if(cos(emissiondisplayinterval * (dt)) * 0.5 + 0.5 > 0.99 && (t - dt) > 0){
                    //     return vec2(0,0);
                    // }
                    
                    vec2 dorigin = position(t - dt);
                    vec2 dorigintopos = pos - dorigin;
                    float ddistsqrd = dot(dorigintopos, dorigintopos);
                    //float brightnessmult = 40000 * zoom * zoom; // enable for visiblity of structure only, nothing else makes sense anymore
                    float brightnessmult = 40000; // enable for visiblity of structure only, nothing else makes sense anymore
                    
                    vec2 term1 = normalize(dorigintopos) / ddistsqrd;
                    return abs(term1) * brightnessmult;
                }

                void main() {
                    vec2 pos = vec2(2560,1440)/2;

                    vec2 pc = (ivec2(gl_FragCoord.xy) + (0.5,0.5) - vec2(2560,1440)/2) / zoom; 
                    vec2 electronpos = position(t);
                    vec2 electrontopos = pc - electronpos;
                    float dist = dot(electrontopos,electrontopos);
                    
                    float blue = 0;
                    
                    if(dist < 100.0){
                        blue = 1;
                    }
                    
                    
                    vec2 power = force(t,pc);
                    
                    out_color = vec4(power,blue,0);
                    // out_color = vec4(length(power),length(power),length(power),1);
                }
            ''',
        )
                    # vec2 postopixel = pc - pos;
                    # float dist = dot(postopixel, postopixel);
                    # float powr, powg, powb;
                    # powr = sin(0.1 * sqrt(dist) - t * 2) * 0.5 + 0.5;
                    # powg = cos(1 * sqrt(dist) - t * 10) * 0.5 + 0.5;
                    # powb = sin(0.01 * sqrt(dist) + t * 1) * 0.5 + 0.5;
        self.sampler = self.ctx.sampler(texture=texture)
        self.vao = self.ctx.vertex_array(self.program, [])
        self.vao.vertices = 3

    def render(self, now,zoom):
        self.ctx.enable_only(self.ctx.NOTHING)
        self.sampler.use()
        self.program['t'] = now
        self.program['zoom'] = zoom
        # print(self.program['time'])
        self.vao.render()


class UniformBuffer:
    def __init__(self):
        self.ctx = moderngl.get_context()
        self.data = bytearray(1024)
        self.ubo = self.ctx.buffer(self.data)

    def set_camera(self, eye, target):
        proj = glm.perspective(45.0, 1.0, 0.1, 1000.0)
        look = glm.lookAt(eye, target, (0.0, 0.0, 1.0))
        camera = proj * look
        self.data[0:64] = camera.to_bytes()
        self.data[64:80] = struct.pack('4f', *eye, 0.0)

    def set_light(self, light_index, position, color, power):
        offset = 80 + light_index * 48
        self.data[offset + 0 : offset + 16] = struct.pack('4f', *position, 0.0)
        self.data[offset + 16 : offset + 32] = struct.pack('4f', *color, 0.0)
        self.data[offset + 32 : offset + 36] = struct.pack('f', power)

    def use(self):
        self.ubo.write(self.data)
        self.ubo.bind_to_uniform_block()


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

        self.uniform_buffer = UniformBuffer()

        self.simulation = Simulation(self.screen)


    def render(self,t,zoom):
        # now = pygame.time.get_ticks() / 1000.0
        
        now = t

        eye = (math.cos(now), math.sin(now), 0.5)

        self.framebuffer.use()

        self.ctx.clear()
        self.ctx.enable(self.ctx.DEPTH_TEST)

        # self.uniform_buffer.set_camera(eye, (0.0, 0.0, 0.0))
        # self.uniform_buffer.set_light(
        #     light_index=0,
        #     position=(1.0, 2.0, 3.0),
        #     color=(1.0, 1.0, 1.0),
        #     power=10.0,
        # )
        # self.uniform_buffer.set_light(
        #     light_index=1,
        #     position=(-2.0, -1.0, 4.0),
        #     color=(1.0, 1.0, 1.0),
        #     power=10.0,
        # )
        
        self.uniform_buffer.use()

        # self.crate.render((0.0, 0.0, 0.0), 0.2)

        # self.color_material.color = (1.0, 0.0, 0.0)
        # self.car.render((-0.4, 0.0, 0.0), 0.2)

        # self.color_material.color = (0.0, 0.0, 1.0)
        # self.car.render((0.4, 0.0, 0.0), 0.2)

        self.ctx.screen.use()
        self.simulation.render(now,zoom)


init_includes()
scene = Scene()

DELTA_T = 0.05
global_t = 0

zoom = 1

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

    
    scene.render(t = global_t,zoom = zoom)
    global_t += DELTA_T

    pygame.display.flip()