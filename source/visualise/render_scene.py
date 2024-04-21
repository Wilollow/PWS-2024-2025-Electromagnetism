#!/usr/bin/env python
import inspect
import os
import sys
import importlib

from manimlib import *
# from manimlib.config import get_module
# from manimlib.extract_scene import is_child_scene

# from electricField.emfield import InteractiveDevelopment

class InteractiveDevelopment(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(BLUE, opacity=0.5)
        circle.set_stroke(BLUE_E, width=4)
        square = Square()
        self.play(ShowCreation(square))
        self.wait()
        self.play(ReplacementTransform(square, circle))
        self.wait()
        self.play(circle.animate.stretch(4, 0))
        self.play(Rotate(circle, 90 * DEGREES))
        self.play(circle.animate.shift(2 * RIGHT).scale(0.25))
        text = Text("""
            In general, using the interactive shell
            is very helpful when developing new scenes
        """)
        self.play(Write(text))
        always(circle.move_to, self.mouse_point)


class RenderScene(Scene):
    def construct(self):
        module = InteractiveDevelopment()
        return module.construct()



# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         raise Exception("No module given.")
#     module_name = sys.argv[1]
#     print("Rendering animations from {}".format(module_name))
#     stage_scenes(module_name)
