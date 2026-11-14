import os
import sys

# Headless Linux has no window system for MuJoCo's default GLFW backend; use EGL for offscreen rendering.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
