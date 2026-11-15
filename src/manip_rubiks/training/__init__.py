"""Policy training (PPO with an optional privileged critic)."""

import os
import sys

# Training renders evaluation videos offscreen. Headless Linux has no window system for MuJoCo's
# default GLFW backend, so use EGL; this must happen before ``mujoco`` is first imported.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
