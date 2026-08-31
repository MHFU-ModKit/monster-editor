"""GLSL, in one file so a driver complaint has one place to look.

`#version 330 core` throughout — see :mod:`.context` for why that is the floor and the
ceiling. No compatibility profile exists on macOS, so there is no fixed-function path
to fall back on: every draw here has a VAO and a program.

Programs are cached per context. A `Scene` reopened, a panel resized or a window
rebuilt must not recompile the same source, and moderngl objects belong to the context
that made them — hence the key.
"""
from __future__ import annotations

from typing import Dict, Tuple

#: flat-coloured lines and points: the grid, the world axes, the bone overlay. Colour
#: rides on the vertex so one draw call can carry several tints.
LINE_VS = """
#version 330 core
uniform mat4 u_mvp;
uniform float u_point_size;
in vec3 in_pos;
in vec4 in_color;
out vec4 v_color;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    gl_PointSize = u_point_size;
    v_color = in_color;
}
"""

LINE_FS = """
#version 330 core
uniform float u_alpha;
in vec4 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color.rgb, v_color.a * u_alpha);
    if (f_color.a < 0.004) discard;
}
"""

#: the ground plane, drawn as one full quad and shaded procedurally rather than as a
#: mesh of line segments. It stays crisp at every zoom and costs two triangles: the
#: line width follows the on-screen derivative, so a grid seen from a long way off
#: fades instead of turning into moire.
GROUND_VS = """
#version 330 core
uniform mat4 u_mvp;
in vec3 in_pos;
out vec3 v_world;
void main() {
    v_world = in_pos;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

GROUND_FS = """
#version 330 core
uniform float u_cell;        // minor cell size, world units
uniform float u_major;       // every Nth line is a major one
uniform float u_extent;      // fade to nothing at this radius
uniform vec3  u_minor_color;
uniform vec3  u_major_color;
uniform vec3  u_axis_x;
uniform vec3  u_axis_z;
in vec3 v_world;
out vec4 f_color;

// distance to the nearest gridline, in PIXELS, per axis.
vec2 line_px(vec2 p, float cell) {
    vec2 d = fwidth(p) / cell;
    vec2 g = abs(fract(p / cell - 0.5) - 0.5) / max(d, vec2(1e-6));
    return g;
}

void main() {
    vec2 p = v_world.xz;
    vec2 minor = line_px(p, u_cell);
    vec2 major = line_px(p, u_cell * u_major);
    float m_minor = 1.0 - min(min(minor.x, minor.y), 1.0);
    float m_major = 1.0 - min(min(major.x, major.y), 1.0);

    // the two world axes, drawn in place of a gridline so the origin is unambiguous.
    vec2 ax = abs(p) / max(fwidth(p), vec2(1e-6));
    float on_z = 1.0 - min(ax.x, 1.0);          // the line x == 0 runs along Z
    float on_x = 1.0 - min(ax.y, 1.0);

    vec3 rgb = u_minor_color;
    float a = m_minor * 0.30;
    if (m_major > a) { rgb = u_major_color; a = m_major * 0.50; }
    if (on_z * 0.75 > a) { rgb = u_axis_z; a = on_z * 0.75; }
    if (on_x * 0.75 > a) { rgb = u_axis_x; a = on_x * 0.75; }

    // circular fade, so the plane has no visible square edge. ⚠️ edge0 < edge1 or
    // smoothstep runs backwards and the grid vanishes at the ORIGIN and appears at
    // the rim — which reads as "the ground is a ceiling".
    a *= 1.0 - smoothstep(u_extent * 0.30, u_extent * 0.50, length(p));
    if (a < 0.004) discard;
    f_color = vec4(rgb, a);
}
"""


#: the skinned monster. Positions and normals arrive ALREADY DEFORMED — the skinning is
#: `core.pose`'s, on the CPU, so the picture and a headless assertion come out of the
#: same maths that `stretch.py` is the oracle for. Doing it again in GLSL would make a
#: second oracle, which is the one thing this package must not grow.
MESH_VS = """
#version 330 core
uniform mat4 u_mvp;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
in float in_group;      // vertex group index, for per-group colouring
in float in_bone;       // dominant joint, for highlight / isolate
out vec3 v_normal;
out vec2 v_uv;
out float v_group;
out float v_bone;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_normal = in_normal;
    v_uv = in_uv;
    v_group = in_group;
    v_bone = in_bone;
}
"""

#: `u_mode`: 0 textured, 1 flat grey, 2 per-vertex-group colour.
#: `u_isolate`: 0 draw everything, 1 only the tagged joints, 2 everything BUT them —
#: `MHFU_VIEW_ONLY`'s two meanings, which answer "what IS that patch of geometry" and
#: "what is left without it".
MESH_FS = """
#version 330 core
uniform sampler2D u_tex;
uniform sampler2D u_bone_tag;   // (n_bones x 1) R8: 1 = tagged
uniform int   u_mode;
uniform int   u_isolate;
uniform int   u_n_bones;
uniform int   u_has_tex;
uniform vec3  u_light;
uniform vec3  u_tint;
uniform float u_alpha;
in vec3 v_normal;
in vec2 v_uv;
in float v_group;
in float v_bone;
out vec4 f_color;

// a stable, well-spread colour per integer — enough to tell 214 groups apart.
vec3 hue(float i) {
    float h = fract(i * 0.6180339887);
    vec3 k = fract(vec3(h) + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0));
    return 0.35 + 0.55 * abs(fract(k * 3.0) * 2.0 - 1.0);
}

float bone_tag(float bone) {
    if (bone < 0.0 || u_n_bones <= 0) return 0.0;
    // sample the texel CENTRE; (bone / n) lands on the boundary and can pick the
    // neighbour under any filtering that is not exactly nearest.
    return texture(u_bone_tag, vec2((bone + 0.5) / float(u_n_bones), 0.5)).r;
}

void main() {
    float tag = bone_tag(v_bone);
    if (u_isolate == 1 && tag < 0.5) discard;
    if (u_isolate == 2 && tag >= 0.5) discard;

    vec3 base;
    if (u_mode == 2)            base = hue(v_group);
    else if (u_mode == 1 || u_has_tex == 0) base = vec3(0.72, 0.72, 0.74);
    else {
        vec4 t = texture(u_tex, v_uv);
        if (t.a < 0.35) discard;          // TMH alpha punches out fins and membranes
        base = t.rgb;
    }
    base *= u_tint;
    if (tag >= 0.5 && u_isolate == 0) base = mix(base, vec3(0.90, 0.13, 0.13), 0.65);

    // two-sided lambert with a fill: a monster PAC's winding is not reliable, so a
    // one-sided term leaves whole plates black.
    vec3 n = normalize(v_normal);
    float d = abs(dot(n, normalize(u_light)));
    f_color = vec4(base * (0.35 + 0.65 * d), u_alpha);
}
"""


def _cache(ctx) -> Dict[Tuple[str, ...], object]:
    store = getattr(ctx, "_mhfu_programs", None)
    if store is None:
        store = {}
        try:
            ctx._mhfu_programs = store
        except AttributeError:                               # pragma: no cover
            return {}
    return store


def program(ctx, vertex: str, fragment: str):
    """Compile once per context, with the GLSL log attached to the exception."""
    store = _cache(ctx)
    key = (vertex, fragment)
    prog = store.get(key)
    if prog is None:
        try:
            prog = ctx.program(vertex_shader=vertex, fragment_shader=fragment)
        except Exception as e:                               # pragma: no cover
            raise RuntimeError("shader failed to compile on %s:\n%s"
                               % (ctx.info.get("GL_RENDERER", "?"), e)) from e
        store[key] = prog
    return prog


def line_program(ctx):
    return program(ctx, LINE_VS, LINE_FS)


def ground_program(ctx):
    return program(ctx, GROUND_VS, GROUND_FS)


def mesh_program(ctx):
    return program(ctx, MESH_VS, MESH_FS)
