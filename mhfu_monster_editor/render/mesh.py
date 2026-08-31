"""The monster: skinned geometry, its own TMH textures, and the draw modes.

    mesh = SkinnedMesh(ctx, scene)
    mesh.set_pose(scene.pose(7, 31))        # CPU skinning, ~0.5 ms for a Tigrex
    mesh.render(mvp)

Three things this gets right, each of which has cost this project time before:

1. **The skin is the PAC's own.** Every vertex carries the `(bone, weight)` list
   `pmo._attach_influences` resolved, by way of `Scene.merged`. The old "mesh group N is
   welded to bone N" fallback put 166 of the native Tigrex's 214 groups on one joint —
   the "cluster of parts floating beside it" in `blender_mhfu/`'s notes.
2. 🔴 **The bind pose cannot validate skinning.** At bind every joint is at bind, so
   every weighting looks perfect and a mismapped rig looks fine. The viewport therefore
   opens on a POSED frame. → :func:`default_pose`
3. **`material` is the PMO's texture index**, resolved against the decoded TMH set the
   way `blender_mhfu/importer.py` resolves it. `Scene` has already done the resolving
   and says so in its notes when an index has no image; here that group simply draws
   untextured rather than sampling whatever texture happened to be bound.

Skinning is on the CPU, which issue #6 asks for ("GPU later if it matters") and which
is also the right call for a different reason: `core.pose` is pinned against
`tools/mhfu_model/stretch.py`, and a GLSL skinner would be a **second implementation of
the same maths** — the one thing this package must not grow. It is not the bottleneck
either: the whole Tigrex deforms in 0.44 ms through `Scene.merged`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera import Bounds, gl_bytes
from .shaders import mesh_program

#: `u_mode`
MODE_TEXTURED = 0
MODE_FLAT = 1
MODE_VGROUP = 2
MODES = ("textured", "flat", "vgroup")

#: `u_isolate`
ISOLATE_OFF = 0
ISOLATE_ONLY = 1        # MHFU_VIEW_ONLY=1 — show ONLY the tagged joints
ISOLATE_HIDE = 2        # MHFU_VIEW_ONLY=2 — show everything BUT them

#: where the key light sits, in world space. Over the animal's left shoulder and
#: slightly in front, so a profile view is not lit flat.
LIGHT = (-0.45, 0.80, 0.40)


def default_pose(scene):
    """A POSED frame to open on, never bind. Returns ``(clip_or_None, frame)``.

    🔴 A rest-pose render cannot validate skinning: at bind every joint is at bind, so
    a mismapped rig and a correct one are the same picture and the one fault a viewport
    exists to catch is invisible. Issue #6 makes a posed frame the default for exactly
    that reason.

    *Which* frame is chosen from the data, not from a hardcoded slot number, because a
    ported monster's slots are not the host's:

    1. **looping, whole-rig clips** — locomotion and idles loop; one-shot attacks and
       the partial head-only slots do not;
    2. of those, the ones driving the **most joints** — the more of the rig a pose
       exercises, the more of the skinning it can falsify;
    3. and among *those*, the **median frame count**. Both extremes are traps: the
       shortest is a twitch, and the longest is the sleep loop — on the native Tigrex
       the longest (slot 58, 360 frames) is him curled up asleep, which reads as a
       broken import at a glance.

    Native Tigrex -> slot 3 (a standing idle); the built Zinogre -> slot 69. Falls back
    through non-looping clips to bind for a PAC with no animation at all.
    """
    pool = [c for c in scene.clips if c.loop and c.whole_rig] \
        or [c for c in scene.clips if c.whole_rig] \
        or list(scene.clips)
    if not pool:
        return None, 0.0
    most = max(len(c.driven) for c in pool)
    widest = sorted([c for c in pool if len(c.driven) == most], key=lambda c: c.frames)
    clip = widest[(len(widest) - 1) // 2]
    return clip, clip.frames * 0.5


class SkinnedMesh:
    """One vertex buffer for the whole animal, drawn in one batch per texture.

    A big-monster group averages 19 vertices, so 214 draw calls would be pure call
    overhead. Groups are merged into a single interleaved buffer — the same merge
    `Scene.merged` does for skinning, in the same order — and the index buffer is
    partitioned by texture, giving one draw per TMH image (five, on a Tigrex).
    """

    def __init__(self, ctx, scene) -> None:
        self.ctx = ctx
        self.scene = scene
        self.prog = mesh_program(ctx)
        self.mode = MODE_TEXTURED
        self.isolate = ISOLATE_OFF
        self.alpha = 1.0
        self.tint = (1.0, 1.0, 1.0)
        self.light = LIGHT

        #: the DEFORMED extent of the last pose — what the camera should frame. Bind
        #: bounds are the wrong thing to frame on: a rig whose bind is a splayed
        #: T-pose puts the posed animal well off the bind centre.
        self.bounds = Bounds.of(np.zeros((1, 3)))

        self._binding = scene.merged
        self._n = self._binding.n_vertices
        self._bind_normals = _merged_normals(scene)
        self._textures = _upload_textures(ctx, scene)
        self._batches = _index_batches(ctx, scene)
        self._tags = _BoneTags(ctx, scene.rig.n_bones)

        # dynamic: positions + normals change every pose. static: uv, group, bone.
        self._dyn = ctx.buffer(reserve=max(self._n * 6 * 4, 4), dynamic=True)
        self._static = ctx.buffer(_static_attributes(scene, self._n).tobytes())
        self._vaos = {
            tex: ctx.vertex_array(
                self.prog,
                [(self._dyn, "3f 3f", "in_pos", "in_normal"),
                 (self._static, "2f 1f 1f", "in_uv", "in_group", "in_bone")],
                ibo)
            for tex, ibo in self._batches.items()}
        self.set_pose(None)

    # ---- posing ------------------------------------------------------- #
    def set_pose(self, pose) -> None:
        """Deform to ``pose``, or to the bind pose when it is None. CPU, then upload."""
        if pose is None:
            pos, nrm = self._binding.positions, self._bind_normals
        else:
            deform = pose.deform
            pos = self._binding.apply(deform)
            nrm = self._binding.apply_directions(deform, self._bind_normals)
        data = np.concatenate([np.asarray(pos, dtype="f4"),
                               np.asarray(nrm, dtype="f4")], axis=1)
        self._dyn.write(np.ascontiguousarray(data, dtype="f4").tobytes())
        self.bounds = Bounds.of(pos) if len(pos) else Bounds.of(np.zeros((1, 3)))

    # ---- highlighting ------------------------------------------------- #
    def tag_joints(self, joints) -> None:
        """Paint these joints red — `MHFU_VIEW_HILITE`, and what `isolate` acts on.

        A vertex belongs to the joint holding most of its weight
        (:meth:`SkinBinding.dominant`), which is what `render_port_views.py`'s
        `dominant_joint` decides per OBJECT — per vertex is the finer version of the
        same rule.
        """
        self._tags.set(joints)

    @property
    def tagged(self) -> Tuple[int, ...]:
        return self._tags.joints

    # ---- drawing ------------------------------------------------------ #
    def render(self, mvp: np.ndarray, *, wireframe: bool = False) -> None:
        p = self.prog
        p["u_mvp"].write(gl_bytes(mvp))
        p["u_mode"].value = int(self.mode)
        p["u_isolate"].value = int(self.isolate)
        p["u_n_bones"].value = int(self.scene.rig.n_bones)
        p["u_light"].value = tuple(self.light)
        p["u_tint"].value = tuple(self.tint)
        p["u_alpha"].value = float(self.alpha)
        p["u_tex"].value = 0
        p["u_bone_tag"].value = 1
        self._tags.use(1)

        was = self.ctx.wireframe
        self.ctx.wireframe = bool(wireframe)
        try:
            for tex_index, vao in self._vaos.items():
                image = self._textures.get(tex_index)
                p["u_has_tex"].value = 1 if image is not None else 0
                if image is not None:
                    image.use(0)
                vao.render()
        finally:
            self.ctx.wireframe = was

    # ---- teardown ----------------------------------------------------- #
    def release(self) -> None:
        for vao in self._vaos.values():
            vao.release()
        for ibo in self._batches.values():
            ibo.release()
        for tex in self._textures.values():
            tex.release()
        self._tags.release()
        self._dyn.release()
        self._static.release()
        self._vaos, self._batches, self._textures = {}, {}, {}

    def __repr__(self) -> str:
        return ("<SkinnedMesh %s: %d verts, %d batches, %d textures>"
                % (self.scene.name, self._n, len(self._batches), len(self._textures)))


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def _merged_normals(scene) -> np.ndarray:
    """Every group's bind normals in `Scene.merged` order; +Y for a group with none."""
    out = np.zeros((scene.n_vertices, 3), dtype=np.float64)
    out[:, 1] = 1.0
    for i, g in enumerate(scene.groups):
        if g.normals is None or not g.n_vertices:
            continue
        lo, hi = scene.group_range(i)
        n = np.asarray(g.normals, dtype=np.float64)
        length = np.linalg.norm(n, axis=1, keepdims=True)
        out[lo:hi] = np.divide(n, length, out=np.zeros_like(n), where=length > 1e-12)
    return out


def _static_attributes(scene, total: int) -> np.ndarray:
    """``(v, 4)`` float32: u, v, group index, dominant joint."""
    out = np.zeros((total, 4), dtype="f4")
    out[:, 3] = -1.0
    for i, g in enumerate(scene.groups):
        if not g.n_vertices:
            continue
        lo, hi = scene.group_range(i)
        if g.uvs is not None:
            # ⚠️ NO v-flip. The decoded TMH array is top-down and is uploaded row for
            # row, so GL's t=0 IS the image's top row — the same origin the game's own
            # `v` uses. `blender_mhfu/importer.py` needs `1 - v` only because it ALSO
            # flips the pixel rows for Blender's bottom-up images; the two cancel.
            out[lo:hi, :2] = np.asarray(g.uvs, dtype="f4")
        out[lo:hi, 2] = float(i)
        out[lo:hi, 3] = g.skin.dominant().astype("f4")
    return out


def _upload_textures(ctx, scene) -> Dict[int, object]:
    """``{position in scene.textures: GL texture}``, decoded RGBA8 straight up."""
    out = {}
    for i, t in enumerate(scene.textures):
        tex = ctx.texture((t.width, t.height), 4,
                          np.ascontiguousarray(t.rgba, dtype=np.uint8).tobytes())
        tex.build_mipmaps()
        tex.repeat_x = tex.repeat_y = True      # UVs are in [0,1] but not exclusively
        tex.anisotropy = 4.0
        out[i] = tex
    return out


def _index_batches(ctx, scene) -> Dict[Optional[int], object]:
    """One index buffer per resolved texture (``None`` = the untextured groups)."""
    by_tex: Dict[Optional[int], List[np.ndarray]] = {}
    for i, g in enumerate(scene.groups):
        if not g.n_faces:
            continue
        lo, _hi = scene.group_range(i)
        by_tex.setdefault(g.texture, []).append(
            np.asarray(g.triangles, dtype=np.int64) + lo)
    out = {}
    for tex, parts in by_tex.items():
        idx = np.concatenate(parts).astype("i4").reshape(-1)
        out[tex] = ctx.buffer(idx.tobytes())
    return out


class _BoneTags:
    """An ``(n_bones x 1)`` R8 texture: 1 where the joint is tagged.

    A uniform array would cap the highlight set at whatever length was compiled in, and
    a bone index is exactly a texture lookup. `nearest` filtering on both axes, because
    the shader samples texel centres and any interpolation would bleed a tag onto its
    neighbour.
    """

    def __init__(self, ctx, n_bones: int) -> None:
        self.n = max(int(n_bones), 1)
        self.joints: Tuple[int, ...] = ()
        self.tex = ctx.texture((self.n, 1), 1, np.zeros(self.n, dtype=np.uint8).tobytes())
        self.tex.filter = (ctx.NEAREST, ctx.NEAREST)
        self.tex.repeat_x = self.tex.repeat_y = False

    def set(self, joints) -> None:
        js = tuple(sorted({int(j) for j in (joints or ()) if 0 <= int(j) < self.n}))
        if js == self.joints:
            return
        self.joints = js
        buf = np.zeros(self.n, dtype=np.uint8)
        if js:
            buf[list(js)] = 255
        self.tex.write(buf.tobytes())

    def use(self, unit: int) -> None:
        self.tex.use(unit)

    def release(self) -> None:
        self.tex.release()
