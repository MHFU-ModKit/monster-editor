#!/usr/bin/env python3
"""Offline oracle: does a ported monster keep its AUTHENTIC per-vertex skin?

Replays the porter's skinning stage on an MHP3rd source, encodes a native MHFU
monster PMO, decodes it back with the independent reader, and compares each
vertex's (bone, weight) influences against what the source declared.

This is the check that does NOT need the emulator: if the bone sets match and the
only weight error is the u8 quantisation step (1/128), the skin that ships is the
skin the monster was authored with.

    python tools/skin_fidelity.py \
        --model workspace/extracted_mhp3/data_files/file_05248.bin \
        --geo   workspace/extracted_mhp3/data_files/file_05249.bin \
        --frame workspace/extracted/data_files/file_06185.bin --source-skeleton
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhfu_model import pmo_p3rd as P3
from mhfu_model import pmo_skin as SK
from mhfu_model import skeleton_p3rd as SKP
from mhfu_model.bone_match import bind_world_positions

QUANT = 1.0 / 0x80          # the encoder stores weights as round(w * 0x80) in a u8


def _subs(blob):
    n = struct.unpack_from("<I", blob, 0)[0]
    return [struct.unpack_from("<II", blob, 4 + i * 8) for i in range(n)]


def _main_pmo(blob):
    best = None
    for o, s in _subs(blob):
        if blob[o:o + 4] != b"pmo\x00":
            continue
        h = struct.unpack_from("<I4f2H8I", blob, o + 8)
        if best is None or h[5] > best[2]:
            best = (o, s, h[5])
    return best


def _find_sub(blob, magic):
    for o, s in _subs(blob):
        if blob[o:o + 4] == magic:
            return o, s
    return None


def _lead_origin(world, eps=1.0):
    c = 0
    for w in world:
        if (w[0] ** 2 + w[1] ** 2 + w[2] ** 2) ** 0.5 < eps:
            c += 1
        else:
            break
    return c


def lead_pad_for(model_pac, frame_pac):
    """The porter's leading-origin pad (host lead chain - source lead chain)."""
    ss = _find_sub(model_pac, b"\x00\x00\x00\x80")
    if ss is None or frame_pac is None:
        return 0
    ssk = SKP.parse(model_pac[ss[0]:ss[0] + ss[1]])
    bw = bind_world_positions([b.parent for b in ssk.bones],
                              [tuple(b.bind_pos) for b in ssk.bones])
    _p, _l, hbw = SK.frame_skeleton(frame_pac)
    return max(0, _lead_origin(hbw) - _lead_origin(bw))


def analyse(model_pac, geo, frame_pac=None, source_skeleton=False):
    mp = _main_pmo(model_pac)
    if mp is None:
        raise SystemExit("no PMO sub in the model PAC")
    o, s, _ = mp
    model = P3.parse(model_pac[o:o + s], geo_blob=geo)
    if not model.mesh_groups:
        raise SystemExit("geometry parsed empty — pass the --geo companion file")

    pad = lead_pad_for(model_pac, frame_pac) if source_skeleton else 0
    remap = (lambda b: b + pad) if pad else None

    r = {"groups": len(model.mesh_groups),
         "verts": sum(len(g.vertices) for g in model.mesh_groups),
         "lead_pad": pad}

    # --- what the SOURCE declares, after the same remap the porter applies -------
    want = []
    r["verts_with_weights"] = 0
    r["frac_verts"] = 0
    maxb = 0
    for g in model.mesh_groups:
        gw = []
        for v in g.vertices:
            src = v.get("influences") or []
            acc = {}
            for b, w in src:
                if w == 0 or b is None or b < 0:
                    continue
                ob = b + pad
                acc[ob] = acc.get(ob, 0.0) + w
            if acc:
                r["verts_with_weights"] += 1
                if any(0.0 < w < 1.0 for w in acc.values()):
                    r["frac_verts"] += 1
            else:
                acc = {0: 1.0}
            t = sum(acc.values()) or 1.0
            gw.append({b: w / t for b, w in acc.items()})
            maxb = max(maxb, len(acc))
        want.append(gw)
    r["max_bones_per_vertex"] = maxb
    r["max_bones_per_vgroup"] = max(
        len({b for vi in gw for b in vi}) for gw in want)
    r["vgroups_over_palette"] = sum(
        1 for gw in want if len({b for vi in gw for b in vi}) > 8)

    # --- the porter's skinning stage -> native PMO -> independent decode ---------
    vgs = SK.from_source_influences(model.mesh_groups, bone_remap=remap,
                                    materials_of=lambda g: g.material, max_pal=8)
    texids = sorted({g.material for g in model.mesh_groups})
    t2i = {t: i for i, t in enumerate(texids)}
    for vg, g in zip(vgs, model.mesh_groups):
        vg.material = t2i[g.material]
    pmo = SK.build(model.scale, vgs, [{"texID": t} for t in texids])
    r["pmo_bytes"] = len(pmo)
    back = SK.read(pmo)
    r["decoded_vgroups"] = len(back.vgroups)

    # --- compare -----------------------------------------------------------------
    bad_set = bad_count = compared = 0
    worst = 0.0
    total_err = 0.0
    sums = []
    for gi, (gw, dvg) in enumerate(zip(want, back.vgroups)):
        if len(dvg.influences) != len(gw):
            bad_count += 1
            continue
        for wi, di in zip(gw, dvg.influences):
            compared += 1
            got = {}
            for b, w in di:
                if w:
                    got[b] = got.get(b, 0.0) + w
            sums.append(sum(got.values()))
            if set(got) != set(wi):
                bad_set += 1
                continue
            t = sum(got.values()) or 1.0
            for b, w in wi.items():
                e = abs(w - got[b] / t)
                worst = max(worst, e)
                total_err += e
    r["compared_verts"] = compared
    r["vgroups_wrong_length"] = bad_count
    r["verts_wrong_bone_set"] = bad_set
    r["max_weight_error"] = worst
    r["mean_weight_error"] = (total_err / compared) if compared else 0.0
    r["quant_step"] = QUANT
    r["weight_sum_min"] = min(sums) if sums else 0
    r["weight_sum_max"] = max(sums) if sums else 0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--geo")
    ap.add_argument("--frame", default="workspace/extracted/data_files/file_06185.bin")
    ap.add_argument("--source-skeleton", action="store_true")
    a = ap.parse_args()
    model = open(a.model, "rb").read()
    geo = open(a.geo, "rb").read() if a.geo else None
    frame = open(a.frame, "rb").read() if os.path.exists(a.frame) else None
    r = analyse(model, geo, frame, source_skeleton=a.source_skeleton)
    for k, v in r.items():
        print("  %-24s %s" % (k, v))
    ok = (r["verts_wrong_bone_set"] == 0 and r["vgroups_wrong_length"] == 0
          and r["max_weight_error"] <= QUANT)
    print("\nAUTHENTIC SKIN PRESERVED: %s" % ("YES" if ok else "NO"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
