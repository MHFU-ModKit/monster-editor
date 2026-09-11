-- Offline harness for mhfu_port.lua's ATTACK-side writer (issue #33): the fake
-- `mhfu` over a sparse byte map seeded with the REAL em75 bytes — the volume
-- set-pointer table, set 0 and set 2 (the Tigrex charge), and the 0x18 attack
-- record array — then the library, a generated hit module that authors set 2
-- and two record levers, one live in-area port, and a tick. The bytes are then
-- checked record by record, and a second export whose `cap` does not match the
-- live set is shown to be REFUSED with nothing written. Driven by
-- tests/test_runtime.py:
--
--   lua lua_attack_harness.lua <dir with atk_ptrs.bin atk_set0.bin atk_set2.bin
--                               atk_recs.bin zinogre_hit.lua bad_hit.lua> <mhfu_port.lua>
local SCR, PORT_LUA = arg[1], arg[2]
local F = dofile((PORT_LUA:match("^(.*)/framework/") or ".")
                 .. "/mhfu_monster_editor/tests/lua_fake_mhfu.lua")
local logs, load_blob = F.logs, F.load_blob

-- em75's tables, at the overlay's own addresses (hitbox.py: handle 0x09D61250)
local PTRS, RECS = 0x09D60768, 0x09D60848
load_blob(SCR .. "/atk_ptrs.bin", PTRS)              -- 56 x u32 -> set VAs
local SET0, SET2 = mhfu.read_u32(PTRS), mhfu.read_u32(PTRS + 2 * 4)
load_blob(SCR .. "/atk_set0.bin", SET0)              -- 5 records + sentinel
load_blob(SCR .. "/atk_set2.bin", SET2)              -- 10 records + sentinel
load_blob(SCR .. "/atk_recs.bin", RECS)              -- 107 x 0x18
-- the seed is what hitbox.py says it is
assert(mhfu.read_u16(SET2) == 10 and math.abs(mhfu.read_f32(SET2 + 0xC) - 150.0) < 1e-3, "set2[0]")
assert(mhfu.read_u16(SET2 + 10 * 0x28) == 0xFFFF, "set2 sentinel at 10")
assert(mhfu.read_u16(SET0) == 35, "set0[0] bone 35")
assert(mhfu.read_u8(RECS + 6 * 0x18 + 0x02) == 64, "record 6 power 64")
assert(mhfu.read_u8(RECS + 6 * 0x18 + 0x0A) == 2, "record 6 -> set 2")
assert(mhfu.read_u8(RECS + 31 * 0x18 + 0x02) == 30, "record 31 power 30")

dofile(PORT_LUA)
dofile(SCR .. "/zinogre_hit.lua")
local P = mhfu.port
local zin = P.define{ name = "zinogre", species = 75, replace = {77} }
zin.ent = 0x090C1B00
mhfu_tick()   -- drains the queued hit module, then applies on the live port
mhfu_tick()   -- second tick: intact, no re-apply

-- set 2: three of ours, then the sentinel, then the ORIGINAL record 4 untouched
assert(mhfu.read_u16(SET2) == 12, "set2[0] bone")
assert(mhfu.read_u16(SET2 + 2) == 0 and mhfu.read_u16(SET2 + 4) == 0 and mhfu.read_u16(SET2 + 6) == 0,
       "set2[0] shape/row/part zero")
assert(math.abs(mhfu.read_f32(SET2 + 0xC) - 170.0) < 1e-3, "set2[0] radius")
assert(math.abs(mhfu.read_f32(SET2 + 0x14) - 20.0) < 1e-3, "set2[0] ay")
assert(mhfu.read_u16(SET2 + 0x28) == 125 and mhfu.read_u16(SET2 + 0x2A) == 1, "set2[1] joiner capsule")
assert(mhfu.read_u16(SET2 + 0x50) == 44 and mhfu.read_u16(SET2 + 0x52) == 1, "set2[2] capsule on 44")
assert(math.abs(mhfu.read_f32(SET2 + 0x50 + 0x24) - (-175.0)) < 1e-3, "set2[2] bz")
assert(mhfu.read_u16(SET2 + 3 * 0x28) == 0xFFFF and mhfu.read_u16(SET2 + 3 * 0x28 + 6) == 0xFFFF, "sentinel")
assert(mhfu.read_u32(SET2 + 3 * 0x28 + 0x0C) == 0, "sentinel zeroed")
assert(mhfu.read_u16(SET2 + 4 * 0x28) == 2 and math.abs(mhfu.read_f32(SET2 + 4 * 0x28 + 0xC) - 250.0) < 1e-3,
       "original record 4 untouched")
-- set 0 was not authored: byte for byte the host's
assert(mhfu.read_u16(SET0) == 35 and math.abs(mhfu.read_f32(SET0 + 0xC) - 180.0) < 1e-3, "set0 untouched")
assert(mhfu.read_u16(SET0 + 5 * 0x28) == 0xFFFF, "set0 sentinel where it was")
-- record 6: power written, element and volume NOT (nil = the host's byte stands)
assert(mhfu.read_u8(RECS + 6 * 0x18 + 0x02) == 40, "record 6 power")
assert(mhfu.read_u8(RECS + 6 * 0x18 + 0x09) == 0x21, "record 6 element untouched")
assert(mhfu.read_u8(RECS + 6 * 0x18 + 0x0A) == 2, "record 6 volume untouched")
assert(mhfu.read_u8(RECS + 6 * 0x18 + 0x01) == 99, "record 6 +0x01 untouched")
-- record 31: only the volume re-pointed
assert(mhfu.read_u8(RECS + 31 * 0x18 + 0x0A) == 5, "record 31 volume")
assert(mhfu.read_u8(RECS + 31 * 0x18 + 0x02) == 30, "record 31 power untouched")
-- record 7 not named: untouched
assert(mhfu.read_u8(RECS + 7 * 0x18 + 0x02) == 40 and mhfu.read_u8(RECS + 7 * 0x18 + 0x0A) == 3, "record 7")
assert(zin._atk_caps and zin._atk_caps[2] == 10, "cap " .. tostring(zin._atk_caps and zin._atk_caps[2]))
local applied = 0
for _, l in ipairs(logs) do
  if l:find("HIT TABLES APPLIED") then
    applied = applied + 1
    assert(l:find("1 attack set%(s%)/3 volume%(s%)") and l:find("2 attack record%(s%)"), l)
  end
end
assert(applied == 1, "applied once, then found intact: " .. applied)
-- a change under us is noticed and re-applied: the set AND a lever
mhfu.write_f32(SET2 + 0xC, 97.0)
mhfu_tick()
assert(math.abs(mhfu.read_f32(SET2 + 0xC) - 170.0) < 1e-3, "set re-applied")
mhfu.write_u8(RECS + 6 * 0x18 + 0x02, 64)
mhfu_tick()
assert(mhfu.read_u8(RECS + 6 * 0x18 + 0x02) == 40, "lever re-applied")

-- THE NEGATIVE: an export whose `cap` for set 0 is not the live count. Nothing
-- may be written — not set 0, and not the good set either, since the whole apply
-- refuses before the first record.
local before2 = mhfu.read_u16(SET2)
mhfu.write_u16(SET2, 12)   -- (already 12; the point is the value below)
dofile(SCR .. "/bad_hit.lua")
local n_logs = #logs
mhfu_tick()
assert(mhfu.read_u16(SET0) == 35 and math.abs(mhfu.read_f32(SET0 + 0xC) - 180.0) < 1e-3,
       "set 0 must not be written under a cap mismatch")
assert(mhfu.read_u16(SET0 + 5 * 0x28) == 0xFFFF, "set 0 sentinel must not move")
local refused = false
for i = n_logs + 1, #logs do
  if logs[i]:find("NOT written") and logs[i]:find("expected 99") then refused = true end
end
assert(refused, "the mismatch was not logged as a refusal")
assert(mhfu.read_u8(RECS + 9 * 0x18 + 0x02) == 70, "record 9 must not be written under a refused apply")
print("HARNESS OK")
