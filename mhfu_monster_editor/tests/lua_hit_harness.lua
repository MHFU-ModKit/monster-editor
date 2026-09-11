-- offline harness: fake `mhfu` over a sparse byte map, run mhfu_port.lua + the
-- generated hit module, tick once with a live in-area port, check the bytes.
-- Offline harness for mhfu_port.lua's hit-table writer (issue #19): a fake `mhfu`
-- over a sparse byte map seeded with the REAL em75 set 0 and Tigrex grid bytes,
-- the library, a generated hit module, one live in-area port, and a tick. Then
-- the bytes are checked record by record. Driven by tests/test_runtime.py:
--
--   lua lua_hit_harness.lua <dir with set0.bin grid.bin zinogre_hit.lua> <mhfu_port.lua>
--
-- Lua 5.4+ on the host; the PSP build is 5.4 with 32-bit integers, and every
-- literal here fits that.
local SCR, PORT_LUA = arg[1], arg[2]
local mem = {}
local function rd(a) return mem[a] or 0 end
local function wr(a, b) mem[a] = b & 0xFF end
local function load_blob(path, at)
  local f = assert(io.open(path, "rb")); local s = f:read("a"); f:close()
  for i = 1, #s do wr(at + i - 1, s:byte(i)) end
end
mhfu = {}
function mhfu.read_u8(a) return rd(a) end
function mhfu.read_u16(a) return rd(a) | (rd(a+1) << 8) end
function mhfu.read_u32(a) return rd(a) | (rd(a+1) << 8) | (rd(a+2) << 16) | (rd(a+3) << 24) end
function mhfu.write_u8(a, v) wr(a, v) end
function mhfu.write_u16(a, v) wr(a, v); wr(a+1, v >> 8) end
function mhfu.write_u32(a, v) for i = 0, 3 do wr(a+i, (v >> (8*i)) & 0xFF) end end
function mhfu.read_f32(a) return (string.unpack("<f", string.char(rd(a), rd(a+1), rd(a+2), rd(a+3)))) end
function mhfu.write_f32(a, v) local s = string.pack("<f", v); for i = 1, 4 do wr(a+i-1, s:byte(i)) end end
function mhfu.mem_valid(a) return a > 0x08000000 and a < 0x0C000000 end
local logs = {}
function mhfu.log(s) logs[#logs+1] = s; if os.getenv("HARNESS_VERBOSE") then print("LOG " .. s) end end
function mhfu.paint_map() end
function mhfu.get_screen_state() return 17 end
function mhfu.get_area_index() return 99 end
function mhfu.entities_of_type() return {} end
function mhfu.entity_alive(e) return e ~= 0 end
for _, n in ipairs{"on_quest_targets_building","on_bigmonster_spawn","on_bigmonster_death",
                   "on_bigmonster_action","on_bigmonster_damaged"} do mhfu[n] = function() end end
mhfu.MON_TIGREX = 75

-- species row 75: +0x240 -> set0, +0x2FC -> state table
local ROW = 0x09BB87C0 + 75 * 0x1D0
mhfu.write_u32(ROW + 0x240, 0x09D58CD0)
mhfu.write_u32(ROW + 0x2FC, 0x09BC6828)
load_blob(SCR .. "/set0.bin", 0x09D58CD0)          -- 42 records + sentinel
load_blob(SCR .. "/grid.bin", 0x09BC6798)          -- 2 blocks + the pointer table
assert(mhfu.read_u32(0x09BC6828) == 0x09BC6798 and mhfu.read_u32(0x09BC682C) == 0x09BC67E0)
assert(mhfu.read_u16(0x09D58CD0) == 40 and mhfu.read_u16(0x09D58CD0 + 42*0x28) == 0xFFFF)

dofile(PORT_LUA)
dofile(SCR .. "/zinogre_hit.lua")
local P = mhfu.port
local zin = P.define{ name = "zinogre", species = 75, replace = {77} }
zin.ent = 0x090C1B00
mhfu_tick()   -- drains the queued hit module, then applies on the live port
mhfu_tick()   -- second tick: intact, no re-apply

-- record 0 = ours, record 1 = sentinel, record 2 untouched original (bone 21)
assert(mhfu.read_u16(0x09D58CD0) == 2, "bone")
assert(mhfu.read_u16(0x09D58CD0+4) == 2 and mhfu.read_u16(0x09D58CD0+6) == 1, "row/part")
assert(math.abs(mhfu.read_f32(0x09D58CD0+0xC) - 582.0) < 1e-3, "radius")
assert(math.abs(mhfu.read_f32(0x09D58CD0+0x14) - (-30.0)) < 1e-3, "ay")
assert(mhfu.read_u16(0x09D58CD0+0x28) == 0xFFFF and mhfu.read_u16(0x09D58CD0+0x2E) == 0xFFFF, "sentinel")
assert(mhfu.read_u32(0x09D58CD0+0x28+0x0C) == 0, "sentinel zeroed")
assert(mhfu.read_u16(0x09D58CD0+0x50) == 21, "record 2 untouched")
-- grid: state 0 row 0 cut=255, raw still 100; state 1 row 6 shot 255; pad intact
assert(mhfu.read_u8(0x09BC6798+1) == 255 and mhfu.read_u8(0x09BC6798) == 100, "grid0")
assert(mhfu.read_u8(0x09BC67E0+63) == 255 and mhfu.read_u8(0x09BC67E0+70) == 0, "grid1/pad")
local applied, intact = 0, 0
for _, l in ipairs(logs) do
  if l:find("HIT TABLES APPLIED") then applied = applied + 1 end
end
assert(applied == 1, "applied once, then found intact: " .. applied)
assert(zin._hit_cap == 42, "cap " .. tostring(zin._hit_cap))
-- a change under us is noticed and re-applied
mhfu.write_f32(0x09D58CD0+0xC, 97.0)
mhfu_tick()
assert(math.abs(mhfu.read_f32(0x09D58CD0+0xC) - 582.0) < 1e-3, "re-applied")
print("HARNESS OK")
