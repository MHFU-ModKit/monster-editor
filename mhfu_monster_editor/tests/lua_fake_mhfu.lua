-- The fake `mhfu` the offline harnesses run mhfu_port.lua under: a sparse byte
-- map with the framework's read/write/log surface over it, and the event
-- registrars as no-ops. `dofile` it, then seed `mem` with real bytes.
--
-- Lua 5.4+ on the host; the PSP build is 5.4 with 32-bit integers, and every
-- literal here fits that. Returns { mem, logs, load_blob }.
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


return { mem = mem, logs = logs, load_blob = load_blob, rd = rd, wr = wr }
