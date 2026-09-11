-- Offline harness for mhfu_port.lua's NATIVE SEAM path (em_vhook v3): a fake
-- `mhfu.em_*` that behaves like the stubs — a request lands in the cells by
-- the next tick, substitutions and rules are recorded slot by slot — and the
-- library's own tick. Checks, in order: the seam is armed on the first live
-- tick (claims -> em_substitute, rules -> em_rule, unused slots cleared);
-- play() goes through em_request and is confirmed when the cells show the
-- pair; the translator's alternative main is tracked, not treated as "ended";
-- a request the engine declines is reported with the ring and does NOT walk
-- `after`; `{raw=true}` writes the cells by hand; hot-reload re-arms. Driven
-- by tests/test_runtime.py:
--
--   lua lua_native_harness.lua <mhfu_port.lua>
local PORT_LUA = arg[1]
local F = dofile((PORT_LUA:match("^(.*)/framework/") or ".")
                 .. "/mhfu_monster_editor/tests/lua_fake_mhfu.lua")
local logs = F.logs

local ENT = 0x090C1B00
local OFF_MAIN, OFF_SUB, OFF_PHASE = 0x298, 0x299, 0x1D5
local function pair() return mhfu.read_u8(ENT + OFF_MAIN), mhfu.read_u8(ENT + OFF_SUB) end
local function set_pair(m, s) mhfu.write_u8(ENT + OFF_MAIN, m); mhfu.write_u8(ENT + OFF_SUB, s) end

-- ---- the fake seam ---------------------------------------------------------
local SEAM = { installed = false, subs = {}, rules = {}, req = nil, req_done = 0,
               land = true, land_as = nil, ring = {} }
mhfu.EM_ANY, mhfu.EM_UNLIMITED = 0xFE, -1
function mhfu.em_installed() return SEAM.installed end
function mhfu.em_substitute(slot, mask, sub, tm, ts, count)
  SEAM.subs[slot] = { mask = mask, sub = sub, to_main = tm, to_sub = ts, count = count }
  return true
end
function mhfu.em_rule(slot, r) SEAM.rules[slot] = r; return true end
function mhfu.em_clear() SEAM.subs, SEAM.rules = {}, {} end
function mhfu.em_request(m, s, mode)
  SEAM.req = { m, s, mode }
  -- the stub issues it on the next AI frame: long before the next 2 Hz tick
  SEAM.req_done = SEAM.req_done + 1
  if SEAM.land then
    local lm, ls = SEAM.land_as and SEAM.land_as[1] or m, SEAM.land_as and SEAM.land_as[2] or s
    set_pair(lm, ls)
    mhfu.write_u8(ENT + OFF_PHASE, 0)
    SEAM.result = { lm, ls }
  else
    SEAM.result = { pair() }
  end
  table.insert(SEAM.ring, { main = m, sub = s, mode = mode, subst = 0 })
  return true
end
function mhfu.em_status()
  local ring = {}
  for i = math.max(1, #SEAM.ring - 7), #SEAM.ring do ring[#ring + 1] = SEAM.ring[i] end
  local r = SEAM.result or { 0, 0 }
  return { installed = SEAM.installed, req_done = SEAM.req_done, req_main = r[1],
           req_sub = r[2], ring = ring, sub_hits = 0, sub_landed = 0, brain_fires = 0 }
end

local HOOK
mhfu.on_bigmonster_action = function(fn) HOOK = fn end

dofile(PORT_LUA)
local P = mhfu.port
local function define()
  local z = P.define{
    name = "zinogre", species = 75, replace = {77},
    clips = { lunge_forward = 6, dash_forward_stop = 21 },
    moves = {
      lunge      = { main = 1, sub = 4, clip = "lunge_forward", after = "lunge_stop",
                     hold_max = 8, claim = { main = 1 } },
      lunge_stop = { main = 0, sub = 3, clip = "dash_forward_stop" },
    },
  }
  z:rule{ from = "lunge", min_frames = 15, dist = { 250, 1e9 }, receding = true,
          play = "lunge_stop", cooldown = 30 }
  return z
end
local zin = define()
zin.ent = ENT
local function count(pat, from)
  local n = 0
  for i = from or 1, #logs do if logs[i]:find(pat) then n = n + 1 end end
  return n
end
local function ticks(n) for _ = 1, n do mhfu_tick() end end

-- 0. seam not live yet: play() falls back to the byte write, nothing armed
ticks(1)
assert(zin:play("lunge"))
local m, s = pair()
assert(m == 1 and s == 4 and SEAM.req == nil, "fallback wrote the cells by hand")
assert(next(SEAM.subs) == nil, "nothing armed while the seam is down")
set_pair(0, 1); ticks(1)                      -- it ends; `after` writes (0,3) by hand
m, s = pair(); assert(m == 0 and s == 3 and zin.move == "lunge_stop", "fallback walked after")
set_pair(0, 1); ticks(2)                      -- the skid ends; nothing follows it
assert(zin.move == nil)

-- 1. the seam comes up: the next tick arms claims and rules
SEAM.installed = true
ticks(1)
assert(count("native seam live") == 1, "arming logged once")
assert(SEAM.subs[0] and SEAM.subs[0].mask == 0x02 and SEAM.subs[0].sub == 0xFE
       and SEAM.subs[0].to_main == 1 and SEAM.subs[0].to_sub == 4
       and SEAM.subs[0].count == -1, "claim main 1 -> (1,4) standing in slot 0")
for k = 1, 3 do assert(SEAM.subs[k] and SEAM.subs[k].count == 0, "slot " .. k .. " cleared") end
local r = SEAM.rules[0]
assert(r and r.from_mask == 0x02 and r.from_sub == 4 and r.to_main == 0 and r.to_sub == 3
       and r.min_frames == 15 and r.receding == true and r.closing == false
       and r.dist_lo == 250 and r.cooldown == 30, "rule 0 installed as declared")
for k = 1, 3 do assert(SEAM.rules[k] == nil, "rule slot " .. k .. " cleared") end
assert(count("claim: host enter%-actions with main in 0x02 %-> 'lunge'") == 1)
assert(count("rule 1: main 0x02 sub 4 >=15 frames d%[250,inf%) receding %-> 'lunge_stop'") == 1)

-- 2. play() through the seam: request, then confirmed on the next tick
local t2 = #logs
assert(zin:play("lunge"))
assert(SEAM.req and SEAM.req[1] == 1 and SEAM.req[2] == 4 and SEAM.req[3] == 0, "em_request(1,4,0)")
assert(zin.move == "lunge" and zin.clip == 6, "move and clip latched before the request")
ticks(1)
assert(count("'lunge' entered natively %(1,4%), provisioned") == 1)
assert(zin._req == nil and zin.move == "lunge")
-- the engine's own hand-off (the charge's budget ran out) -> after is adopted
set_pair(0, 3)
ticks(1)
assert(count("move 'lunge' %(1,4%) ended after", t2) == 1)
assert(zin.move == "lunge_stop" and count("adopted", t2) == 1, "after adopted, not re-requested")
assert(SEAM.req[1] == 1, "no second request was made for the skid")
set_pair(0, 1); ticks(2)
assert(zin.move == nil)

-- 3. the translator routes id 4 to its other main: tracked, not "ended"
SEAM.land_as = { 2, 4 }
assert(zin:play("lunge"))
ticks(1)
assert(count("entered natively as %(2,4%) — the translator's main for id 4") == 1)
assert(zin.move == "lunge" and zin._entered[1] == 2, "tracking (2,4)")
ticks(3)
assert(zin.move == "lunge", "a tracked alternative pair is not treated as ended")
set_pair(0, 3); ticks(1)
assert(count("move 'lunge' %(2,4%) ended after") == 1)
SEAM.land_as = nil
set_pair(0, 1); ticks(3)
assert(zin.move == nil)

-- 3b. landed and already OVER within the tick (a charge the 30 Hz rule ended in
--     0.5 s): result-after-call is our pair, the cells are past it -> not a
--     decline; the ended path adopts `after` where the engine stands
local t3b = #logs
assert(zin:play("lunge"))
set_pair(0, 3)                        -- ... and the rule handed him to the skid
ticks(1)
assert(count("'lunge' entered natively %(1,4%) and was over within the tick %(now %(0,3%)%)", t3b) == 1)
assert(count("was requested and issued but", t3b) == 0, "not read as a decline")
assert(zin.move == "lunge_stop" and count("adopted", t3b) == 1, "after adopted from the skid")
set_pair(0, 1); ticks(3)
assert(zin.move == nil)

-- 4. the engine declines the request: reported with the ring, `after` NOT walked
SEAM.land = false
local before = #logs
assert(zin:play("lunge"))
ticks(2)
assert(count("was requested and issued but the cells read %(0,1%)", before) == 1,
       "declined request reported")
assert(count("last enter%-actions: .*%(1,4,m0%)", before) == 1, "the ring is in the line")
assert(zin.move == nil and zin.clip == nil, "the move never started")
assert(count("after='lunge_stop'", before) == 0, "no after walk for a move that never ran")
m, s = pair(); assert(m == 0 and s == 1, "cells untouched")
SEAM.land = true

-- 5. {raw=true} writes the cells by hand even with the seam live
local reqs = SEAM.req_done
assert(zin:play("lunge", 0, { raw = true }))
m, s = pair()
assert(m == 1 and s == 4 and SEAM.req_done == reqs, "raw: no request, cells written")
ticks(1)
assert(zin.move == "lunge")
set_pair(0, 1); ticks(2)

-- 6. a redefine (hot reload) re-arms, and the seam-live line is not repeated
SEAM.subs, SEAM.rules = {}, {}
zin = define(); zin.ent = ENT
ticks(1)
assert(SEAM.subs[0] and SEAM.subs[0].to_sub == 4 and SEAM.rules[0], "re-armed after redefine")
assert(count("native seam live") == 1, "the seam-live line is said once per boot")

-- 7. the executor hook still paints a substituted (engine-entered) charge
set_pair(1, 4)
assert(HOOK({ entity = ENT, action_id = 17 }) == 6, "a claimed pair the engine enters is painted")
print("HARNESS OK")
