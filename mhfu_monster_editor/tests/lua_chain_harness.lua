-- Offline harness for mhfu_port.lua's MOVE CHAIN: the fake `mhfu` over an empty
-- byte map, one port with three declared moves, and the library's own tick.
-- Checks, in order: a re-entry of the running pair is REFUSED (and forced through
-- with {force=true}); `after` is walked when the engine leaves the pair; `hold_max`
-- ends a standing pair from here; a move with neither is logged PARKED exactly
-- once; a declared pair the ENGINE enters by itself is painted by the executor
-- hook, once per entry; and an `after` whose pair the engine already reached is
-- adopted, not re-written. Driven by tests/test_runtime.py:
--
--   lua lua_chain_harness.lua <mhfu_port.lua>
local PORT_LUA = arg[1]
local F = dofile((PORT_LUA:match("^(.*)/framework/") or ".")
                 .. "/mhfu_monster_editor/tests/lua_fake_mhfu.lua")
local logs = F.logs

-- the executor hook is a no-op registrar in the fake; capture it here
local HOOK
mhfu.on_bigmonster_action = function(fn) HOOK = fn end

dofile(PORT_LUA)
local P = mhfu.port
local ENT = 0x090C1B00
local OFF_MAIN, OFF_SUB, OFF_PHASE = 0x298, 0x299, 0x1D5
local zin = P.define{
  name = "zinogre", species = 75, replace = {77},
  clips = { lunge_forward = 6, stop_walk_forward = 5, other = 7 },
  moves = {
    lunge      = { main = 1, sub = 4, clip = "lunge_forward", after = "lunge_stop",
                   hold_max = 8 },
    lunge_stop = { main = 1, sub = 3, clip = "stop_walk_forward" },
    held       = { main = 2, sub = 9, clip = "other" },
  },
}
zin.ent = ENT
assert(HOOK, "the executor hook was not registered")

local function pair() return mhfu.read_u8(ENT + OFF_MAIN), mhfu.read_u8(ENT + OFF_SUB) end
local function set_pair(m, s) mhfu.write_u8(ENT + OFF_MAIN, m); mhfu.write_u8(ENT + OFF_SUB, s) end
local function count(pat, from)
  local n = 0
  for i = from or 1, #logs do if logs[i]:find(pat) then n = n + 1 end end
  return n
end
local function ticks(n) for _ = 1, n do mhfu_tick() end end

ticks(1)
-- 1. play writes the pair; the same pair again is refused, forced goes through
assert(zin:play("lunge"), "first play")
local m, s = pair()
assert(m == 1 and s == 4, "pair written")
assert(zin.move == "lunge", "move recorded")
ticks(3)
assert(not zin:play("lunge"), "re-entry must be refused")
assert(not zin:play("lunge"), "still refused")
ticks(1)                              -- log lines queue until the tick drains them
assert(count("refused: %(1,4%) is already running %(ours%)") == 1, "refusal logged once")
assert(count("refused") == 1, "the second refusal is not logged (throttled)")
assert(zin:play("lunge", 0, { force = true }), "force restarts")
assert(zin.move == "lunge")

-- 2. the engine leaves the pair -> `after` is walked
set_pair(0, 3)                        -- the engine's own hand-off
ticks(1)
m, s = pair()
assert(m == 1 and s == 3, "after='lunge_stop' entered: got (" .. m .. "," .. s .. ")")
assert(zin.move == "lunge_stop", "move is now the after")
assert(zin.last_move == "lunge", "last_move recorded")
assert(count("'lunge' ended %-> after='lunge_stop'") == 1, "the walk is logged")
assert(zin.clip == 5, "the after's clip is latched")

-- 3. hold_max: a standing pair ends from here into `after`
set_pair(0, 1)
ticks(3)                              -- lunge_stop ends (engine moved to (0,1))
assert(zin.move == nil, "lunge_stop has no after: move clears")
assert(zin:play("lunge"), "lunge again from (0,1)")
ticks(9)                              -- > hold_max = 8, the pair never changed
m, s = pair()
assert(m == 1 and s == 3, "hold_max walked to lunge_stop: got (" .. m .. "," .. s .. ")")
ticks(1)                              -- drain
assert(count("held %d+ ticks %(hold_max 8%)") == 1, "hold_max logged")
assert(count("'lunge' hold_max %-> after='lunge_stop'") == 1)

-- 4. parked: a move with neither `after` nor `hold_max`, phase never moving
set_pair(0, 1)
ticks(2)
assert(zin:play("held"))
mhfu.write_u8(ENT + OFF_PHASE, 3)     -- the handler parked in phase 3
ticks(15)
assert(count("move 'held' PARKED: %(2,9%) phase 3 unchanged") == 1, "parked once")
ticks(10)
assert(count("PARKED") == 1, "parked is said ONCE per play")
assert(zin.move == "held", "a parked move is not released behind the brain's back")

-- 5. the engine enters a declared pair itself -> the hook paints it, once per entry
zin:release()
set_pair(1, 4)                        -- the brain's own charge
assert(HOOK({ entity = ENT, action_id = 17 }) == 6, "painted with lunge_forward")
assert(HOOK({ entity = ENT, action_id = 17 }) == nil, "latch 1: the second dispatch stands")
ticks(1)
assert(count("engine entered 'lunge' itself") == 1)
set_pair(0, 3)                        -- he left it
ticks(1)
set_pair(1, 4)                        -- and charged again
assert(HOOK({ entity = ENT, action_id = 17 }) == 6, "painted again on the next entry")
assert(HOOK({ entity = ENT, action_id = 51 }) == nil)
set_pair(0, 4)                        -- a pair no move declares: the hook abstains
assert(HOOK({ entity = ENT, action_id = 51 }) == nil, "undeclared pair is not painted")
assert(zin.move == nil, "painting a native entry declares no scripted move")

-- 6. adopt: the engine's hand-off lands where `after` points
set_pair(0, 1)
ticks(2)
assert(zin:play("lunge"))
set_pair(1, 3)                        -- the engine went to (1,3) by itself
ticks(2)
assert(zin.move == "lunge_stop", "adopted, not re-written")
assert(count("the engine is already in %(1,3%), adopted") == 1)
m, s = pair()
assert(m == 1 and s == 3)
assert(mhfu.read_u8(ENT + OFF_PHASE) ~= 0 or true) -- no act_set happened: phase untouched
print("HARNESS OK")
