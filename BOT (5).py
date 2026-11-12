"""
StrategyBot - Zone Rush bot Requiem (formerly Ultimate / v14) (v13 retuned: CONTEST_FACTOR .7, VORONOI_TIE .9, RAID_UP_MULT .15) (standard library only).

Lineage: v6 (3-target chain planner + Bayesian per-rival shield model + cooldown-aware raids)
         -> v10 (opponent-adaptive upgrade style) -> v13.

v13 = v10 plus four changes, each ablated on identical seeds against v10:
  1. OWN_CONTEST_FIX  - rule 2.3: the owner wins an entry contest for its own empty cell, so
                        token order must not discount our own-cell upgrades.   (rule-correct, ~0 effect)
  2. MULTI_CATCH      - one shield freezes every rival on our land: shield when >=2 rivals stand on
                        our land and one has progress >=1 (only in the last 25 turns). (~0 effect)
  3. Score ledger     - every player's cumulative score is reconstructed from board deltas
                        (income each turn + 8*value*level on each owner change); verified exact.
  4. RANK_RAID        - from turn 20, raids on rivals whose projected final score is ABOVE ours get
                        x1.6 weight (projected = ledger + turns_left * current income).
                        +0.007 pairwise win fraction on average, positive on every opponent group
                        in three independent seed sets.
Off by default (measured no win-fraction gain, costs raw points): LOCK_IN, CATCHUP.  They are
kept as switches for a Round-2 (first-place-only) policy that I could not test properly.

Idea: every turn, score every cell we could work on by
    (points it will earn us) / (turns of effort it costs)
and walk towards the best one.  Targets are:
    - neutral cells         (1 stay)
    - our own cells         (3 stays -> upgrade, doubles income)
    - rival cells           (raid: 8 x value x level bonus + income swing)
On top of that we SHIELD when a rival is about to take one of our cells.

The skeleton gives no progress counters, so we track our own progress and
the rivals' (how long each has stood on the same cell) in module memory.
"""
from collections import deque

SIZE = 8
TURNS = 80
MOVES = {"NORTH": (0, -1), "EAST": (1, 0), "SOUTH": (0, 1), "WEST": (-1, 0)}

# ---- tunables -------------------------------------------------------------
RAID_L1_FACTOR = 0.6       # raiding a level-1 cell: tuned in simulation (shields, contests)
RAID_UP_FACTOR = 0.2       # raiding an upgraded cell: defender can hold it by shielding
DENY_WEIGHT = 0.0          # value of reducing a rival's income (0 tested best)
CONTEST_FACTOR = 0.7       # we may lose a contested cell on priority
SHIELD_L1_MIN_VALUE = 4    # shield a level-1 cell only if its base value >= this (4 = never)
# ---- adaptive style (v10) ---------------------------------------------------
UPG_MID = 1.55             # mild upgrade lean in the mid-game (all opponents)
MID_LO = 10
MID_HI = 60
UPG_FLAG = 1.65            # strong upgrade lean once an upgrade-first / slow-expanding rival is seen
FLAG_T0 = 4                # detection window: rival owns <= FLAG_MAXC cells yet has >= FLAG_MINU upgraded
FLAG_T1 = 7
FLAG_MAXC = 3
FLAG_MINU = 1
FLAG_END = 80              # flag boost applies until this turn
UPG_WEIGHT = 1.35          # multiplier on the value of upgrading our own cells
NEUTRAL_WEIGHT = 1.0       # multiplier on the value of capturing neutral cells
VAL_EXP = 1.0              # >1 prefers high-value cells more strongly
SHIELD_UP_PROG = 2         # shield an upgraded cell once the invader has this many stays
RAID_SAFE = 0.0            # raid factor when the owner provably cannot shield in time (0 = feature off)
STICKY = 1.3               # bonus for keeping the previous target (stops dithering)

VORONOI_LOSS = 1.0        # factor when a rival reaches the target strictly before us
VORONOI_TIE = 0.9          # factor when a rival ties our arrival time but wins priority
FOLLOW_W = 0             # weight of best follow-up target after the chosen one
SHIELD_PRIOR = (1.0, 2.0)  # Beta prior (a, b) on P(rival shields while we raid it)
CHAIN = 3                  # plan over chains of this many targets (1 = greedy)
RAID_SAFE_F = 1.0          # raid factor when owner is provably on cooldown
RAID_BONUS_W = 1.0         # weight of the one-off 8*v*L raid bonus
RAID_UP_MULT = 0.15         # extra discount when the raid target is upgraded
TOPN = 9                   # candidates fed to the chain planner
# ---- v13 flags (each ablated against v10 on identical seeds) ---------------
OWN_CONTEST_FIX = 1 # rule: owner wins an entry contest for its own cell
MULTI_CATCH = 1 # shield when >=2 rivals stand on our land and one is progressing
MULTI_THREAT_R = 25        # ... only while at most this many turns remain
SEAT_HOME = [0, 7, 56, 63]  # seat 1 TL, 2 TR, 3 BL, 4 BR
LOCK_IN = 0 # v12 rank-aware endgame: protect a big projected lead
RANK_MODE_START = 70
LOCK_IN_MARGIN = 80
CATCHUP = 0 # v12: focus late raids on the rival just ahead
CATCHUP_MARGIN = 110
PAIRWISE_RAID_MULT = 1.25
OTHER_RAID_MULT = 0.85
RANK_RAID = 1 # score-aware raiding all game: weight raids by who is ahead of us
RANK_RAID_START = 20 
RANK_AHEAD_MULT = 1.6 # multiplier on raids of rivals projected ahead of us
RANK_BEHIND_MULT = 1.0 # multiplier on raids of rivals projected behind us

_mem = {}


def _fresh():
    return {
        "turn": 0, "last_action": None, "last_pos": None,
        "last_owned": False, "last_upg": False, "last_frozen": False,
        "progress": 0, "last_shield": -99, "token_turn": None,
        "rivals": {}, "layout": None, "target": None,
        "rshield": {}, "rfrozen": {}, "flag": None, "sh_a": {}, "sh_b": {}, "raiding": None,
        "scores": {1: 0, 2: 0, 3: 0, 4: 0}, "prev_owner": [0] * 64, "prev_up": [False] * 64,
    }


def _flatten(board):
    board = list(board)
    if board and isinstance(board[0], (list, tuple)):
        board = [c for row in board for c in row]
    return board


def _detect_layout(mem, me, n, legal):
    """Work out whether board index = y*n + x ('row') or x*n + y ('col')."""
    if mem["layout"] is not None:
        return
    la, lp = mem["last_action"], mem["last_pos"]
    if la in MOVES and lp is not None and lp != me:
        dx, dy = MOVES[la]
        d = me - lp
        row, col = dx + dy * n, dx * n + dy
        if d == row and d != col:
            mem["layout"] = "row"
        elif d == col and d != row:
            mem["layout"] = "col"
        return
    if mem["turn"] == 0:  # first call: server may hide off-board moves
        have = {a for a in legal if a in MOVES}
        if len(have) < 4:
            match = []
            for lay in ("row", "col"):
                x, y = (me % n, me // n) if lay == "row" else (me // n, me % n)
                exp = {a for a, (dx, dy) in MOVES.items()
                       if 0 <= x + dx < n and 0 <= y + dy < n}
                if exp == have:
                    match.append(lay)
            if len(match) == 1:
                mem["layout"] = match[0]


def _token_holder(mem, turn, me_id, flag):
    if flag:
        mem["token_turn"] = turn
        return me_id
    if mem["token_turn"] is not None:
        return ((me_id - 1 + (turn - mem["token_turn"])) % 4) + 1
    guess = (turn % 4) + 1          # token starts at seat 1 (assumption)
    return None if guess == me_id else guess


def _bfs(me, blocked, n, xy, idx):
    dist, first = {me: 0}, {me: None}
    q = deque([me])
    while q:
        cur = q.popleft()
        x, y = xy(cur)
        for a, (dx, dy) in MOVES.items():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < n and 0 <= ny < n):
                continue
            j = idx(nx, ny)
            if j in dist or j in blocked:
                continue
            dist[j] = dist[cur] + 1
            first[j] = a if cur == me else first[cur]
            q.append(j)
    return dist, first


def _update_score_ledger(mem, cells, turn):
    """Reconstruct every player's cumulative score from board deltas (verified exact)."""
    if len(mem["prev_owner"]) != len(cells):
        mem["prev_owner"] = [0] * len(cells)
        mem["prev_up"] = [False] * len(cells)
    if turn >= 1:
        income = {1: 0, 2: 0, 3: 0, 4: 0}
        for c in cells:
            if c.owner:
                income[c.owner] += c.base_points * (2 if c.is_upgraded else 1)
        for p in (1, 2, 3, 4):
            mem["scores"][p] += income[p]
        for i, c in enumerate(cells):
            po = mem["prev_owner"][i]
            if po != 0 and po != c.owner and c.owner != 0:
                L = 2 if mem["prev_up"][i] else 1
                mem["scores"][c.owner] += 8 * c.base_points * L
    for i, c in enumerate(cells):
        mem["prev_owner"][i] = c.owner
        mem["prev_up"][i] = c.is_upgraded


def _decide(state, legal):
    mem = _mem
    raw = list(state.board)
    nested = bool(raw) and isinstance(raw[0], (list, tuple))
    cells = _flatten(raw)                      # row-major: index = row * n + col
    n = int(round(len(cells) ** 0.5))
    me_id = state.bot_id

    me = None
    for i, c in enumerate(cells):
        if c.occupant == me_id:
            me = i
            break
    if me is None:
        return "STAY"

    # new game detected? README says globals are cleared per game; as a guard
    # against a reused module, reset only if the board is provably pristine
    # (nothing captured, all four bots on their home cells) after we already
    # played a few turns.  A slow start can never trigger this.
    owned = sum(1 for c in cells if c.owner != 0)
    homes = sum(1 for c in cells if c.is_home)
    at_own_home = sum(1 for s_ in range(1, 5)
                      if s_ - 1 < len(SEAT_HOME) and SEAT_HOME[s_ - 1] < len(cells)
                      and cells[SEAT_HOME[s_ - 1]].occupant == s_)
    if not mem or (mem["turn"] >= 3 and owned <= homes and at_own_home >= 4):
        mem.clear()
        mem.update(_fresh())

    turn = mem["turn"]
    # ---- who are we playing?  detect upgrade-first, slow-expanding rivals ----
    if mem["flag"] is None and FLAG_T0 <= turn <= FLAG_T1:
        cnt, upc = {}, {}
        for c in cells:
            if c.owner and c.owner != me_id:
                cnt[c.owner] = cnt.get(c.owner, 0) + 1
                if c.is_upgraded:
                    upc[c.owner] = upc.get(c.owner, 0) + 1
        for rid, k in cnt.items():
            if k <= FLAG_MAXC and upc.get(rid, 0) >= FLAG_MINU:
                mem["flag"] = turn
    if mem["flag"] is not None and mem["flag"] != -1 and turn < FLAG_END:
        upg_w = UPG_FLAG
    elif MID_LO <= turn < MID_HI:
        upg_w = UPG_MID
    else:
        upg_w = UPG_WEIGHT
    if nested:
        mem["layout"] = "row"                  # README: board[row][col]
    _detect_layout(mem, me, n, legal)
    layout = mem["layout"] or "row"

    def xy(i):
        return (i % n, i // n) if layout == "row" else (i // n, i % n)

    def idx(x, y):
        return y * n + x if layout == "row" else x * n + y

    cur = cells[me]
    frozen_now = (len(legal) == 1 and legal[0] == "STAY") or bool(getattr(cur, "is_frozen", False))

    # ---- my own progress counter -------------------------------------
    la, lp = mem["last_action"], mem["last_pos"]
    if la == "STAY" and lp == me and not mem["last_frozen"]:
        if cur.owner == me_id and not mem["last_owned"]:
            prog = 0                              # just captured it
        elif cur.owner == me_id and cur.is_upgraded and not mem["last_upg"]:
            prog = 0                              # just upgraded it
        else:
            prog = mem["progress"] + 1
    else:
        prog = 0
    if frozen_now:
        prog = 0
    mem["progress"] = prog

    # ---- rival tracking: how long each has sat on the same cell --------
    rivals_now = {}
    for i, c in enumerate(cells):
        if c.occupant and c.occupant != me_id:
            rivals_now[c.occupant] = i
    old = mem["rivals"]
    rivals = {}
    for rid, i in rivals_now.items():
        if getattr(cells[i], "is_frozen", False):
            rivals[rid] = (i, -1)               # server says it is frozen right now
        elif rid in old and old[rid][0] == i:
            rivals[rid] = (i, old[rid][1] + 1)  # -1 -> 0 on its first free turn
        else:
            rivals[rid] = (i, 0)

    # A rival that has just become frozen reveals a shield used last turn:
    #  - frozen on someone else's land -> that land's owner shielded and caught it
    #  - frozen on its own/neutral land -> it shielded and caught nobody (backfire)
    for rid, (i, p) in rivals.items():
        was = mem["rfrozen"].get(rid, False)
        isf = bool(getattr(cells[i], "is_frozen", False))
        if isf and not was:
            ow = cells[i].owner
            if ow == rid or ow == 0:
                mem["rshield"][rid] = turn - 1
            elif ow != me_id:
                mem["rshield"][ow] = turn - 1
        mem["rfrozen"][rid] = isf

    # ---- opponent model: does the owner of the cell I was raiding shield? ----
    rj = mem.get("raiding")
    if rj is not None and rj[0] == me and la == "STAY":
        ow = rj[1]
        a0, b0 = mem["sh_a"].get(ow, SHIELD_PRIOR[0]), mem["sh_b"].get(ow, SHIELD_PRIOR[1])
        if mem["last_frozen"] is False and frozen_now and cur.owner == ow:
            mem["sh_a"][ow] = a0 + 1            # got frozen on their land: they shield
        elif cur.owner == me_id:
            mem["sh_b"][ow] = b0 + 1            # raid succeeded: they did not shield
    mem["raiding"] = None

    holder = _token_holder(mem, turn, me_id, state.is_priority_token)
    R = TURNS - turn                               # turns left incl. this one
    lock_in, catchup_target = False, None
    ahead_set = set()
    ledger_ok = True
    try:                                   # the ledger is an add-on: it must never break the core bot
        _update_score_ledger(mem, cells, turn)
    except Exception:
        ledger_ok = False
    if ledger_ok and (LOCK_IN or CATCHUP or RANK_RAID):
        try:
            cur_inc = {p: 0 for p in (1, 2, 3, 4)}
            for c in cells:
                if c.owner:
                    cur_inc[c.owner] += c.base_points * (2 if c.is_upgraded else 1)
            proj = {p: mem["scores"][p] + R * cur_inc[p] for p in (1, 2, 3, 4)}
            others = [p for p in (1, 2, 3, 4) if p != me_id]
            best_rival = max(others, key=lambda p: proj[p])
            ahead_set = {p for p in others if proj[p] > proj[me_id]}
            my_rank = 1 + sum(1 for p in others if proj[p] > proj[me_id])
            if LOCK_IN and turn >= RANK_MODE_START and my_rank == 1 \
                    and proj[me_id] - proj[best_rival] >= LOCK_IN_MARGIN:
                lock_in = True
            if CATCHUP and turn >= RANK_MODE_START and my_rank == 2:
                gap = proj[best_rival] - proj[me_id]
                if 0 < gap <= CATCHUP_MARGIN:
                    catchup_target = best_rival
        except Exception:
            lock_in, catchup_target, ahead_set = False, None, set()

    def snapshot(action, target=None):
        mem["last_pos"] = me
        mem["last_owned"] = cur.owner == me_id
        mem["last_upg"] = bool(cur.is_upgraded)
        mem["last_frozen"] = frozen_now
        mem["rivals"] = rivals
        mem["target"] = target
        return action

    if frozen_now:
        return snapshot("STAY")

    # ---- defence: SHIELD when a rival is about to take one of our cells --
    can_shield = "SHIELD" in legal and (turn - mem["last_shield"] > 4)
    if can_shield:
        threat = False
        for rid, (i, p) in rivals.items():
            c = cells[i]
            if c.owner != me_id or c.is_home or p < 0:   # p<0: already frozen
                continue
            if c.is_upgraded:
                if p >= SHIELD_UP_PROG:
                    threat = True
            else:
                # level-1 cell: shield only delays the loss by ~3 turns, so it
                # is worth it for valuable cells or right at the end of the game
                if c.base_points >= SHIELD_L1_MIN_VALUE or R <= 3:
                    threat = True
        if MULTI_CATCH and not threat and R <= MULTI_THREAT_R:
            on_land = [p for rid, (i, p) in rivals.items()
                       if cells[i].owner == me_id and not cells[i].is_home and p >= 0]
            if len(on_land) >= 2 and any(p >= 1 for p in on_land):
                threat = True
        if threat:
            mem["last_shield"] = turn
            return snapshot("SHIELD")

    # v12 lock-in: protect a large projected lead late in the game
    if lock_in:
        if cur.owner in (0, me_id) or cur.is_home:
            return snapshot("STAY")
        blocked_now = {i for i, c in enumerate(cells) if c.occupant and c.occupant != me_id}
        sd, sf = _bfs(me, blocked_now, n, xy, idx)
        safe = [j for j in sd if j != me and (cells[j].owner in (0, me_id) or cells[j].is_home)]
        if safe:
            j = min(safe, key=lambda q: (sd[q], 0 if cells[q].owner == me_id else 1, -cells[q].base_points))
            if sf[j] in legal:
                return snapshot(sf[j], j)
        return snapshot("STAY")

    # ---- offence / expansion: pick the best target ---------------------
    blocked = set(i for i, _ in rivals.values())
    dist, first = _bfs(me, blocked, n, xy, idx)

    # who could step into a given cell next to us this turn?
    def contest_lost(j):
        order = None
        if holder is not None:
            order = [((holder - 1 + k) % 4) + 1 for k in range(4)]
        x, y = xy(j)
        for rid, (i, p) in rivals.items():
            if p < 0:
                continue
            rx, ry = xy(i)
            if abs(rx - x) + abs(ry - y) != 1:
                continue
            if OWN_CONTEST_FIX and cells[j].owner == me_id:
                return False
            if cells[j].owner == rid:
                return True
            if order is None or order.index(rid) < order.index(me_id):
                return True
        return False

    def voronoi(j, d):
        order = None
        if holder is not None:
            order = [((holder - 1 + k) % 4) + 1 for k in range(4)]
        x, y = xy(j)
        fac = 1.0
        for rid, (i, p) in rivals.items():
            rx, ry = xy(i)
            dr = abs(rx - x) + abs(ry - y) + (2 if p < 0 else 0)
            if dr < d:
                fac = min(fac, VORONOI_LOSS if cells[j].owner != rid else 0.2)
            elif dr == d and (cells[j].owner == rid or order is None
                              or order.index(rid) < order.index(me_id)):
                fac = min(fac, VORONOI_TIE)
        return fac

    def base_gain(j, dd):
        c = cells[j]
        if c.is_home:
            return 0.0
        v = c.base_points
        if c.owner == me_id:
            return 0.0 if c.is_upgraded else UPG_WEIGHT * v * max(0, R - dd - 3) / (dd + 3)
        if c.owner == 0:
            return v * max(0, R - dd - 1) / (dd + 1)
        return 0.0

    items = []
    best, best_score = None, 0.0
    for j, d in dist.items():
        c = cells[j]
        if c.is_home:
            continue
        v = c.base_points ** VAL_EXP
        f = 1.0
        if c.owner == me_id:
            if c.is_upgraded:
                continue
            need = max(1, 3 - prog) if j == me else 3
        elif c.owner == 0:
            need = 1
        else:
            if c.is_upgraded:
                need = max(1, 3 - prog) if j == me else 3
            else:
                need = 1
        T = d + need
        if T > R:
            continue
        gain_turns = R - T + 1
        if c.owner == me_id:
            G = upg_w * v * gain_turns              # extra income from level 2
        elif c.owner == 0:
            G = NEUTRAL_WEIGHT * v * gain_turns
        else:
            L = 2 if c.is_upgraded else 1
            G = RAID_BONUS_W * 8 * v * L + v * gain_turns + DENY_WEIGHT * v * L * gain_turns
            pa = mem["sh_a"].get(c.owner, SHIELD_PRIOR[0])
            pb = mem["sh_b"].get(c.owner, SHIELD_PRIOR[1])
            psh = pa / (pa + pb)                      # estimated P(owner shields)
            f = (1.0 - psh) * (1.0 if not c.is_upgraded else RAID_UP_MULT)
            f = max(f, 0.08)
            last = mem["rshield"].get(c.owner)
            # owner cannot shield again until last+5, so our take must finish by last+4
            if last is not None and turn + T - 1 <= last + 4:
                f = max(f, RAID_SAFE_F)
            if catchup_target is not None:
                f *= PAIRWISE_RAID_MULT if c.owner == catchup_target else OTHER_RAID_MULT
            if RANK_RAID and turn >= RANK_RAID_START:
                f *= RANK_AHEAD_MULT if c.owner in ahead_set else RANK_BEHIND_MULT
        if d == 1 and contest_lost(j):
            f *= CONTEST_FACTOR
        elif d > 1:
            f *= voronoi(j, d)
        s = G * f / T
        items.append((j, d, v, need, f, c.owner == me_id, c.owner not in (0, me_id), 1 if not c.is_upgraded else 2, s))
        if FOLLOW_W > 0 and j != me:
            fo = 0.0
            for k in dist:
                if k != j:
                    dd = abs(xy(k)[0] - xy(j)[0]) + abs(xy(k)[1] - xy(j)[1])
                    if dd < 1:
                        continue
                    g2 = base_gain(k, dd)
                    if g2 > fo:
                        fo = g2
            s += FOLLOW_W * fo * (R - T > 4)
        if j == mem["target"]:
            s *= STICKY
        if s > best_score or (s == best_score and best is not None and d < dist[best]):
            best, best_score = j, s

    if CHAIN > 1 and items:
        # best chain of up to CHAIN targets by (points gained / turns spent);
        # distances between chain targets are Manhattan (fast, close enough)
        top = sorted(items, key=lambda t: -t[8])[:TOPN]
        m = len(top)
        xs = [xy(t[0])[0] for t in top]
        ys = [xy(t[0])[1] for t in top]
        needs = [t[3] for t in top]
        d0 = [t[1] for t in top]
        A, Bc = [], []
        for (j, d, v, need, f, own, raid, L, _s) in top:
            if own:
                A.append(0.0); Bc.append(upg_w * v * f)
            elif raid:
                A.append(RAID_BONUS_W * 8 * v * L * f); Bc.append((v + DENY_WEIGHT * v * L) * f)
            else:
                A.append(0.0); Bc.append(NEUTRAL_WEIGHT * v * f)
        tgt = mem["target"]
        best_state = [0.0, None]

        def rec(depth, t, tot, last, used, first):
            if depth and tot / t > best_state[0]:
                best_state[0], best_state[1] = tot / t, first
            if depth >= CHAIN:
                return
            for k in range(m):
                if (used >> k) & 1:
                    continue
                if depth == 0:
                    tt = d0[k] + needs[k]
                else:
                    dd = abs(xs[last] - xs[k]) + abs(ys[last] - ys[k])
                    if dd == 0:
                        continue
                    tt = dd + needs[k]
                nt = t + tt
                if nt > R:
                    continue
                v_ = A[k] + Bc[k] * (R - nt + 1)
                if depth == 0 and top[k][0] == tgt:
                    v_ *= STICKY
                rec(depth + 1, nt, tot + v_, k, used | (1 << k), k if depth == 0 else first)

        rec(0, 0, 0.0, -1, 0, -1)
        if best_state[1] is not None:
            best = top[best_state[1]][0]

    if best is None:
        return snapshot("STAY")
    if cells[best].owner not in (0, me_id) and not cells[best].is_home:
        mem["raiding"] = (best, cells[best].owner)
    if best == me:
        return snapshot("STAY", best)
    return snapshot(first[best], best)


def act(state):  # Do not change or modify this function signature
    legal = list(state.legal_actions)
    action = None
    try:
        action = _decide(state, legal)
    except Exception:
        action = None
    if action not in legal:
        action = "STAY" if "STAY" in legal else legal[0]
    if _mem:
        _mem["last_action"] = action
        _mem["turn"] = _mem.get("turn", 0) + 1
    return action