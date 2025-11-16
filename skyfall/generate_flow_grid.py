# At a specific moment, analyze the traffic throughput of each link (ISLs and GSLs) under the network topology and traffic conditions.
# Network Topology: User blocks connect to a satellite, and satellites forms +Grid.
# Traffic: Based on the Starlink traffic of each block from CloudFlare, flows probabilistically increasing by 0.5M each sampling time (ISL capacity 20Gbps; GSL Uplink/Downlink: 4Gbps).
# as described in the experimental setting (Section V.A)

import math
import sys
import numpy as np
import random, json
import os

# ---- comb fallback (Python < 3.8) ----
try:
    from math import comb as _comb
    def nCk(n, k):
        return _comb(n, k)
except Exception:
    def nCk(n, k):
        """binomial coefficient C(n,k) without math.comb (works on Python 3.6/3.7)"""
        if k < 0 or k > n:
            return 0
        k = min(k, n - k)
        res = 1
        # multiplicative formula to avoid big factorials
        for i in range(1, k + 1):
            res = res * (n - k + i) // i
        return res

# Constants
RADIUS = 6371
Flow_size = 0.5  # 0.5M

f = open("../config.json", "r", encoding='utf8')
table = json.load(f)
cons_name = table["Name"]
altitude = int(table["Altitude (km)"])
num_of_orbit = table["# of orbit"]
sat_of_orbit = table["# of satellites"]
inclination = table["Inclination"]  # inclination
sat_num = num_of_orbit * sat_of_orbit
user_num = inclination * 2 * 360
sat_pos_car = []  # 1584 satellites' positions
user_pos_car = []  # inclination * 2 * 360 blocks' positions (from high-latitude to low-latitude and high-longitude to low-longitude areas)
GS_pos_car = []  # GS positions
pop_count = []  # 53 ~ -53, traffic probability per block
user_connect_sat = []  # satellite connected to each block 
sat_connect_gs = []  # GS connected to each satellite 
gsl_occurrence = [[] for i in range(sat_num)] # blocks served by each GSL 
gsl_occurrence_num = [-1 for i in range(sat_num)] # number of blocks served by each GSL 
path = [-1 for _ in range(sat_num)]  # path[i][j] = k: k is the next hop from i to j; k == j shows that k/j is a landing satellite connected to a GS
link_utilization_ratio = 100
isl_capacity = 20480 
uplink_capacity = downlink_capacity = 4096
bandwidth_isl = isl_capacity * link_utilization_ratio / 100
bandwidth_uplink = uplink_capacity * link_utilization_ratio / 100
bandwidth_downlink = uplink_capacity * link_utilization_ratio / 100
link_traffic = [
    0
] * sat_num * 6  # 6 links in total for a satellite, including 4 ISLs, one downlink (6*i+2), and one uplink (6*i+3)
isl_traffic = [0] * sat_num  # egress traffic per satellite
downlink_traffic = [0] * sat_num  # downlink traffic per satellite
uplink_traffic = [0] * sat_num  # uplink traffic per satellite
sat_cover_pop = [0] * sat_num # total user traffic accessed by a satellite
sat_cover_user = [[] for i in range(sat_num)
                  ]  
flows = [
]  # all the candidate flows. flow[k] = [src_sat, dst_sat, weight]. 
flows_selected = {
}  # legal background flows according to the probability. {(src_sat,dst,sat):bandwidth,...} 。环状为{(src_sat,sat):bandwidth,...}
flows_num = 0
flows_cumulate_weight = []
flows_sum_weight = 0

# ==== K-DS switches (no packet splitting) ====
KDS_ENABLE = False           # ← k-DS 라우팅 활성화 여부 (True로 켜기)
KDS_K = 10                    # 후보 경로 수 k
KDS_NODE_DISJOINT = True     # True=노드-분리(내부 위성 겹침 금지), False=링크-분리
KDS_CHECK_ALL_CAPS = False   # False: DL만 사전검사(기존 정책과 유사), True: UL/ISL/DL 모두 검사

# k-DS 캐시/상태
KDS_PATHSET_CACHE = {}       # {(src, landing, k, node_disjoint): [paths...]}
KDS_RR_IDX = {}              # {(src, landing, k, node_disjoint): next_index}  # 라운드로빈 커서

# ==== PCDR switches ====
PCDR_ENABLE = True           # 분산 라우팅 on/off
PCDR_P = 0.8                 # 동적 n 계산 시 확률 가중 p (0~1)
PCDR_MIN_CHUNK = 0.05         # [Mb] 최소 청크(Flow_size=0.5M과 정합)
PCDR_MAX_K = 64              # 동일 코스트 최단 경로를 한 번에 샘플링할 상한

rng = random.Random(20250917)  # 시드



# 여러 착륙위성(같은 GS)으로 분산하기 위한 후보 폭
# - CANDIDATE_SLACK: 기본 착륙까지의 최단 hop L0와 |L-L0|<=slack인 위성만 후보로 사용
# - MAX_CANDIDATES: 후보 위성 수의 상한 (가벼운 라운드로빈/그리디 용)
PCDR_CANDIDATE_SLACK = 1
PCDR_MAX_CANDIDATES = 8
PCDR_CROSS_GS = True       # 착륙을 다른 GS까지 허용할지 여부. 기본값: 금지(False)

assert KDS_K <= PCDR_MAX_K, "KDS_K must be <= PCDR_MAX_K"

# 캐시
landing_cache = {}           # key: src_sat -> landing_sat
path_pool_cache = {}         # key: (src_sat, landing_sat) -> [경로리스트...]
# 착륙 후보 사전계산 캐시
LANDINGS_BY_GS = {}          # {gs_id: [착륙위성들]}
LANDINGS_ALL  = []           # (NEW) 모든 착륙 가능 위성 리스트
BASE_LANDING  = None         # [src_sat] -> 기본 착륙 위성
CANDS_BY_SRC  = None         # [src_sat] -> 사전필터된 후보 위성 리스트

# ==== Loss accounting ====
# 1) 링크별 실제 누적 트래픽(link_traffic)과 용량의 '초과분'을 사후 계산해 손실로 본다
# 2) add_flow*_ 가 전체 거부(-1)한 청크의 총량을 별도 합산
LOSS_BLOCKED_TOTAL = 0.0     # [Mb] 완전 차단된(전혀 수용 안 된) 청크 합계


def cir_to_car_np(lat, lng, h):
    x = (RADIUS + h) * math.cos(math.radians(lat)) * math.cos(
        math.radians(lng))
    y = (RADIUS + h) * math.cos(math.radians(lat)) * math.sin(
        math.radians(lng))
    z = (RADIUS + h) * math.sin(math.radians(lat))
    return np.array([x, y, z])


def link_seq(sati, satj): 
    orbiti = int(sati / sat_of_orbit)
    orbitj = int(satj / sat_of_orbit)
    if orbiti == orbitj and (satj == sati + 1 or satj == sati -
                             (sat_of_orbit - 1)):
        return sati * 6
    elif satj == (sati + sat_of_orbit) % sat_num:
        return sati * 6 + 1
    elif orbiti == orbitj and (satj == sati - 1 or satj == sati +
                               (sat_of_orbit - 1)):
        return sati * 6 + 2
    elif satj == (sati - sat_of_orbit + sat_num) % sat_num:
        return sati * 6 + 3
    else:
        return -1


def floyd(): # calculate the next hop from A to B
    global path
    global gsl_occurrence
    global sat_cover_user
    # inter-orbit first, then intra-orbit
    for orbit_id in range(num_of_orbit):
        for sat_id in range(sat_of_orbit):
            next_sat, sat_index = find_next_sat(
                orbit_id, sat_id, sat_of_orbit,
                sat_connect_gs)  # the nearest landing satellite (-1: no landing satellite)
            path[sat_index] = next_sat  # -1: no more landing satellite. Land from this satellite
    for orbit_id in range(num_of_orbit):
        for sat_id in range(sat_of_orbit):
            landing_sat = path[orbit_id * sat_of_orbit + sat_id]
            while landing_sat != path[landing_sat]:
                landing_sat = path[landing_sat]
            for item in sat_cover_user[orbit_id * sat_of_orbit + sat_id]:
                gsl_occurrence[landing_sat].append(item)


def find_next_sat(orbit_id, sat_id, sat_of_orbit,
                  sat_connect_gs):  # the the next satellite (-1: no landing satellite)
    next_sat = -1
    min_hops = int(sat_of_orbit / 2) + int(num_of_orbit / 2)
    sat_index = orbit_id * sat_of_orbit + sat_id
    if sat_connect_gs[sat_index] != -1:
        return sat_index, sat_index
    for item in range(sat_num):
        if sat_connect_gs[item] != -1:
            item_orbit = int(item / sat_of_orbit)
            item_sat = item % sat_of_orbit
            orbit_diff = abs(item_orbit -
                             orbit_id) if abs(item_orbit - orbit_id) <= int(
                                 num_of_orbit /
                                 2) else num_of_orbit - abs(item_orbit -
                                                            orbit_id)
            sat_diff = abs(item_sat - sat_id) if abs(item_sat - sat_id) <= int(
                sat_of_orbit / 2) else sat_of_orbit - abs(item_sat - sat_id)
            if (sat_diff + orbit_diff) >= min_hops:
                continue

            min_hops = (sat_diff + orbit_diff)
            if item_orbit == orbit_id:  # same orbit
                if item_sat > sat_id:
                    if (item_sat - sat_id) <= (sat_of_orbit -
                                               (item_sat - sat_id)):
                        next_sat = sat_index + 1
                    elif sat_id == 0:
                        next_sat = sat_index + sat_of_orbit - 1
                    else:
                        next_sat = sat_index - 1
                else:
                    if (sat_id - item_sat) <= (sat_of_orbit -
                                               (sat_id - item_sat)):
                        next_sat = sat_index - 1
                    elif sat_id == sat_of_orbit - 1:
                        next_sat = sat_index - sat_of_orbit + 1
                    else:
                        next_sat = sat_index + 1
            else:  # not same orbit
                if item_orbit > orbit_id:
                    if (item_orbit - orbit_id) <= (num_of_orbit -
                                                   (item_orbit - orbit_id)):
                        next_sat = sat_index + sat_of_orbit
                    elif orbit_id == 0:
                        next_sat = sat_index + sat_of_orbit * (num_of_orbit -
                                                               1)
                    else:
                        next_sat = sat_index - sat_of_orbit
                else:
                    if (orbit_id - item_orbit) <= (num_of_orbit -
                                                   (orbit_id - item_orbit)):
                        next_sat = sat_index - sat_of_orbit
                    elif orbit_id == num_of_orbit - 1:
                        next_sat = sat_index - sat_of_orbit * (num_of_orbit -
                                                               1)
                    else:
                        next_sat = sat_index + sat_of_orbit

    return next_sat, sat_index


def init_flows():  # initiate flows and weights
    global flows
    global flows_sum_weight
    global flows_cumulate_weight
    global flows_num
    for block in range(inclination * 2 * 360):
        if pop_count[block] == 0:
            continue
        weight = math.ceil(pop_count[block]) 
        flows.append([block, weight])
        flows_sum_weight += weight
        flows_cumulate_weight.append(flows_sum_weight)  # weight for each link
    flows_num = len(flows_cumulate_weight)


def get_one_flow(
        cumulate_weight, num,
        sum_weight):  # randomly choose one
    rand = random.randint(1, sum_weight)
    low = 0
    high = num - 1
    while low < high:
        mid = (low + high) >> 1
        if rand > cumulate_weight[mid]:
            low = mid + 1
        elif rand < cumulate_weight[mid]:
            high = mid
        else:
            return mid
    return low

def add_flow(src_block, rate=Flow_size):  
    global link_traffic
    src_sat = user_connect_sat[src_block]
    if src_sat == -1:
        return -1
    # traverse all the paths to update link_traffic
    uplink = src_sat * 6 + 5
    # determine whether the constraints are met
    from_sat = src_sat
    if path[from_sat] == -1:
        return 0
    while True:
        to_sat = path[from_sat]
        if to_sat != from_sat:
            link_id_1 = link_seq(from_sat, to_sat)
            if link_id_1 == -1:
                print('error!')
                print(from_sat, to_sat)
                exit(0)
            link_traffic[link_id_1] += rate
            if link_id_1 % 6 == 0:
                isl_traffic[from_sat] += rate  # isl_traffic for dual-traffic 
            elif link_id_1 % 6 == 1:
                isl_traffic[to_sat] += rate  # isl_traffic for dual-traffic 
            from_sat = to_sat
        else:
            break
    downlink = from_sat * 6 + 4
    link_traffic[uplink] += rate
    link_traffic[downlink] += rate
    uplink_traffic[src_sat] += rate
    downlink_traffic[from_sat] += rate

    if downlink_traffic[from_sat] < downlink_capacity:
        # not enough traffic
        return 0
    else:  # minus extra traffic if overloaded
        src_sat = user_connect_sat[src_block]
        # traverse all the paths to update link_traffic
        uplink = src_sat * 6 + 5
        # determine whether the constraints are met
        from_sat = src_sat
        while True:
            to_sat = path[from_sat]
            if to_sat != from_sat:
                link_id_1 = link_seq(from_sat, to_sat)
                if link_id_1 == -1:
                    print('error!')
                    print(from_sat, to_sat)
                    exit(0)
                link_traffic[link_id_1] -= rate
                if link_id_1 % 6 == 0:
                    isl_traffic[from_sat] -= rate  # isl_traffic for dual-traffic 
                elif link_id_1 % 6 == 1:
                    isl_traffic[to_sat] -= rate  # isl_traffic for dual-traffic 
                from_sat = to_sat
            else:
                break
        downlink = from_sat * 6 + 4
        link_traffic[uplink] -= rate
        link_traffic[downlink] -= rate
        uplink_traffic[src_sat] -= rate
        downlink_traffic[from_sat] -= rate
        return -1



def torus_diff(a, b, mod):
    d = (b - a) % mod
    return (d, +1) if d <= mod // 2 else (mod - d, -1)

def grid_l1(src, dst, sat_of_orbit, num_of_orbit):
    so = sat_of_orbit; no = num_of_orbit
    o_src, s_src = divmod(src, so); o_dst, s_dst = divmod(dst, so)
    d_orb, dir_orb = torus_diff(o_src, o_dst, no)
    d_sat, dir_sat = torus_diff(s_src, s_dst, so)
    return d_orb, dir_orb, d_sat, dir_sat

def step_neighbor(cur, choose_orbit, dir_orb, dir_sat, sat_of_orbit, num_of_orbit):
    if choose_orbit:  # inter-orbit
        nxt_orb = (cur // sat_of_orbit + dir_orb) % num_of_orbit
        nxt_sat = cur % sat_of_orbit
        return nxt_orb * sat_of_orbit + nxt_sat
    else:             # intra-orbit
        nxt_orb = cur // sat_of_orbit
        nxt_sat = (cur % sat_of_orbit + dir_sat) % sat_of_orbit
        return nxt_orb * sat_of_orbit + nxt_sat

def shortest_random_path(src, dst, sat_of_orbit, num_of_orbit, rng_obj):
    d_orb, dir_orb, d_sat, dir_sat = grid_l1(src, dst, sat_of_orbit, num_of_orbit)
    cur = src; nodes = [cur]
    while d_orb > 0 or d_sat > 0:
        choose_orbit = (rng_obj.random() < 0.5) if (d_orb > 0 and d_sat > 0) else (d_orb > 0)
        cur = step_neighbor(cur, choose_orbit, dir_orb, dir_sat, sat_of_orbit, num_of_orbit)
        nodes.append(cur)
        if choose_orbit: d_orb -= 1
        else: d_sat -= 1
    return nodes

def k_shortest_random_eqcost_paths(src, dst, k, sat_of_orbit, num_of_orbit, rng_obj):
        # (legacy helper left as-is; not used after we switch to the no-rejection sampler)
    paths = []
    for _ in range(k):
        nodes = shortest_random_path(src, dst, sat_of_orbit, num_of_orbit, rng_obj)
        paths.append(nodes)
    return paths


def get_landing_sat_cached_path(next_hop_table, src_sat):
    # next_hop_table: +Grid는 path, +Circle은 cycle_path (1차원 next-hop)
    if src_sat in landing_cache:
        return landing_cache[src_sat]
    seen = set(); cur = src_sat
    max_steps = len(next_hop_table) + 5
    for _ in range(max_steps):
        if cur == -1:
            landing_cache[src_sat] = -1; return -1
        nxt = next_hop_table[cur]
        if nxt == -1:
            landing_cache[src_sat] = -1; return -1
        if nxt == cur:
            landing_cache[src_sat] = cur; return cur
        if cur in seen:
            landing_cache[src_sat] = -1; return -1
        seen.add(cur); cur = nxt
    landing_cache[src_sat] = -1; return -1

def get_path_pool_cached(src_sat, landing_sat, sat_of_orbit, num_of_orbit):
    key = (src_sat, landing_sat)
    if key in path_pool_cache:
        return path_pool_cache[key]
    # Use no-rejection combinational unranking sampler for unique equal-cost shortest paths
    pool = k_eqcost_paths_no_rejection(src_sat, landing_sat, PCDR_MAX_K,
                                       sat_of_orbit, num_of_orbit, rng)
    path_pool_cache[key] = pool
    return pool

def _get_link_capacity_by_index(idx: int) -> float:
    m = idx % 6
    if m in (0, 1, 2, 3):   # ISL
        return bandwidth_isl
    elif m == 4:            # DL
        return bandwidth_downlink
    else:                   # m == 5 → UL
        return bandwidth_uplink

def _kds_is_feasible(nodes, rate: float) -> bool:
    """
    선택한 단일 경로에 rate(=Flow_size)를 태웠을 때 용량 여유가 있는지.
    - KDS_CHECK_ALL_CAPS=False: DL만 '추가 후 < capacity' 검사(기존 add_flow 정책과 유사)
    - True: UL/ISL/DL 모두 검사
    """
    if not nodes:
        return False

    # DL만 검사(기본)
    if not KDS_CHECK_ALL_CAPS:
        dl = nodes[-1] * 6 + 4
        return (link_traffic[dl] + rate) < _get_link_capacity_by_index(dl)

    # 모두 검사(UL/ISL/DL)
    ul = nodes[0] * 6 + 5
    if (link_traffic[ul] + rate) >= _get_link_capacity_by_index(ul):
        return False
    for u, v in zip(nodes[:-1], nodes[1:]):
        lid = link_seq(u, v)
        if lid == -1:
            return False
        if (link_traffic[lid] + rate) >= _get_link_capacity_by_index(lid):
            return False
    dl = nodes[-1] * 6 + 4
    if (link_traffic[dl] + rate) >= _get_link_capacity_by_index(dl):
        return False
    return True

def _kds_commit(nodes, rate: float):
    """경로에 rate를 원자적으로 반영(UL→ISL→DL)."""
    # UL
    ul = nodes[0] * 6 + 5
    link_traffic[ul] += rate
    uplink_traffic[nodes[0]] += rate

    # ISL
    for u, v in zip(nodes[:-1], nodes[1:]):
        lid = link_seq(u, v)
        if lid == -1:
            return
        link_traffic[lid] += rate
        m = lid % 6
        if m == 0:
            isl_traffic[u] += rate
        elif m == 1:
            isl_traffic[v] += rate

    # DL
    dl = nodes[-1] * 6 + 4
    link_traffic[dl] += rate
    downlink_traffic[nodes[-1]] += rate

def _kds_pick_disjoint_paths_from_pool(pool, k: int, node_disjoint: bool):
    """
    src→landing '동코스트 최단경로' 풀에서 ISL-분리 집합을 최대 k개 그리디로 선택.
    - node_disjoint=True  → 내부 위성(경로의 첫/마지막 제외) 노드-분리
    - node_disjoint=False → 방향간선(u->v) 링크-분리
    """
    if not pool:
        return []
    cand = list(pool)
    rng.shuffle(cand)  # 편향 방지

    chosen = []
    used_nodes = set()
    used_edges = set()

    for path_nodes in cand:
        if len(chosen) >= k:
            break
        if node_disjoint:
            internal = path_nodes[1:-1]      # src/landing 제외 내부 노드
            if any(n in used_nodes for n in internal):
                continue
            chosen.append(path_nodes)
            used_nodes.update(internal)
        else:
            edges = {(u, v) for u, v in zip(path_nodes[:-1], path_nodes[1:])}
            if edges & used_edges:
                continue
            chosen.append(path_nodes)
            used_edges |= edges
    return chosen

def get_kds_pathset(src_sat: int, landing_sat: int):
    """(src, landing)에 대해 k-DS 경로 집합을 캐시로 반환(부족하면 k 미만도 허용)."""
    key = (src_sat, landing_sat, KDS_K, int(KDS_NODE_DISJOINT))
    if key in KDS_PATHSET_CACHE:
        return KDS_PATHSET_CACHE[key]

    # 동코스트 최단경로 풀(최대 PCDR_MAX_K개)에서 k-DS 분리 집합 뽑기
    pool = get_path_pool_cached(src_sat, landing_sat, sat_of_orbit, num_of_orbit)
    if not pool:
        KDS_PATHSET_CACHE[key] = []
        return []

    paths = _kds_pick_disjoint_paths_from_pool(pool, KDS_K, KDS_NODE_DISJOINT)
    KDS_PATHSET_CACHE[key] = paths
    return paths


# -------- Fast equal-cost shortest-path sampler (no rejection) --------
def unrank_combination(n, k, rank):
    """
    Restore rank-th k-combination (lexicographic) in length-n bit vector.
    Returns a list of 0/1 of length n. 1=inter-orbit hop, 0=intra-orbit hop.
    """
    bits = [0] * n
    r = rank
    ones = k
    for i in range(n):
        if ones == 0:
            break
        c = nCk(n - i - 1, ones - 1)
        if r < c:
            bits[i] = 1
            ones -= 1
        else:
            r -= c
    return bits

def k_eqcost_paths_no_rejection(src, dst, k, so, no, rng_obj):
    """
    Generate up to k unique equal-cost shortest paths in O(k·L) without retries.
    """
    d_orb, dir_orb, d_sat, dir_sat = grid_l1(src, dst, so, no)
    L = d_orb + d_sat
    if L == 0:
        return [[src]]
    P = nCk(L, d_orb)  # number of unique shortest paths
    need = min(k, P)
    idxs = rng_obj.sample(range(P), need)  # sample unique indices
    paths = []
    for r in idxs:
        bits = unrank_combination(L, d_orb, r)  # 1=inter, 0=intra
        cur = src
        nodes = [cur]
        for b in bits:
            cur = step_neighbor(cur, bool(b), dir_orb, dir_sat, so, no)
            nodes.append(cur)
        paths.append(nodes)
    return paths



def get_landing_sat(src_sat):  # 기존 이름 호환
    return get_landing_sat_cached_path(path, src_sat)


def l1_hops(src, dst):
    """+Grid 토러스에서 src→dst 최단 hop 수(L1)."""
    d_orb, _, d_sat, _ = grid_l1(src, dst, sat_of_orbit, num_of_orbit)
    return d_orb + d_sat

def build_landing_precompute():
    """floyd() 이후 1회 호출: 소스별 착륙 후보를 미리 계산.
    - PCDR_CROSS_GS=False: 같은 GS의 착륙 위성만 후보
    - PCDR_CROSS_GS=True : 모든 GS의 착륙 위성까지 후보
    """
    global LANDINGS_BY_GS, LANDINGS_ALL, BASE_LANDING, CANDS_BY_SRC
    LANDINGS_BY_GS = {}
    LANDINGS_ALL = []
    for s in range(sat_num):
        gs = sat_connect_gs[s]
        if gs != -1 and path[s] == s:   # 실제 착륙 가능한 위성만
            LANDINGS_BY_GS.setdefault(gs, []).append(s)
            LANDINGS_ALL.append(s)
    BASE_LANDING = [-1] * sat_num
    CANDS_BY_SRC = [[] for _ in range(sat_num)]
    for src in range(sat_num):
        base = get_landing_sat(src)
        BASE_LANDING[src] = base
        if base == -1:
            continue
        baseL = l1_hops(src, base)
        if not PCDR_CROSS_GS:
            gs = sat_connect_gs[base]
            landings = LANDINGS_BY_GS.get(gs, [base])
        else:
            landings = LANDINGS_ALL if LANDINGS_ALL else [base]
        # slack 이내 후보만 필터
        cand = [s for s in landings if abs(l1_hops(src, s) - baseL) <= PCDR_CANDIDATE_SLACK]
        if base not in cand:
            cand.insert(0, base)
        # 중복 제거 + 상한
        seen = set(); out = []
        for s in cand:
            if s not in seen:
                out.append(s); seen.add(s)
            if len(out) >= PCDR_MAX_CANDIDATES:
                break
        CANDS_BY_SRC[src] = out

def get_landing_candidates(src_sat,
                           slack=PCDR_CANDIDATE_SLACK,
                           max_cands=PCDR_MAX_CANDIDATES):
    """사전계산된 후보만 사용하고, 현재 DL 부하 낮은 순으로 정렬."""
    if CANDS_BY_SRC is None:
        return []
    cand = CANDS_BY_SRC[src_sat]
    if not cand:
        return []
    cand = sorted(cand, key=lambda s: downlink_traffic[s])
    return cand[:max_cands]

def pick_paths_diverse(pool, n, rng_obj):
    """pool에서 링크(방향 포함) 겹침이 최소가 되도록 n개 경로를 greedy로 선택"""
    if n >= len(pool):
        return pool[:n]

    def edges(path):
        # 방향 있는 ISL (u->v)로 셋 구성
        return {(u, v) for u, v in zip(path[:-1], path[1:])}

    # 1개는 랜덤으로 시작
    start_idx = rng_obj.randrange(len(pool))
    chosen = [pool[start_idx]]
    used_edges = set(edges(pool[start_idx]))
    remain = [p for i, p in enumerate(pool) if i != start_idx]

    # 나머지는 '겹치는 간선 수'가 최소인 경로를 차례로 추가
    while len(chosen) < n and remain:
        best = min(remain, key=lambda p: len(edges(p) & used_edges))
        chosen.append(best)
        used_edges |= edges(best)
        remain.remove(best)
    return chosen



def add_flow_pcdr(src_block, rate=Flow_size):
    """
    (중요) 기존과 달리 '착륙 위성'을 여러 개로 분산.
    - 같은 GS에 연결된 착륙 위성들 중 hop 비용이 비슷한 후보들을 모으고,
    - rate를 n개 청크로 나눈 뒤 라운드로빈으로 후보들에 배정,
    - 각 후보의 downlink 용량을 초과하지 않도록 미리 검사하고,
    - 전부 배정 가능할 때만 실제 누적(원자적 적용).
    """
    global link_traffic, isl_traffic, downlink_traffic, uplink_traffic

    src_sat = user_connect_sat[src_block]
    if src_sat == -1:
        return -1

    # 후보 착륙 위성 집합
    candidates = get_landing_candidates(src_sat)
    if not candidates:
        return 0

    # 동적 n 계산 (논문식) — per가 최소청크 미만이 되지 않도록 cap
    F = rate
    S = PCDR_MIN_CHUNK
    u = rng.random()
    n_raw = math.ceil((F / S) * (1 + PCDR_P * u))
    n_cap = max(1, int(F / S))   # floor(F/S)
    n_total = max(1, min(n_raw, n_cap))
    per = F / n_total

    # 사전 용량 점검: 후보 downlink의 남은 용량 합이 rate 이상인지
    total_avail = 0.0
    for c in candidates:
        total_avail += max(0.0, downlink_capacity - downlink_traffic[c])
    if total_avail + 1e-9 < rate:
        return -1  

    # 후보별 경로 풀은 한 번만 준비
    pool_local = {}
    for c in candidates:
        pool = get_path_pool_cached(src_sat, c, sat_of_orbit, num_of_orbit)
        pool_local[c] = pool if pool else []

    # 거대한 0-배열 대신 작은 dict에 임시 누적
    ops_link = {}  # {link_idx: delta}
    ops_isl  = {}  # {sat_idx:  delta}
    ops_dl   = {}  # {sat_idx:  delta}
    ops_ul   = {}  # {sat_idx:  delta}
    def acc(d, k, v): d[k] = d.get(k, 0) + v

    # 후보는 get_landing_candidates에서 이미 DL부하 낮은 순

    rr = 0
    placed = 0
    # 각 후보별 임시로 추가되는 downlink를 추적(사전 배정 시 용량 체크용)
    cand_td = {c: 0.0 for c in candidates}

    while placed < n_total:
        progressed = False
        for _ in range(len(candidates)):
            c = candidates[rr % len(candidates)]
            rr += 1
            # 이 후보의 downlink 용량 여유?
            if downlink_traffic[c] + cand_td[c] + per > downlink_capacity:
                continue

            # (src→c) 최단 경로 중 하나 선택 (사전준비 풀 사용)
            pool = pool_local[c]
            if not pool:
                continue
            nodes = rng.choice(pool)

            # 임시 누적 (UL)
            ul = nodes[0] * 6 + 5
            acc(ops_link, ul, per)
            acc(ops_ul,   nodes[0], per)

            # ISL
            for u_sat, v_sat in zip(nodes[:-1], nodes[1:]):
                lid = link_seq(u_sat, v_sat)
                if lid == -1:
                    return 0  # 연결 테이블 오류
                acc(ops_link, lid, per)
                m = lid % 6
                if m == 0:
                    acc(ops_isl, u_sat, per)
                elif m == 1:
                    acc(ops_isl, v_sat, per)
            # DL (nodes[-1] == c)
            dl = nodes[-1] * 6 + 4
            acc(ops_link, dl, per)
            acc(ops_dl,   nodes[-1], per)
            cand_td[c] += per

            placed += 1
            progressed = True
            if placed >= n_total:
                break
        if not progressed:
            # 한 바퀴 돌 동안 아무 것도 못 배정 → 전체 취소
            return -1

    # 모든 청크 배정 성공 → 원샷 커밋
    for i, v in ops_link.items(): link_traffic[i]     += v
    for i, v in ops_isl.items():  isl_traffic[i]      += v
    for i, v in ops_dl.items():   downlink_traffic[i] += v
    for i, v in ops_ul.items():   uplink_traffic[i]   += v
    return 0


def add_flow_kds(src_block, rate=Flow_size):
    """
    k-DS (no packet splitting) per-chunk ECMP:
    - (src_sat, landing_sat) 쌍별로 ISL-분리 k경로 집합 생성/캐시
    - 같은 쌍이 재등장할 때마다 라운드로빈으로 다음 경로 시도
    - 용량 사전검사 통과 시 원샷 커밋, 전부 실패면 -1
    """
    global link_traffic, isl_traffic, downlink_traffic, uplink_traffic

    src_sat = user_connect_sat[src_block]
    if src_sat == -1:
        return -1

    landing_sat = get_landing_sat(src_sat)  # path[] 기반 기본 착륙 위성
    if landing_sat == -1:
        return 0

    paths = get_kds_pathset(src_sat, landing_sat)
    if not paths:
        # 동코스트 경로 풀 없음 → 기본 경로 하나도 없는 특이 케이스: 그냥 무시
        return 0

    # 라운드로빈 시작 위치
    rr_key = (src_sat, landing_sat, KDS_K, int(KDS_NODE_DISJOINT))
    i0 = KDS_RR_IDX.get(rr_key, 0)
    L = len(paths)

    # i0부터 한 바퀴 돌며 가능한 경로를 찾는다
    for shift in range(L):
        nodes = paths[(i0 + shift) % L]
        if _kds_is_feasible(nodes, rate):
            _kds_commit(nodes, rate)
            # 이번에 성공한 다음 칸으로 커서 전진
            KDS_RR_IDX[rr_key] = i0 + shift + 1
            return 0

    # 모든 후보가 현재 용량 제약으로 실패
    return -1


if __name__ == "__main__":
    bound = 3.743333 * 299792.458 * 0.001
    b2 = bound * bound
    time_slot = sys.argv[1]

    # load satellite positions
    pos_filename = '../' + cons_name + '/sat_lla/%s.txt' % time_slot
    with open(pos_filename, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            satPos = line.split(',')
            sat_pos_car.append(
                cir_to_car_np(float(satPos[0]), float(satPos[1]),
                              float(satPos[2])))
        sat_pos_car = np.array(sat_pos_car)

    # load user positions
    for lat in range(inclination, -inclination, -1):  # [inclination, -inclination]
        for lon in range(-180, 180, 1):  # [-179.5,179.5]
            user_pos_car.append(cir_to_car_np(lat - 0.5, lon + 0.5, 0))
    user_pos_car = np.array(user_pos_car)

    # load traffic distribution
    traffic_file = './starlink_count.txt'
    with open(traffic_file, 'r') as fr:
        lines = fr.readlines()
        for row in range(90 - inclination, 90 + inclination):
            pop_count.extend([float(x) for x in lines[row].split(' ')[:-1]] + [0])

    print("generating +Grid traffic for timeslot: " + str(time_slot) + "...")  

    # the satellite each block is connected to (vectorized batch: O(batches) instead of O(#users * #sats))
    B = 2000  # batch size (tune if needed)
    user_connect_sat = [-1] * user_num
    for start in range(0, user_num, B):
        end = min(user_num, start + B)
        # (end-start, sat_num, 3) broadcast → squared distances
        diff = sat_pos_car[None, :, :] - user_pos_car[start:end, None, :]
        dis2 = np.einsum('ijk,ijk->ij', diff, diff)  # sum of squares; no sqrt needed
        mins = dis2.min(axis=1)
        argmins = dis2.argmin(axis=1)
        sel = mins <= b2
        # write back
        for i, ok in enumerate(sel):
            uid = start + i
            if ok:
                sid = int(argmins[i])
                user_connect_sat[uid] = sid
                if pop_count[uid] > 0:
                    sat_cover_pop[sid] += pop_count[uid]
                    sat_cover_user[sid].append(uid)
    # (list → list 그대로 유지)

    # load GS positions
    f = open("./GS.json", "r", encoding='utf8')
    GS_info = json.load(f)
    count = 0
    for key in GS_info:
        GS_pos_car.append(
            cir_to_car_np(float(GS_info[key]['lat']),
                          float(GS_info[key]['lng']), 0))
        count = count + 1
    GS_pos_car = np.array(GS_pos_car)

    # the GS a satellite is connected to
    for sat_id in range(sat_num):
        dis2 = np.sqrt(
            np.sum(np.square(GS_pos_car - sat_pos_car[sat_id]),
                   axis=1))  
        if min(dis2) > bound:
            sat_connect_gs.append(-1)  # -1 for no connection
            continue
        min_dis_sat = np.argmin(dis2)  
        sat_connect_gs.append(min_dis_sat) 

    # initiate topology and routing
    floyd() 
    if PCDR_ENABLE:
        build_landing_precompute()

    # new timeslot → clear k-DS caches
    if KDS_ENABLE:
        KDS_PATHSET_CACHE.clear()
        KDS_RR_IDX.clear()

    
    # initiate traffic flows and weights
    init_flows() 

    for add_flow_times in range(2000000):  # randomly choose 2000000 flows
        flow_id = get_one_flow(flows_cumulate_weight, flows_num,
                               flows_sum_weight)
        # 기존: res = add_flow(flows[flow_id][0])
        # 라우팅 함수 선택
        if KDS_ENABLE:
            res = add_flow_kds(flows[flow_id][0])
        elif PCDR_ENABLE:
            res = add_flow_pcdr(flows[flow_id][0])
        else:
            res = add_flow(flows[flow_id][0])



        if res == -1:
            # 패킷 손실 집계
            LOSS_BLOCKED_TOTAL += Flow_size
            continue
        # add a new flow
        flow_pair = (flows[flow_id][0], flows[flow_id][1])
        if flow_pair in flows_selected:
            flows_selected[flow_pair] += Flow_size  
        else:
            flows_selected[flow_pair] = Flow_size

    # outputs: ISL, GSL down/uplink, block connecstions, satellite connections and so on
    os.system("mkdir -p ../" + cons_name + "/+grid_data/link_traffic_data/" +
              str(time_slot))
    isl_traffic = np.array(isl_traffic, dtype=int)
    np.savetxt('../' + cons_name + '/+grid_data/link_traffic_data/' + str(time_slot) +
               '/isl_traffic.txt',
               isl_traffic,
               fmt='%d')
    downlink_traffic = np.array(downlink_traffic, dtype=int)
    np.savetxt('../' + cons_name + '/+grid_data/link_traffic_data/' + str(time_slot) +
               '/downlink_traffic.txt',
               downlink_traffic,
               fmt='%d')
    uplink_traffic = np.array(uplink_traffic, dtype=int)
    np.savetxt('../' + cons_name + '/+grid_data/link_traffic_data/' + str(time_slot) +
               '/uplink_traffic.txt',
               uplink_traffic,
               fmt='%d')
    sat_connect_gs = np.array(sat_connect_gs, dtype=int)
    np.savetxt('../' + cons_name + '/+grid_data/link_traffic_data/' + str(time_slot) +
               '/sat_connect_gs.txt',
               sat_connect_gs,
               fmt='%d')
    user_connect_sat = np.array(user_connect_sat, dtype=int)
    np.savetxt('../' + cons_name + '/+grid_data/link_traffic_data/' + str(time_slot) +
               '/user_connect_sat.txt',
               user_connect_sat,
               fmt='%d')
    id = 0
    gs_occurrence_num = [0 for i in range(len(GS_pos_car))]
    for item in gsl_occurrence:
        gsl_occurrence_num[id] = len(item) if len(item) > 0 else -1
        if sat_connect_gs[id] != -1:
            gs_occurrence_num[sat_connect_gs[id]] += gsl_occurrence_num[id]
        id +=1
    gsl_occurrence_num = np.array(gsl_occurrence_num, dtype=int)
    np.savetxt('../' + cons_name + '/+grid_data/link_traffic_data/' + str(time_slot) +
               '/gsl_occurrence_num.txt',
               gsl_occurrence_num,
               fmt='%d')
    gs_occurrence_num = np.array(gs_occurrence_num, dtype=int)
    np.savetxt('../' + cons_name + '/+grid_data/link_traffic_data/' + str(time_slot) +
               '/gs_occurrence_num.txt',
               gs_occurrence_num,
               fmt='%d')

    # ===== (NEW) Loss metrics =====
    # 링크 인덱스별(capacity와 비교) 손실량 계산
    N = sat_num * 6
    link_loss = np.zeros(N, dtype=float)
    link_cap  = np.zeros(N, dtype=float)
    for i in range(N):
        m = i % 6
        if m in (0, 1, 2, 3):           # ISL 4개
            cap = bandwidth_isl
        elif m == 4:                    # DL
            cap = bandwidth_downlink
        else:                           # m == 5 → UL
            cap = bandwidth_uplink
        link_cap[i] = cap
        over = float(link_traffic[i]) - float(cap)
        if over > 0:
            link_loss[i] = over

    # 위성 단위 UL/DL 손실(참고용: link_loss에서 m==4/5만 발췌해도 동일)
    downlink_loss = (np.array(downlink_traffic, dtype=float) - float(bandwidth_downlink))
    downlink_loss[downlink_loss < 0] = 0.0
    uplink_loss   = (np.array(uplink_traffic,   dtype=float) - float(bandwidth_uplink))
    uplink_loss[uplink_loss < 0] = 0.0

    out_dir = '../' + cons_name + '/+grid_data/link_traffic_data/' + str(time_slot)
    # 소수점 유지(기존 *_traffic.txt는 정수 저장 유지)
    np.savetxt(out_dir + '/link_loss.txt',      link_loss,      fmt='%.3f')
    np.savetxt(out_dir + '/link_capacity.txt',  link_cap,       fmt='%.3f')
    np.savetxt(out_dir + '/downlink_loss.txt',  downlink_loss,  fmt='%.3f')
    np.savetxt(out_dir + '/uplink_loss.txt',    uplink_loss,    fmt='%.3f')
    np.savetxt(out_dir + '/blocked_loss_total.txt', np.array([LOSS_BLOCKED_TOTAL], dtype=float), fmt='%.3f')
    # 참고: 손실 총합(추정) = sum(link_loss) 중 (m==4,5) + LOSS_BLOCKED_TOTAL
