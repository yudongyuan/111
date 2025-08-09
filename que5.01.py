
import numpy as np
import pandas as pd
from math import isfinite, log2
import itertools

# ========== 参数（按需调整） ==========
# 多基站输入：三微站 + 一宏站 的信道文件 + 任务到达文件
# 首选新命名（SBS_1..3, MBS_1），若不存在则回退到旧命名（BS1..3, BS4/MBS_1 不存在时以 200dB/1.0 占位）
SBS1_PATH_CANDIDATES = ["SBS_1.xlsx", "BS1.xlsx"]
SBS2_PATH_CANDIDATES = ["SBS_2.xlsx", "BS2.xlsx"]
SBS3_PATH_CANDIDATES = ["SBS_3.xlsx", "BS3.xlsx"]
MBS1_PATH_CANDIDATES = ["MBS_1.xlsx", "BS4.xlsx"]
TASK_PATH = "taskflow4.xlsx"

SHEET_PL = "大规模衰减"
SHEET_FADING = "小规模瑞丽衰减"
SHEET_TASK = "用户任务流"

P_tx_dBm = 30.0  # 默认功率（若未启用功率优化时使用）
B = 360e3           # RB 带宽 360 kHz

def dbm_to_mw(p_dbm: float) -> float:
    return 10 ** (p_dbm / 10.0)
def dbm_to_watt(p_dbm: float) -> float:
    # P(W) = 10^{(P(dBm)-30)/10}
    return 10 ** ((p_dbm - 30.0) / 10.0)
NF = 7.0            # 噪声系数 dB
# 四站容量：SBS=50，MBS=100
BS_NAMES = {0: "SBS_1", 1: "SBS_2", 2: "SBS_3", 3: "MBS_1"}
BS_CAP = {0: 50, 1: 50, 2: 50, 3: 100}

# 能耗模型参数（第五问）
P_FIXED_WATT = 28.0          # 固定能耗（W）
K_RB_WATT_PER_RB = 0.75      # RB 激活能耗系数（W/RB）
ETA_PA = 0.35                # 功放效率（损耗系数）

# 惩罚参数（你可以调整）
kappa = 0.1
alpha_pen =1e-6

# 切片参数（例：URLLC/eMBB/mMTC）
alpha_U = 0.95
beta_U = 5.0
beta_E = 3.0
beta_M = 1.0
t_max_U = 5e-3
t_max_E = 100e-3
t_max_M = 500e-3
R_min_E = 50e6

# DP / search 限幅：每用户最大试探 RB（按类设置上限以照顾 eMBB 可达性）
# 全局备用上限（不直接使用，保留兼容）
RMAX_PER_USER = 50
# 按类上限（可按需调整）
RMAX_U = 20
RMAX_E = 50
RMAX_M = 12

# 公平性权重（降低 eMBB 权重避免过度偏置）
CLASS_WEIGHT = {'U': 1.0, 'E': 1.4, 'M': 1.0}
# 当某基站存在活跃任务时，为各类预留的最小 RB 比例
MIN_E_FRACTION_PER_BS = 0.10
MIN_M_FRACTION_PER_BS = 0.10
MIN_U_FRACTION_PER_BS = 0.00
# 为避免某一类独占，设置每类的最大 RB 比例上限
MAX_E_FRACTION_PER_BS = 0.60

# epoch 设置
TOTAL_TIME_MS = 1000
EPOCH_COUNT = 10
EPOCH_MS = TOTAL_TIME_MS // EPOCH_COUNT
EPOCH_S = EPOCH_MS / 1000.0

VERY_NEG = -1e9
TIE_BREAK = 1e-9   # tie-break 小增量：越小越不影响真实边际值，只用于打破平局，按需调整

# 预计算噪声功率表（按 r）以避免重复开销
MAX_R_FOR_NOISE = max(BS_CAP.values())  # 以最大 RB 容量（MBS=100）
NOISE_MW_BY_R = np.zeros(MAX_R_FOR_NOISE + 1, dtype=float)
for r_idx in range(MAX_R_FOR_NOISE + 1):
    if r_idx <= 0:
        NOISE_MW_BY_R[r_idx] = 0.0
    else:
        # 等价于 compute_noise_mW 但只保留 mW 值
        N_dBm = -174.0 + 10.0 * np.log10(r_idx * B) + NF
        NOISE_MW_BY_R[r_idx] = 10 ** (N_dBm / 10.0)

# ========== 帮助函数 ==========
def compute_P_rx_mW(phi_dB, h, P_tx_mW=None):
    if phi_dB is None or not isfinite(phi_dB):
        phi_dB = 200.0
    if h is None or not isfinite(h):
        h = 1.0
    if P_tx_mW is None:
        P_tx_mW = dbm_to_mw(P_tx_dBm)
    return h * P_tx_mW * 10 ** (-phi_dB / 10.0)

def compute_noise_mW(r):
    if r <= 0:
        return 0.0, -999.0
    N_dBm = -174.0 + 10.0 * np.log10(r * B) + NF
    N_mW = 10 ** (N_dBm / 10.0)
    return N_mW, N_dBm

def rate_from_Pr_rx_and_r(P_rx_mW, r):
    if r <= 0:
        return 0.0, 0.0
    N_mW, _ = compute_noise_mW(r)
    if N_mW <= 0:
        return 0.0, 0.0
    snr = P_rx_mW / N_mW
    R = r * B * np.log2(np.maximum(1 + snr, 1e-12))
    return float(R), float(snr)

def rate_with_interference(P_sig_mW, P_int_mW, r):
    """计算考虑同频小区间干扰的速率，将其他基站的接收功率视为附加噪声。
    P_sig_mW: 服务基站收到功率
    P_int_mW: 其他基站合并干扰功率
    r: 分配的 RB 数
    """
    if r <= 0:
        return 0.0, 0.0
    N_mW, _ = compute_noise_mW(r)
    denom = N_mW + max(P_int_mW, 0.0)
    if denom <= 0:
        return 0.0, 0.0
    snr_eff = P_sig_mW / denom
    R = r * B * np.log2(np.maximum(1 + snr_eff, 1e-12))
    return float(R), float(snr_eff)

def mm1_Wq(lambda_rate, E_S):
    """M/M/1 排队平均等待时间近似（用于 URLLC）"""
    if E_S <= 0:
        return float('inf'), float('inf')
    rho = lambda_rate * E_S
    if rho >= 1.0 - 1e-12:
        return float('inf'), rho
    # Wq = lambda * E[S]^2 / (1 - rho)
    Wq = (lambda_rate * (E_S ** 2)) / (1.0 - rho)
    return float(Wq), float(rho)

def kingman_Wq(lambda_rate, E_S, Ca2=1.0, Cs2=0.0):
    """Kingman 近似（G/G/1）用于 eMBB/mMTC"""
    if E_S <= 0:
        return float('inf'), float('inf')
    rho = lambda_rate * E_S
    if rho >= 1.0 - 1e-12:
        return float('inf'), rho
    Wq = (rho / (1.0 - rho)) * ((Ca2 + Cs2) / 2.0) * E_S
    return float(Wq), float(rho)

def Q_base_function(utype, R, L):
    """切片基础质量函数"""
    if R <= 0 or not isfinite(L):
        return -beta_U if utype == 'U' else -beta_E if utype == 'E' else -beta_M
    if utype == 'U':
        return (alpha_U ** L) if L <= t_max_U else -beta_U
    elif utype == 'E':
        if L <= t_max_E:
            return 1.0 if R >= R_min_E else (R / R_min_E)
        else:
            return -beta_E
    else:
        return 1.0 if L <= t_max_M else -beta_M

# ========== 读取数据与预处理 ==========
def parse_sheet_with_fallback(xls, sheet_name, default_index=0):
    try:
        return xls.parse(sheet_name)
    except Exception:
        return xls.parse(default_index)

def open_excel_with_candidates(candidates):
    for p in candidates:
        try:
            return pd.ExcelFile(p), p
        except Exception:
            continue
    return None, None

# 读取四个基站的信道（大规模衰减 + 小规模瑞丽衰减）
xls_list = [None]*4
xls_list[0], path0 = open_excel_with_candidates(SBS1_PATH_CANDIDATES)
xls_list[1], path1 = open_excel_with_candidates(SBS2_PATH_CANDIDATES)
xls_list[2], path2 = open_excel_with_candidates(SBS3_PATH_CANDIDATES)
xls_list[3], path3 = open_excel_with_candidates(MBS1_PATH_CANDIDATES)

pl_df = [None]*4
fading_df = [None]*4
for b in range(4):
    if xls_list[b] is not None:
        pl_df[b] = parse_sheet_with_fallback(xls_list[b], SHEET_PL)
        fading_df[b] = parse_sheet_with_fallback(xls_list[b], SHEET_FADING)
    else:
        # 若缺失某站数据，则用占位（大损耗 + 单位增益）
        pl_df[b] = pd.DataFrame()
        fading_df[b] = pd.DataFrame()

# 读取任务到达
task_xls = pd.ExcelFile(TASK_PATH)
try:
    task_df = task_xls.parse(SHEET_TASK)
except Exception:
    task_df = task_xls.parse(0)

# 识别所有用户（排除 Time 列）
time_col = None
if 'Time' in task_df.columns:
    time_col = 'Time'
elif 'time' in task_df.columns:
    time_col = 'time'

if time_col is None:
    raise SystemExit("任务表中找不到 Time 列，请确保存在 Time 列（单位 s 或 ms，脚本假设单位为 s 或 ms 自动归一）")

# 统一时间为毫秒（如果 Time 最大值 ~1 则视为秒）
time_vals = task_df[time_col].astype(float).values
# If times appear between 0..1 (seconds), convert to ms by *1000 if max <=1.1
if np.max(time_vals) <= 1.1:
    time_vals_ms = time_vals * 1000.0
else:
    time_vals_ms = time_vals.copy()

# create mapping rows->epoch index (0..EPOCH_COUNT-1)
row_epoch_idx = np.minimum((time_vals_ms // EPOCH_MS).astype(int), EPOCH_COUNT - 1)

user_cols = [c for c in task_df.columns if c != time_col]
# filter user names that start with U/e/m (case-insensitive)
users = [c for c in user_cols if len(str(c))>0 and str(c)[0].upper() in ('U','E','M')]
users = sorted(users, key=lambda x: (x[0].upper(), int(''.join(filter(str.isdigit, x)) or 0)))
users_by_type = {'U':[u for u in users if u[0].upper()=='U'],
                 'E':[u for u in users if u[0].upper()=='E'],
                 'M':[u for u in users if u[0].upper()=='M']}

print("检测到用户（总数）:", len(users), users_by_type)

# 每行按列给出的单位是 Mbit（题目说明），所以转换为 bit
# 计算每用户典型任务大小 D_u（平均非零样本）
D_u = {}
for u in users:
    arr = []
    for v in task_df[u].astype(float).values:
        if v > 0:
            arr.append(v)
    if len(arr) == 0:
        D_u[u] = 0.01 * 1e6
    else:
        D_u[u] = float(np.mean(arr) * 1e6)

# 估计历史每用户 lambda（tasks/s）与 Ca2（用于 kingman）
# use counts per epoch from full task table
counts_by_epoch = {u: [] for u in users}
for e in range(EPOCH_COUNT):
    idxs = np.where(row_epoch_idx == e)[0]
    for u in users:
        s = 0.0
        for i in idxs:
            s += float(task_df.iloc[i][u]) * 1e6
        cnt = s / D_u[u] if D_u[u] > 0 else 0.0
        counts_by_epoch[u].append(cnt)

lam_emp = {}
Ca2 = {}
for u in users:
    arr = np.array(counts_by_epoch[u])
    mean_cnt = float(np.mean(arr))
    var_cnt = float(np.var(arr, ddof=0))
    lam_emp[u] = mean_cnt / EPOCH_S  # tasks/s historical average
    if mean_cnt > 1e-9:
        ca2 = var_cnt / (mean_cnt ** 2)
        Ca2[u] = max(0.01, min(ca2, 10.0))
    else:
        Ca2[u] = 1.0
# force URLLC Ca2 = 1.0
for u in users_by_type['U']:
    Ca2[u] = 1.0

# ========== 主循环：每个 epoch 求解三类总 RB 最优分配 ==========
results = []

for epoch in range(EPOCH_COUNT):
    # gather rows in this epoch
    idxs = np.where(row_epoch_idx == epoch)[0]
    if len(idxs) == 0:
        print(f"Epoch {epoch}: no samples, skip")
        for b in (0, 1, 2, 3):
            results.append({'epoch':epoch, 'bs': b + 1, 'P_dBm': P_tx_dBm, 'rU':0, 'rE':0, 'rM':BS_CAP[b], 'total_util_epoch':0.0})
        continue

    # per-time arrivals (bits) 与每个基站的链路增益 g = h * 10^(-phi/10)
    arrivals_time = {u: [] for u in users}
    gains_time = {0: {u: [] for u in users}, 1: {u: [] for u in users}, 2: {u: [] for u in users}, 3: {u: [] for u in users}}
    pl_time = {0: {u: [] for u in users}, 1: {u: [] for u in users}, 2: {u: [] for u in users}, 3: {u: [] for u in users}}
    times_in_epoch = []

    for i in idxs:
        times_in_epoch.append(time_vals_ms[i]/1000.0)  # seconds
        for u in users:
            arrivals_time[u].append(float(task_df.iloc[i][u]) * 1e6)
            # 记录每个基站的路径损耗与小尺度增益（4 站）
            for b in (0, 1, 2, 3):
                try:
                    phi_v = float(pl_df[b].iloc[i][u])
                except Exception:
                    phi_v = 200.0
                try:
                    h_v = float(fading_df[b].iloc[i][u])
                except Exception:
                    h_v = 1.0
                if not isfinite(phi_v):
                    phi_v = 200.0
                if not isfinite(h_v):
                    h_v = 1.0
                pl_time[b][u].append(phi_v)
                gains_time[b][u].append((h_v ** 2.0) * (10 ** (-phi_v / 10.0)))

    T_samp = len(times_in_epoch)

    # compute arrival totals and observed task counts in this epoch
    arrival_epoch_bits = {u: sum(arrivals_time[u]) for u in users}
    observed_tasks = {u: (arrival_epoch_bits[u] / D_u[u]) if D_u[u] > 0 else 0.0 for u in users}

    # decide lambda for each user this epoch:
    # - URLLC: use observed_tasks / epoch_time (Poisson empirical)
    # - e/m: use historical mean lam_emp (按题意“按平均分布”)
    lambda_epoch = {}
    for u in users:
        if u in users_by_type['U']:
            lambda_epoch[u] = observed_tasks[u] / EPOCH_S
        else:
            lambda_epoch[u] = lam_emp[u]  # tasks/s historical average

    # 为每个用户确定该 epoch 的服务基站（带偏置的最近微站规则 vs 宏站）
    BIAS_U = 3.0
    BIAS_E = 6.0
    BIAS_M = 8.0
    serving_bs = {}
    for u in users:
        if len(arrivals_time[u]) == 0:
            serving_bs[u] = 0
            continue
        utype = str(u)[0].upper()
        bias = BIAS_U if utype == 'U' else BIAS_E if utype == 'E' else BIAS_M
        # 计算平均路径损耗
        mean_phi = {}
        for b in (0,1,2,3):
            arr_phi = pl_time[b][u]
            mean_phi[b] = float(np.mean(arr_phi)) if len(arr_phi) > 0 else 200.0
        # 最近微站
        sbs_candidates = [0,1,2]
        sbs_best = min(sbs_candidates, key=lambda b: mean_phi[b])
        phi_sbs_biased = mean_phi[sbs_best] - bias
        phi_mbs = mean_phi[3]
        serving_bs[u] = sbs_best if (phi_sbs_biased <= phi_mbs) else 3

    # 预计算链路增益数组
    gains_arr = {0: {}, 1: {}, 2: {}, 3: {}}
    for b in (0, 1, 2, 3):
        for u in users:
            gains_arr[b][u] = np.array(gains_time[b][u], dtype=float)

    # 每个基站下的用户集合
    users_by_bs = {0: [], 1: [], 2: [], 3: []}
    for u in users:
        users_by_bs[serving_bs[u]].append(u)

    def build_U_table_with_power_and_load(powers_dbm_vec, load_vec):
        """根据功率和负载（RB 占用比例）计算 U_table。
        powers_dbm_vec: [P_SBS1, P_SBS2, P_SBS3, P_MBS]
        load_vec: [l_SBS1, l_SBS2, l_SBS3, l_MBS] in [0,1]
        """
        P_mW = [dbm_to_mw(p) for p in powers_dbm_vec]
        U_table_local = {u: [] for u in users}
        for u in users:
            utype = str(u)[0].upper()
            # 按类设置每用户试探 RB 上限
            if utype == 'U':
                max_r_try = min(RMAX_U, MAX_R_FOR_NOISE)
            elif utype == 'E':
                max_r_try = min(RMAX_E, MAX_R_FOR_NOISE)
            else:
                max_r_try = min(RMAX_M, MAX_R_FOR_NOISE)
            if arrival_epoch_bits.get(u, 0.0) <= 0.0:
                U_table_local[u] = [0.0] * (max_r_try + 1)
                continue
            b_serv = serving_bs[u]
            # 预取增益序列
            g_serv = gains_arr[b_serv][u]
            # 干扰仅来自其它 SBS（0,1,2）
            g_int = []
            for b in (0, 1, 2):
                if b == b_serv:
                    continue
                g_int.append(gains_arr[b][u])

            # 预计算向量化的接收/干扰功率序列
            P_sig_vec = P_mW[b_serv] * g_serv  # shape (T_samp,)
            P_int_vec = np.zeros_like(P_sig_vec)
            if b_serv in (0,1,2):
                int_idx = 0
                for b in (0, 1, 2):
                    if b == b_serv:
                        continue
                    P_int_vec += P_mW[b] * g_int[int_idx] * float(load_vec[b])
                    int_idx += 1
            else:
                # MBS 频谱不重叠，无跨站干扰
                pass

            for r in range(0, max_r_try + 1):
                if r == 0:
                    # 无 RB 分配
                    U_table_local[u].append(0.0)
                    continue
                N_vec = NOISE_MW_BY_R[r]
                denom_vec = N_vec + P_int_vec
                # 避免除零
                denom_vec = np.where(denom_vec > 0.0, denom_vec, 1e-30)
                snr_eff_vec = P_sig_vec / denom_vec
                R_t = r * B * np.log2(np.maximum(1.0 + snr_eff_vec, 1e-12))
                # 服务时间序列
                S_t = np.where(R_t > 1e-12, (D_u[u] / R_t), np.inf)

                finite_S = S_t[np.isfinite(S_t)]
                E_S = float(np.mean(finite_S)) if finite_S.size > 0 else float('inf')
                lam = lambda_epoch[u]
                if u in users_by_type['U']:
                    Wq, _ = mm1_Wq(lam, E_S)
                else:
                    Wq, _ = kingman_Wq(lam, E_S, Ca2[u], 0.0)

                if utype == 'M':
                    active_count = 0
                    success_count = 0
                    sum_L_success = 0.0
                    has_task_mask = (np.array(arrivals_time[u]) > 0)
                    if np.any(has_task_mask):
                        S_active = S_t[has_task_mask]
                        R_active = R_t[has_task_mask]
                        active_count = int(np.sum(has_task_mask))
                        L_vec = np.where(
                            np.isfinite(Wq) & np.isfinite(S_active),
                            (Wq + S_active),
                            np.inf,
                        )
                        success_mask = (np.isfinite(L_vec) & (L_vec <= t_max_M) & (R_active > 0.0))
                        success_count = int(np.sum(success_mask))
                        sum_L_success = float(np.sum(L_vec[success_mask])) if success_count > 0 else 0.0
                    if active_count > 0:
                        if success_count > 0:
                            D_avg_success = sum_L_success / float(success_count)
                        else:
                            D_avg_success = float('inf')
                        if D_avg_success <= t_max_M:
                            U_avg = float(success_count) / float(active_count)
                        else:
                            U_avg = -beta_M
                    else:
                        U_avg = 0.0
                else:
                    s_sum = 0.0
                    num_active = 0
                    has_task_mask = (np.array(arrivals_time[u]) > 0)
                    if np.any(has_task_mask):
                        R_active = R_t[has_task_mask]
                        S_active = S_t[has_task_mask]
                        L_vec = np.where(
                            np.isfinite(Wq) & np.isfinite(S_active),
                            (Wq + S_active),
                            np.inf,
                        )
                        num_active = int(np.sum(has_task_mask))
                        # 逐元素计算基础效用与惩罚
                        Qb_vec = np.array([Q_base_function(utype, R_active[i], L_vec[i]) for i in range(len(R_active))], dtype=float)
                        Pen_vec = np.where(np.isfinite(L_vec), (kappa + alpha_pen * L_vec), (kappa + alpha_pen * 1e6))
                        s_sum = float(np.sum(Qb_vec - Pen_vec))
                    U_avg = float(s_sum / num_active) if num_active > 0 else 0.0

                # 应用类公平性权重
                U_table_local[u].append(CLASS_WEIGHT.get(utype, 1.0) * U_avg)
        return U_table_local

    def precompute_class_utility_curve(U_table_local, users_in_bs, class_cap_r, bs_cap):
        """为某基站内某一类用户，预计算 f[r] = 在类内贪心分配 r 个 RB 时的最大总效用。
        通过堆合并每个用户的边际增益，得到每个 r 的前缀和曲线。
        返回长度 bs_cap+1 的数组，超过可用增量的 r 将被用最后值填充。
        """
        active_users = [u for u in users_in_bs if arrival_epoch_bits.get(u, 0.0) > 0.0]
        if len(active_users) == 0:
            return np.zeros(bs_cap + 1, dtype=float)

        # 基础值：r=0 时的总效用（通常是 0）
        base_sum = 0.0
        # 为每个用户构造边际增益序列
        import heapq
        heap = []  # max-heap by pushing negative values

        n_users = len(active_users)
        priority = {u: float(n_users - idx) / float(max(1, n_users)) for idx, u in enumerate(active_users)}

        total_possible_increments = 0
        for u in active_users:
            util_list = U_table_local[u]
            # 可分配的最大 RB 数等于列表长度-1 与类上限的较小值
            cap_u = min(class_cap_r, max(0, len(util_list) - 1))
            if len(util_list) == 0:
                continue
            base_sum += util_list[0]
            # 第一个可用边际增益来自从 0 -> 1
            for k in range(1, cap_u + 1):
                delta = util_list[k] - util_list[k - 1]
                # tie-break 调整
                delta_adj = delta + (TIE_BREAK * priority.get(u, 0.0))
                # 使用负值构建最大堆
                heapq.heappush(heap, (-delta_adj, u, k))
                total_possible_increments += 1

        # 逐步取出前 r 个最大边际，构建前缀和
        f = np.zeros(bs_cap + 1, dtype=float)
        f[0] = base_sum
        csum = 0.0
        r_limit = min(bs_cap, total_possible_increments)
        for r_take in range(1, r_limit + 1):
            if not heap:
                f[r_take] = base_sum + csum
                continue
            neg_val, u, k = heapq.heappop(heap)
            csum += (-neg_val)
            f[r_take] = base_sum + csum
        # 对于剩余 r（超过可用增量），保持最后值
        if r_limit < bs_cap:
            f[r_limit + 1:] = base_sum + csum
        return f

    def allocate_within_bs(U_table_local, users_in_bs, r_total, bs_cap):
        # 类内贪心分配，沿用原先带优先级的策略
        alloc = {u: 0 for u in users_in_bs}
        active_users = [u for u in users_in_bs if arrival_epoch_bits.get(u, 0.0) > 0.0]
        if len(active_users) == 0:
            return alloc, 0.0
        n_users = len(active_users)
        priority = {u: float(n_users - idx) / float(max(1, n_users)) for idx, u in enumerate(active_users)}
        for _ in range(r_total):
            best_delta = -1e18
            best_u = None
            for u in active_users:
                cur = alloc[u]
                # 依据 U_table 长度与类上限限制单用户 RB（避免越界导致 VERY_NEG）
                utype = str(u)[0].upper()
                table_len = len(U_table_local[u])
                # 可分配的最大 RB 数等于可索引的最后位置（table_len-1）
                class_cap = RMAX_E if utype == 'E' else RMAX_U if utype == 'U' else RMAX_M
                max_r_u = min(class_cap, bs_cap, max(0, table_len - 1))
                if cur >= max_r_u:
                    continue
                util_cur = U_table_local[u][cur] if cur < len(U_table_local[u]) else VERY_NEG
                util_next = U_table_local[u][cur + 1] if (cur + 1) < len(U_table_local[u]) else VERY_NEG
                delta = util_next - util_cur
                delta_adj = delta + (TIE_BREAK * priority.get(u, 0.0))
                if delta_adj > best_delta:
                    best_delta = delta_adj
                    best_u = u
            if best_u is None:
                break
            alloc[best_u] += 1

        class_util = sum(U_table_local[u][alloc[u]] if alloc[u] < len(U_table_local[u]) else VERY_NEG for u in users_in_bs)
        return alloc, class_util

    # 功率控制 + 负载耦合迭代
    # 功率网格：SBS ∈ [15,20,25,30]；MBS ∈ [20,30,35,40]
    grid_sbs = [15.0, 20.0, 25.0, 30.0]
    grid_mbs = [20.0, 30.0, 35.0, 40.0]
    best_epoch_util = -1e18
    best_epoch_solution = None

    for P0, P1, P2, P3 in itertools.product(grid_sbs, grid_sbs, grid_sbs, grid_mbs):
        loads = [0.3, 0.3, 0.3, 0.3]  # 初始负载猜测（更保守，降低干扰）
        for _ in range(7):  # 迭代上限
            U_table_now = build_U_table_with_power_and_load([P0, P1, P2, P3], loads)
            # 每个基站内的三类总 RB 贪心枚举
            # 为降低复杂度，沿用原策略：遍历 (rU, rE) 求 rM = bs_cap - rU - rE
            total_util_epoch = 0.0
            rU_b = [0, 0, 0, 0]
            rE_b = [0, 0, 0, 0]
            rM_b = [0, 0, 0, 0]
            util_b = [0.0, 0.0, 0.0, 0.0]

            for b in (0, 1, 2, 3):
                users_U = [u for u in users_by_bs[b] if u[0].upper() == 'U']
                users_E = [u for u in users_by_bs[b] if u[0].upper() == 'E']
                users_M = [u for u in users_by_bs[b] if u[0].upper() == 'M']

                # 预计算每类的效用曲线 f_class[r]
                bs_cap = BS_CAP[b]
                fU = precompute_class_utility_curve(U_table_now, users_U, RMAX_U, bs_cap)
                fE = precompute_class_utility_curve(U_table_now, users_E, RMAX_E, bs_cap)
                fM = precompute_class_utility_curve(U_table_now, users_M, RMAX_M, bs_cap)

                best_util_b = -1e18
                best_tuple = (0, 0, bs_cap)
                # 预留与上限：若某类存在活跃任务，为其预留最小 RB；同时限制每类最大占比
                e_active = any(arrival_epoch_bits.get(u, 0.0) > 0.0 for u in users_E)
                m_active = any(arrival_epoch_bits.get(u, 0.0) > 0.0 for u in users_M)
                u_active = any(arrival_epoch_bits.get(u, 0.0) > 0.0 for u in users_U)

                min_e_rb = int(np.ceil(MIN_E_FRACTION_PER_BS * bs_cap)) if e_active else 0
                min_m_rb = int(np.ceil(MIN_M_FRACTION_PER_BS * bs_cap)) if m_active else 0
                min_u_rb = int(np.ceil(MIN_U_FRACTION_PER_BS * bs_cap)) if u_active else 0

                # 最大上限（占比）
                max_e_rb = int(np.floor(MAX_E_FRACTION_PER_BS * bs_cap))

                # rU 外层循环：考虑 U 的最小预留
                for rU in range(min_u_rb, bs_cap + 1):
                    # 留给 E+M 的 RB 数
                    leftover_after_u = bs_cap - rU
                    if leftover_after_u < (min_e_rb + min_m_rb):
                        break

                    # rE 内层循环：考虑 E 的最小与最大；M 会拿剩余
                    rE_min = min_e_rb
                    rE_max = min(max_e_rb, leftover_after_u - min_m_rb)
                    if rE_min > rE_max:
                        continue
                    for rE in range(rE_min, rE_max + 1):
                        rM = bs_cap - rU - rE
                        if rM < min_m_rb:
                            continue
                        util_sum_b = fU[rU] + fE[rE] + fM[rM]
                        if util_sum_b > best_util_b:
                            best_util_b = util_sum_b
                            best_tuple = (rU, rE, rM)
                rU_b[b], rE_b[b], rM_b[b] = best_tuple
                util_b[b] = best_util_b
                total_util_epoch += best_util_b

            new_loads = [
                float(rU_b[0] + rE_b[0] + rM_b[0]) / float(BS_CAP[0]),
                float(rU_b[1] + rE_b[1] + rM_b[1]) / float(BS_CAP[1]),
                float(rU_b[2] + rE_b[2] + rM_b[2]) / float(BS_CAP[2]),
                float(rU_b[3] + rE_b[3] + rM_b[3]) / float(BS_CAP[3]),
            ]
            if max(abs(new_loads[i] - loads[i]) for i in range(4)) < 1e-2:
                loads = new_loads
                break
            loads = new_loads

        # 计算本组合的总能耗（按能耗模型）
        def compute_total_energy_watt(p_vec_dbm, rU, rE, rM):
            total = 0.0
            for b in (0,1,2,3):
                n_rb = int(rU[b] + rE[b] + rM[b])
                p_output_w = dbm_to_watt(p_vec_dbm[b])
                p_tx = p_output_w / ETA_PA
                total += (P_FIXED_WATT + K_RB_WATT_PER_RB * n_rb + p_tx)
            return float(total)
        combo_energy_w = compute_total_energy_watt([P0,P1,P2,P3], rU_b, rE_b, rM_b)

        # 双目标选择：先最大化效用，再在最大效用集合内最小化能耗
        UTIL_EPS = 1e-6
        if (best_epoch_solution is None) or (total_util_epoch > best_epoch_util + UTIL_EPS):
            best_epoch_util = total_util_epoch
            best_epoch_solution = {
                'powers_dbm': [P0, P1, P2, P3],
                'rU': rU_b,
                'rE': rE_b,
                'rM': rM_b,
                'util_b': util_b,
                'loads': loads,
                'serving_bs': serving_bs,
                'total_energy_watt': combo_energy_w,
            }
        elif abs(total_util_epoch - best_epoch_util) <= UTIL_EPS:
            if combo_energy_w < best_epoch_solution.get('total_energy_watt', float('inf')):
                best_epoch_solution = {
                    'powers_dbm': [P0, P1, P2, P3],
                    'rU': rU_b,
                    'rE': rE_b,
                    'rM': rM_b,
                    'util_b': util_b,
                    'loads': loads,
                    'serving_bs': serving_bs,
                    'total_energy_watt': combo_energy_w,
                }

    # 输出本 epoch 结果（逐基站）
    if best_epoch_solution is None:
        print(f"Epoch {epoch + 1}: 未找到可行解")
        for b in (0, 1, 2, 3):
            results.append({
                'epoch': epoch,
                'bs': b + 1,
                'P_dBm': P_tx_dBm,
                'rU': 0,
                'rE': 0,
                'rM': BS_CAP[b],
                'total_util_epoch': 0.0
            })
    else:
        P0, P1, P2, P3 = best_epoch_solution['powers_dbm']
        rU_b = best_epoch_solution['rU']
        rE_b = best_epoch_solution['rE']
        rM_b = best_epoch_solution['rM']
        util_b = best_epoch_solution['util_b']
        print(
            f"Epoch {epoch + 1}/{EPOCH_COUNT} [{epoch * EPOCH_MS}-{(epoch + 1) * EPOCH_MS} ms]: "
            f"SBS1 P={P0:.1f}dBm (U={rU_b[0]},E={rE_b[0]},M={rM_b[0]}), "
            f"SBS2 P={P1:.1f}dBm (U={rU_b[1]},E={rE_b[1]},M={rM_b[1]}), "
            f"SBS3 P={P2:.1f}dBm (U={rU_b[2]},E={rE_b[2]},M={rM_b[2]}), "
            f"MBS1 P={P3:.1f}dBm (U={rU_b[3]},E={rE_b[3]},M={rM_b[3]}), "
            f"epoch_util={best_epoch_util:.4f}, "
            (f"energy={best_epoch_solution.get('total_energy_watt', 0.0):.3f} W" if best_epoch_solution.get('total_energy_watt') is not None else "")
        )
        for b, P in enumerate([P0, P1, P2, P3]):
            results.append({
                'epoch': epoch,
                'bs': b + 1,
                'P_dBm': float(P),
                'rU': int(rU_b[b]),
                'rE': int(rE_b[b]),
                'rM': int(rM_b[b]),
                'total_util_epoch': float(best_epoch_util),
                'total_energy_epoch_W': float(best_epoch_solution.get('total_energy_watt', 0.0))
            })

        # 记录接入决策（用户->基站）
        if epoch == 0:
            attach_records = []
        else:
            try:
                attach_records
            except NameError:
                attach_records = []
        for u in users:
            b = int(best_epoch_solution['serving_bs'][u])
            bs_name = BS_NAMES.get(b, f'BS{b+1}')
            attach_records.append({
                'epoch': epoch,
                'user': str(u),
                'bs': b + 1,
                'bs_name': bs_name,
                'attach': f"{str(u)}:{bs_name}"
            })

# 保存结果（CSV）
out_df = pd.DataFrame(results)
out_df.to_csv("epoch_bs_allocations_and_powers.csv", index=False)
try:
    attach_df = pd.DataFrame(attach_records)
    attach_df.to_csv("epoch_user_attach_decisions.csv", index=False)
    print("已保存 epoch_user_attach_decisions.csv")
except Exception:
    pass
print("已保存 epoch_bs_allocations_and_powers.csv")