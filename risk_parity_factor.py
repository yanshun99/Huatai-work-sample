# -*- coding: utf-8 -*-
"""
宏观因子 -> 资产 的完整实现（含主观映射约束）
1) 数据：每类3只ETF，组内PCA取第1主成分(PC1)作为“公共因子”
2) 因子空间做风险平价(ERC)，得到目标因子暴露 e
3) 估计资产对因子的映射系数 A（带主观零约束），并估计特质协方差 Q
4) 解论文目标：min (Aw - e)' Σ (Aw - e) + λ w' Q w,  s.t. w≥0, 1'w=1（可选上限）
5) 月度再平衡回测，输出净值/回撤/绩效/权重

作者提示：
- 这里的“国内增长因子”来自 Growth 组（上证50、恒科、创业板）
- 我们在回归时可以约束“海外组资产”（纳指/标普/美REIT）不受 Growth 影响：A[Overseas assets, 'Growth']=0
- 你可以在 FACTOR_ALLOW_MAP 里自由增删这些假设
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import efinance as ef
import ffn

# ======================== 参数区 ========================
BEG = "2014-01-01"           # 起始日期（会自动对齐公共交易日）
RET_METHOD = "simple"        # 'simple' or 'log'
LOOKBACK = 60               # 窗口长度（天），用于PCA/回归/协方差估计
REBAL_FREQ = "M"             # 再平衡频率（这里统一按月末）
LAM = 0.5                    # 论文目标中的 λ（0~1 越大越保守）
MIN_WEIGHT = 0.0             # 单资产最小持仓（>0可去毛毛雨）
COST_BPS = 0                 # 调仓成本（bp）

# 4个宏观因子，每类3只ETF（按你提供的代码）
GROUPS = {
    "Growth":       ["510050", "513180", "159915"],   # 上证50、恒生科技、创业板
    "Financial":    ["159649", "159972", "511220"],   # 国开债、5年地债、城投债
    "Inflation":    ["518880", "159985", "165513"],   # 黄金、豆粕、全球商品LOF
    "Overseas":     ["159941", "513500"],   # 纳指100、标普500、美国REIT没找到先不包含在这
}

# 投资范围：None = 默认“分组内所有ETF均可投资”；也可自定义列表子集
INVESTABLE = None

# （可选）单资产上限：不设即不限制；示例：{"159941":0.4, "513500":0.4}
ASSET_CAPS = None

# 主观零约束：给每只“可投资资产”指定允许加载的因子列表（不在列表内的因子系数强制为0）
# 示例假设：海外资产不受“国内增长因子”影响 ⇒ 仅允许 Overseas/Inflation/Financial
# 未出现在字典里的资产默认允许加载所有因子
FACTOR_ALLOW_MAP = {
    # Overseas三只：不允许 Growth
    "159941": ["Overseas", "Inflation", "Financial"],  # 纳指100
    "513500": ["Overseas", "Inflation", "Financial"],  # 标普500
    "160140": ["Overseas", "Inflation", "Financial"],  # 美国REIT
    # 其它资产如果有明确假设，也可在此处添加
}

# ======================== 工具函数 ========================
def fetch_etf_closes(codes, beg="2010-01-01", end=None, adj=True):
    """抓多只ETF前复权收盘价；返回 DataFrame(index=日期, columns=代码)"""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    fqt = 1 if adj else 0
    frames = []
    for code in codes:
        df = ef.stock.get_quote_history(code, beg=beg.replace("-",""), end=end.replace("-",""), fqt=fqt)
        if df is None or df.empty:
            print(f"[WARN] {code} 无数据，跳过")
            continue
        df = df.sort_values("日期").copy()
        df["日期"] = pd.to_datetime(df["日期"])
        s = pd.Series(pd.to_numeric(df["收盘"], errors="coerce").values, index=df["日期"], name=code).dropna()
        frames.append(s)
    if not frames:
        raise ValueError("无可用行情数据")
    close_df = pd.concat(frames, axis=1, join="inner").sort_index()
    close_df = close_df[~close_df.index.duplicated(keep="last")]
    return close_df

def to_returns(close_df, method="simple"):
    """价格→日收益"""
    return (np.log(close_df).diff().dropna() if method=="log" else close_df.pct_change().dropna())

# 组内PCA：PC1作为公共因子；并做符号对齐（与组内平均收益正相关）
def group_pca_factors(returns, groups):
    factors = []
    for gname, members in groups.items():
        cols = [c for c in members if c in returns.columns]
        if len(cols)==0:
            print(f"[WARN] 组 {gname} 无可用成分，跳过")
            continue
        X = returns[cols].dropna()
        if X.empty:
            print(f"[WARN] 组 {gname} 在窗口内无数据，跳过")
            continue
        if len(cols)==1:
            f = X.iloc[:,0].rename(gname)
        else:
            pca = PCA(n_components=1)
            z = pca.fit_transform(X.values)[:,0]
            f = pd.Series(z, index=X.index, name=gname)
            anchor = X.mean(axis=1)
            corr = np.corrcoef(f.fillna(0), anchor.loc[f.index].fillna(0))[0,1]
            if np.isnan(corr) or corr < 0:
                f = -f
        factors.append(f)
    if not factors:
        raise ValueError("PCA因子生成失败")
    F = pd.concat(factors, axis=1, join="inner").sort_index()
    return F  # DataFrame T×G（列=因子名）

# 在因子收益上做 ERC，得到目标因子暴露 e（等风险）
def target_factor_exposure_erc(factor_returns):
    u = ffn.core.calc_erc_weights(
        returns=factor_returns, covar_method='standard',
        risk_parity_method='ccd', maximum_iterations=200, tolerance=1e-10
    ).astype(float)
    u = u.clip(lower=0); s = u.sum()
    return (u/s) if s>0 else u  # Series（索引=因子名）

# 受“允许因子集合”约束的回归：r_i = A_i f + ε_i（只在允许的因子上回归）
def estimate_A_Q_constrained(asset_rets, factor_rets, allow_map=None, ridge=1e-6):
    allow_map = allow_map or {}
    df = asset_rets.join(factor_rets, how="inner")
    Y = df[asset_rets.columns]     # T×N
    X_all = df[factor_rets.columns]# T×K
    N, K = Y.shape[1], X_all.shape[1]
    A = pd.DataFrame(0.0, index=asset_rets.columns, columns=factor_rets.columns)
    resid_cols = []
    for asset in asset_rets.columns:
        allowed = allow_map.get(asset, list(factor_rets.columns))  # 默认允许全部因子
        allowed = [f for f in allowed if f in factor_rets.columns]
        if len(allowed)==0:
            # 极端：不允许任何因子，A_i 全0，残差=自身收益
            eps_i = Y[asset].values
        else:
            X = X_all[allowed].values
            y = Y[asset].values
            beta = np.linalg.solve(X.T @ X + ridge*np.eye(X.shape[1]), X.T @ y)
            A.loc[asset, allowed] = beta
            eps_i = y - X @ beta
        resid_cols.append(eps_i)
    R = np.column_stack(resid_cols)               # T×N
    Q = np.cov(R.T)                               # N×N
    Q = pd.DataFrame(Q, index=asset_rets.columns, columns=asset_rets.columns)
    return A, Q  # A: N×K, Q: N×N

# 简单的“有上界的单纯形投影”：解 min ||w - v||^2 s.t. w≥0, w≤u, 1'w=1；二分求 τ
def project_to_capped_simplex(v, u=None, tol=1e-12, max_iter=100):
    v = np.asarray(v, float)
    n = len(v)
    if u is None:
        u = np.full(n, np.inf)
    u = np.asarray(u, float)
    low, high = -1e6, 1e6
    for _ in range(max_iter):
        tau = (low + high)/2
        w = np.clip(v - tau, 0.0, u)              # 先截断到[0,u]
        s = w.sum()
        if abs(s - 1) < tol:
            return w
        if s > 1:
            low = tau
        else:
            high = tau
    # 容忍终止
    w = np.clip(v - (low+high)/2, 0.0, u)
    s = w.sum()
    if s>0: w = w/s
    return w

# 论文目标：min (Aw-e)'Σ(Aw-e) + λ w'Qw，PGD求解（多头满仓，可选上限）
def solve_mapping_weights(A, Sigma, Q, e, lam=0.5, w0=None, max_iter=2000, tol=1e-9, step=None,
                          min_w=0.0, caps=None):
    asset_index = A.index if isinstance(A, pd.DataFrame) else pd.Index(range(A.shape[0]))
    A = A.values if isinstance(A, pd.DataFrame) else np.asarray(A, float)
    Sigma = Sigma.values if isinstance(Sigma, pd.DataFrame) else np.asarray(Sigma, float)
    Q = Q.values if isinstance(Q, pd.DataFrame) else np.asarray(Q, float)
    e = e.values if isinstance(e, pd.Series) else np.asarray(e, float)
    N = A.shape[0]
    eps = 1e-10
    Sigma = Sigma + eps*np.eye(Sigma.shape[0])
    Q = Q + eps*np.eye(Q.shape[0])
    G = A @ Sigma @ A.T + lam * Q
    g = - A @ Sigma @ e
    # 构造上界向量 u（若未给caps则全∞；若给了dict则对应资产使用上限）
    if caps is not None:
        u = np.array([caps.get(str(asset_index[i]), np.inf) for i in range(N)], float)
    else:
        u = np.full(N, np.inf)
    # 步长：幂迭代估最大特征值
    if step is None:
        try:
            x = np.ones(N)/N
            for _ in range(50):
                x = G @ x; n = np.linalg.norm(x)
                if n==0: break
                x /= n
            L = float(x @ (G @ x)); step = 1.0/(L + 1e-12)
        except Exception:
            step = 1e-2
    # 初始化：均匀并投影
    w = project_to_capped_simplex(np.ones(N)/N if w0 is None else np.asarray(w0, float), u=u)
    for _ in range(max_iter):
        grad = 2*(G @ w + g)                                # ∇J(w)
        w_new = project_to_capped_simplex(w - step*grad, u=u)
        if min_w>0:
            w_new = np.where(w_new < min_w, 0.0, w_new)
            s = w_new.sum()
            if s>0: w_new = w_new/s
        if np.linalg.norm(w_new - w, 1) < tol:
            w = w_new; break
        w = w_new
    return pd.Series(w, index=asset_index)

# 月度回测：因子ERC → A/Q（带约束）→ 求w → 固定持有到下月末
def simulate_factor_erc_mapping(all_rets, investable_codes, groups, allow_map, lam=0.5,
                                lookback=252, cost_bps=0, min_w=0.0, caps=None,
                                label="Factor-ERC Mapped"):
    asset_rets = all_rets[investable_codes]
    month_ends = asset_rets.groupby(asset_rets.index.to_period('M')).tail(1).index
    port_rets = pd.Series(index=asset_rets.index, dtype=float)
    prev_w = pd.Series(0, index=asset_rets.columns, dtype=float)
    weights_records, turnover_list, cost_list = [], [], []

    for i, t_end in enumerate(month_ends):
        end_loc = asset_rets.index.get_loc(t_end)
        if end_loc < lookback - 1:
            continue
        window_all = all_rets.iloc[end_loc - lookback + 1 : end_loc + 1]
        window_assets = window_all[investable_codes]

        # 1) 组内PCA → 4因子
        F = group_pca_factors(window_all, groups)

        # 2) A/Q（带主观约束）
        A, Q = estimate_A_Q_constrained(window_assets, F, allow_map=allow_map, ridge=1e-6)

        # 3) 因子协方差 Σ_f
        Sigma = pd.DataFrame(np.cov(F.T), index=F.columns, columns=F.columns)

        # 4) 因子ERC → 目标因子暴露 e
        e = target_factor_exposure_erc(F)

        # 5) 解优化 → w
        w = solve_mapping_weights(A, Sigma, Q, e, lam=lam, min_w=min_w, caps=caps)

        # 6) 应用权重到 (本月末, 下月末]
        if i == len(month_ends) - 1:
            idx_slice = asset_rets.index[end_loc + 1:]
        else:
            next_end_loc = asset_rets.index.get_loc(month_ends[i + 1])
            idx_slice = asset_rets.index[end_loc + 1 : next_end_loc + 1]
        if len(idx_slice)==0:
            prev_w = w; continue

        turnover = float((w - prev_w).abs().sum())
        cost = (cost_bps/10000.0) * turnover
        turnover_list.append(turnover); cost_list.append(cost)

        first_day = idx_slice[0]
        port_rets.loc[first_day] = asset_rets.loc[first_day].mul(w, axis=0).sum() - cost
        if len(idx_slice) > 1:
            daily = asset_rets.loc[idx_slice[1:]].mul(w, axis=1).sum(axis=1)
            port_rets.loc[idx_slice[1:]] = daily.values

        weights_records.append((t_end, w))
        prev_w = w

    port_rets = port_rets.dropna()
    nav = (1 + port_rets).cumprod()
    weights_df = pd.DataFrame({d: w for d, w in weights_records}).T
    weights_df.index.name = "rebalance_date"

    ann = (1 + port_rets).prod() ** (252/len(port_rets)) - 1 if len(port_rets)>0 else np.nan
    vol = port_rets.std()*np.sqrt(252) if len(port_rets)>1 else np.nan
    sharpe = 0 if (vol is None or vol==0 or np.isnan(vol)) else ann/vol
    mdd = ((nav / nav.cummax()) - 1).min() if len(nav)>0 else np.nan
    stats = {
        "label": label, "annual_return": ann, "annual_vol": vol,
        "sharpe": sharpe, "max_drawdown": mdd,
        "avg_monthly_turnover": np.mean(turnover_list) if turnover_list else 0.0,
        "avg_monthly_cost": np.mean(cost_list) if cost_list else 0.0,
        "last_nav": nav.iloc[-1] if len(nav)>0 else np.nan
    }
    return port_rets, nav, weights_df, stats

# 等权基准：在“可投资集合”里做月度等权
def simulate_equal_weight(asset_rets, cost_bps=0, label="EW (Investable)"):
    month_ends = asset_rets.groupby(asset_rets.index.to_period('M')).tail(1).index
    port_rets = pd.Series(index=asset_rets.index, dtype=float)
    prev_w = pd.Series(0, index=asset_rets.columns, dtype=float)
    weights_records, turnover_list, cost_list = [], [], []
    for i, t_end in enumerate(month_ends):
        w = pd.Series(1.0/asset_rets.shape[1], index=asset_rets.columns)
        end_loc = asset_rets.index.get_loc(t_end)
        if i == len(month_ends) - 1:
            idx_slice = asset_rets.index[end_loc + 1:]
        else:
            next_end_loc = asset_rets.index.get_loc(month_ends[i + 1])
            idx_slice = asset_rets.index[end_loc + 1: next_end_loc + 1]
        if len(idx_slice)==0:
            prev_w = w; continue
        turnover = float((w - prev_w).abs().sum()); cost = (cost_bps/10000.0)*turnover
        turnover_list.append(turnover); cost_list.append(cost)
        first_day = idx_slice[0]
        port_rets.loc[first_day] = asset_rets.loc[first_day].mul(w, axis=0).sum() - cost
        if len(idx_slice)>1:
            daily = asset_rets.loc[idx_slice[1:]].mul(w, axis=1).sum(axis=1)
            port_rets.loc[idx_slice[1:]] = daily.values
        prev_w = w; weights_records.append((t_end, w))
    port_rets = port_rets.dropna(); nav = (1 + port_rets).cumprod()
    weights_df = pd.DataFrame({d: w for d, w in weights_records}).T; weights_df.index.name="rebalance_date"
    ann = (1 + port_rets).prod() ** (252/len(port_rets)) - 1 if len(port_rets)>0 else np.nan
    vol = port_rets.std()*np.sqrt(252) if len(port_rets)>1 else np.nan
    sharpe = 0 if (vol is None or vol==0 or np.isnan(vol)) else ann/vol
    mdd = ((nav/nav.cummax()) - 1).min() if len(nav)>0 else np.nan
    stats = {"label": label, "annual_return": ann, "annual_vol": vol,
             "sharpe": sharpe, "max_drawdown": mdd, "last_nav": nav.iloc[-1]}
    return port_rets, nav, weights_df, stats

# ======================== 主程序 ========================
if __name__ == "__main__":
    # 分组里的所有ETF（去重）
    group_codes = sorted({c for lst in GROUPS.values() for c in lst})

    # 可投资集合：如果 INVESTABLE=None → 默认“分组内所有ETF”；否则用你指定的列表
    investable_codes = group_codes if INVESTABLE is None else list(dict.fromkeys(INVESTABLE))

    # 需要抓数的全部代码
    all_codes = sorted(set(group_codes + investable_codes))

    # 1) 拉价→收益（公共交易日对齐）
    close_df = fetch_etf_closes(all_codes, beg=BEG, adj=True)
    rets_all = to_returns(close_df, RET_METHOD)
    print("公共起始交易日：", rets_all.index[0].date())

    # 2) 因子ERC→资产映射（带主观A约束）
    fm_rets, fm_nav, fm_wts, fm_stats = simulate_factor_erc_mapping(
        rets_all, investable_codes, GROUPS, allow_map=FACTOR_ALLOW_MAP,
        lam=LAM, lookback=LOOKBACK, cost_bps=COST_BPS, min_w=MIN_WEIGHT,
        caps=ASSET_CAPS, label=f"Factor-ERC Mapped (λ={LAM})"
    )

    # 3) 等权对照
    ew_rets, ew_nav, ew_wts, ew_stats = simulate_equal_weight(rets_all[investable_codes], cost_bps=0)

    # 4) 参考基准（可选：若存在）
    baselines = []
    if "510300" in rets_all.columns:
        baselines.append((1 + rets_all["510300"]).cumprod().rename("510300 (HS300 B&H)"))
    if "159941" in rets_all.columns:
        baselines.append((1 + rets_all["159941"]).cumprod().rename("159941 (Nas100 B&H)"))

    # 5) 画图&导出
    nav_list = [
        fm_nav.rename("Factor-ERC Mapped"),
        ew_nav.rename("EW (Investable)"),
        *baselines
    ]
    nav_df = pd.concat(nav_list, axis=1).dropna()

    plt.figure(figsize=(10,6))
    (nav_df / nav_df.iloc[0]).plot(ax=plt.gca())
    plt.title("净值对比（起点归一）"); plt.ylabel("净值(归一)"); plt.xlabel("日期")
    plt.legend(); plt.tight_layout(); plt.savefig("nav_compare_factor_erc_full.png", dpi=150)

    dd_df = nav_df / nav_df.cummax() - 1
    plt.figure(figsize=(10,4))
    dd_df.plot(ax=plt.gca())
    plt.title("回撤对比"); plt.ylabel("回撤"); plt.xlabel("日期")
    plt.legend(); plt.tight_layout(); plt.savefig("drawdown_compare_factor_erc_full.png", dpi=150)

    # 绩效汇总
    def stats_from_nav(nav, label):
        pr = nav.pct_change().dropna()
        ann = (1 + pr).prod() ** (252/len(pr)) - 1
        vol = pr.std()*np.sqrt(252)
        sharpe = 0 if vol==0 else ann/vol
        mdd = ((nav/nav.cummax()) - 1).min()
        return {"label": label, "annual_return": ann, "annual_vol": vol,
                "sharpe": sharpe, "max_drawdown": mdd, "last_nav": nav.iloc[-1]}

    rows = [fm_stats, ew_stats] + [stats_from_nav(s, s.name) for s in baselines]
    stats_df = pd.DataFrame(rows).set_index("label")

    nav_df.to_csv("nav_compare_factor_erc_full.csv", encoding="utf-8-sig")
    dd_df.to_csv("drawdown_compare_factor_erc_full.csv", encoding="utf-8-sig")
    fm_wts.to_csv("factor_erc_monthly_weights_full.csv", encoding="utf-8-sig")

    print("\n=== 绩效汇总（年化/波动/夏普/最大回撤/换手/成本/最终净值） ===")
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(stats_df)
    print("\n文件已保存：nav_compare_factor_erc_full.csv, drawdown_compare_factor_erc_full.csv, "
          "factor_erc_monthly_weights_full.csv, nav_compare_factor_erc_full.png, drawdown_compare_factor_erc_full.png")
