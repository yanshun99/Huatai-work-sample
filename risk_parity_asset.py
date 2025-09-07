#risk_parity_local.py —— 4只ETF的风险平价 vs 市场基准 对比（逐行注释版）

import pandas as pd            # 处理表格数据
import numpy as np             # 数值计算
import matplotlib.pyplot as plt# 画图
import efinance as ef          # 拉取A股/ETF历史数据
import ffn                     # 里面有 calc_erc_weights，帮我们算等风险贡献(ERC)权重

# ========= 参数区（可以按需改） =========
CODES = ["511260", "518880", "510300", "159941"]  # 我们的4只ETF：十年国债/黄金/沪深300/纳指100
BEG   = "2014-01-01"                              # 开始日期（设置早一点没关系，后面会按公共起点对齐）
LOOKBACK = 60                                     # 看最近60个交易日来估计协方差并计算ERC权重
COVAR_METHOD = "ledoit-wolf"                      # 协方差估计方法：Ledoit–Wolf收缩，更稳
CASH_BUFFER = 0.995                               # 权重整体乘0.995，留0.5%现金缓冲（避免满仓导致资金不够）
COST_BPS_RP = 0                                   # 风险平价组合每次调仓成本（基点，1bp=0.01%），这里先设0
COST_BPS_EW = 0                                   # 等权组合每次调仓成本（基点），先设0
RET_METHOD = "simple"                             # 收益率口径：'simple'算术收益，或'log'对数收益（这里用simple）

# ========= 拉取收盘价（前复权） =========
def fetch_etf_closes(codes, beg="2010-01-01", end=None, adj=True):
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")       # 如果没给end，就用今天
    fqt = 1 if adj else 0                                        # efinance的复权参数：1=前复权，0=不复权
    frames = []                                                  # 存每只ETF的一列(Series)

    for code in codes:                                           # 逐只ETF拉数据
        df = ef.stock.get_quote_history(                         # 拉历史行情
            code, beg=beg.replace("-", ""),                      # efinance要不带“-”的日期
            end=end.replace("-", ""), fqt=fqt                    # fqt控制是否复权
        )
        if df is None or df.empty:                               # 如果这只ETF没数据
            raise ValueError(f"{code} 无数据")

        df = df.sort_values('日期').copy()                        # 按日期升序
        df['日期'] = pd.to_datetime(df['日期'])                   # 把“日期”列转成时间类型
        close = pd.to_numeric(df['收盘'], errors='coerce').dropna()# “收盘”转成数字，并丢掉无法转换的
        s = pd.Series(close.values, index=df['日期'], name=code)  # 做成一列：索引=日期，值=收盘价，列名=代码
        frames.append(s)                                         # 收集起来

    close_df = pd.concat(frames, axis=1, join="inner").sort_index() # 横向拼接：只保留大家都同时有数据的日期（内连接）
    close_df = close_df[~close_df.index.duplicated(keep='last')]    # 防重复索引
    return close_df                                                # 返回“日期×代码”的价格矩阵

# ========= 价格→日收益 =========
def to_returns(close_df, method="simple"):
    if method == "log":                                          # 对数收益：log(P_t)-log(P_{t-1})
        return np.log(close_df).diff().dropna()
    return close_df.pct_change().dropna()                        # 算术收益：(P_t/P_{t-1}-1)

# ========= 用ffn算ERC权重 =========
def erc_weights(returns_window, covar_method="ledoit-wolf",
                cash_buffer=CASH_BUFFER, min_w=0.0):
    """
    returns_window: 近LOOKBACK天的收益率子表（行=时间，列=资产）
    返回：权重Series，索引=资产代码，和≈cash_buffer（比如0.995）
    """
    try:
        w = ffn.core.calc_erc_weights(                           # 核心：按协方差解等风险贡献权重
            returns=returns_window,
            covar_method=covar_method,                           # 'ledoit-wolf'更稳，或'standard'
            risk_parity_method='ccd',                            # 求解器（循环坐标下降）
            maximum_iterations=100, tolerance=1e-8               # 收敛控制
        ).astype(float)

        w = w.clip(lower=0)                                      # 保守：不允许负权重
        s = w.sum()
        w = (w / s) if s > 0 else w                              # 先归一化到和=1
        w = w * cash_buffer                                      # 乘以缓冲（比如0.995），留一点现金
        if min_w > 0:                                            # 可选：小权重截断，减少“毛毛雨”调仓
            w = w.where(w >= min_w, 0.0)
            s2 = w.sum()
            if s2 > 0:
                w = w * (cash_buffer / s2)                       # 再归一化回 cash_buffer
        return w
    except Exception as e:                                       # 如果ERC失败（数据不稳等）
        w = pd.Series(1.0 / returns_window.shape[1],             # 回退：等权
                      index=returns_window.columns) * cash_buffer
        print(f"[ERC失败回退等权] {e}")
        return w

# ========= 等权权重（基准用） =========
def ew_weights(columns, cash_buffer=1.0):
    n = len(columns)                                             # 资产数量
    if n == 0:
        return pd.Series(dtype=float)
    w = pd.Series(1.0/n, index=columns)                          # 每个1/n
    return w * cash_buffer                                       # 乘缓冲（等权基准通常=1.0，不留现金）

# ========= 通用“月度再平衡”模拟器 =========
def simulate_monthly(rets, weight_fn, lookback=None,
                     cost_bps=0, label="portfolio"):
    """
    rets: 日收益矩阵（行=日期，列=资产）
    weight_fn: 给窗口数据返回权重的函数（如ERC）；如果lookback=None，表示不需要窗口（如等权）
    lookback: 计算权重用的窗口长度（天数）；None=不用窗口
    cost_bps: 每次调仓成本（基点），用“权重变动的绝对值和×费率”扣在第一天
    label: 名称（用于报表）
    """
    month_ends = rets.groupby(rets.index.to_period('M')).tail(1).index  # 每个月最后一个交易日
    port_rets = pd.Series(index=rets.index, dtype=float)                # 组合的“日收益”序列
    weights_records = []                                                # 记录每个月的权重
    prev_w = pd.Series(0, index=rets.columns, dtype=float)              # 上一期权重（初始空仓）
    turnover_list, cost_list = [], []                                   # 记录换手与成本

    for i, t_end in enumerate(month_ends):                   # 遍历每个“月末”
        end_loc = rets.index.get_loc(t_end)                  # 月末在索引中的位置
        if lookback is not None and end_loc < lookback - 1:  # 如果历史天数不够窗口长度
            continue                                         # 跳过，直到够为止

        window = (rets.iloc[end_loc - lookback + 1: end_loc + 1]
                  if lookback is not None else None)         # 取“近LOOKBACK天”的子表
        # 根据窗口拿权重：ERC用窗口，等权不需要
        w = (weight_fn(window, rets.columns)
             if lookback is not None else weight_fn(None, rets.columns))

        # —— 换手与成本（把权重变化的绝对值求和，理解为双边换手比例）——
        turnover = float((w - prev_w).abs().sum())           # 总换手比例（买+卖）
        cost = (cost_bps / 10000.0) * turnover               # 基点→比例，算一次性成本
        turnover_list.append(turnover)                       # 记账：本次换手
        cost_list.append(cost)                               # 记账：本次成本

        # —— 权重的应用区间：(本月月末, 下月月末] —— 从下一个交易日起开始生效，持有到下一个月末
        if i == len(month_ends) - 1:                         # 如果是最后一个月末
            idx_slice = rets.index[end_loc + 1:]             # 生效到数据结尾
        else:
            next_end_loc = rets.index.get_loc(month_ends[i + 1])     # 下一个月末的位置
            idx_slice = rets.index[end_loc + 1: next_end_loc + 1]    # (本月末, 下月末]

        if len(idx_slice) == 0:                              # 如果没有生效日（极少见）
            prev_w = w
            continue

        # —— 把成本扣在“生效的第一天” ——（简单可行的近似）
        first_day = idx_slice[0]                             # 生效的第一天
        port_rets.loc[first_day] = rets.loc[first_day].mul(w, axis=0).sum() - cost
        # 其余生效日：按固定权重计算每天的组合收益（不再调仓）
        if len(idx_slice) > 1:
            daily = rets.loc[idx_slice[1:]].mul(w, axis=1).sum(axis=1)  # 按列名对齐，更安全
            port_rets.loc[idx_slice[1:]] = daily.values

        weights_records.append((t_end, w))                   # 记录当月末的权重（用于输出表格）
        prev_w = w                                           # 更新“上期权重”，下一轮用

    port_rets = port_rets.dropna()                           # 去掉未填的日期
    nav = (1 + port_rets).cumprod()                          # 净值曲线：把日收益累乘

    # —— 绩效指标（年化收益/波动/夏普/最大回撤）——
    ann = (1 + port_rets).prod() ** (252 / len(port_rets)) - 1 if len(port_rets)>0 else np.nan
    vol = port_rets.std() * np.sqrt(252) if len(port_rets)>1 else np.nan
    sharpe = 0 if (vol is None or vol==0 or np.isnan(vol)) else ann / vol
    roll_max = nav.cummax() if len(nav)>0 else nav
    mdd = ((nav / roll_max) - 1).min() if len(nav)>0 else np.nan

    # —— 汇总“每月权重表”：行=再平衡日，列=资产代码，值=当期权重 —— 
    weights_df = pd.DataFrame({d: w for d, w in weights_records}).T
    weights_df.index.name = "rebalance_date"

    # —— 打包统计 —— 
    stats = {
        "label": label,
        "annual_return": ann,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "avg_monthly_turnover": np.mean(turnover_list) if turnover_list else 0.0,
        "avg_monthly_cost": np.mean(cost_list) if cost_list else 0.0,
        "last_nav": nav.iloc[-1] if len(nav)>0 else np.nan
    }
    return port_rets, nav, weights_df, stats                 # 返回：组合日收益、净值曲线、月度权重表、指标

# ========= 主流程（跑对比） =========
if __name__ == "__main__":
    close_df = fetch_etf_closes(CODES, beg=BEG, adj=True)    # 拉数据（前复权收盘）
    rets = to_returns(close_df, RET_METHOD)                  # 变成日收益
    print("四只ETF的共同起始日期：", close_df.index[0].date())  # 实际的“公共起点”（信息提示）

    # 1) 我们的“风险平价”组合（ERC，月度再平衡，窗口LOOKBACK）
    rp_weight_fn = lambda window, cols: erc_weights(         # 定义一个“给窗口→返回权重”的函数
        window, COVAR_METHOD, cash_buffer=CASH_BUFFER, min_w=0.0
    )
    rp_rets, rp_nav, rp_wts, rp_stats = simulate_monthly(    # 跑月度再平衡模拟
        rets, rp_weight_fn, lookback=LOOKBACK,
        cost_bps=COST_BPS_RP, label="ERC Risk Parity"
    )

    # 2) 四资产“等权”基准（每月再平衡）
    ew_weight_fn = lambda window, cols: ew_weights(cols, cash_buffer=1.0) # 等权：不留现金
    ew_rets, ew_nav, ew_wts, ew_stats = simulate_monthly(
        rets, ew_weight_fn, lookback=None,                 # 等权不需要窗口
        cost_bps=COST_BPS_EW, label="EW 4 Assets (Monthly)"
    )

    # 3) 单资产买入持有基准：510300（沪深300）、159941（纳指100）
    hs300_nav = (1 + rets["510300"]).cumprod()             # 沪深300买入持有的净值
    nas100_nav = (1 + rets["159941"]).cumprod()            # 纳指100买入持有的净值

    # —— 把几条净值合在一起，方便画图/导出 —— 
    nav_df = pd.concat([
        rp_nav.rename("ERC Risk Parity"),
        ew_nav.rename("EW 4 Assets (Monthly)"),
        hs300_nav.rename("510300 (HS300 B&H)"),
        nas100_nav.rename("159941 (Nas100 B&H)"),
    ], axis=1).dropna()

    # —— 绩效表（把单资产基准也做成一行指标）——
    def stats_from_nav(nav, label):
        pr = nav.pct_change().dropna()
        ann = (1 + pr).prod() ** (252 / len(pr)) - 1
        vol = pr.std() * np.sqrt(252)
        sharpe = 0 if vol==0 else ann/vol
        mdd = ((nav / nav.cummax()) - 1).min()
        return {"label": label, "annual_return": ann, "annual_vol": vol,
                "sharpe": sharpe, "max_drawdown": mdd, "last_nav": nav.iloc[-1]}

    stats_rows = [
        rp_stats,
        ew_stats,
        stats_from_nav(hs300_nav, "510300 (HS300 B&H)"),
        stats_from_nav(nas100_nav, "159941 (Nas100 B&H)"),
    ]
    stats_df = pd.DataFrame(stats_rows).set_index("label")  # 指标汇总表

    # ====== 画图：净值 & 回撤 ======
    plt.figure(figsize=(10,6))                              # 设图大小
    (nav_df / nav_df.iloc[0]).plot(ax=plt.gca())            # 把起点都归一到1，便于对比
    plt.title("净值对比（起点归一）")
    plt.ylabel("净值(归一)")
    plt.xlabel("日期")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nav_compare.png", dpi=150)                 # 存图：净值对比

    dd_df = nav_df / nav_df.cummax() - 1                    # 计算各策略的回撤曲线
    plt.figure(figsize=(10,4))
    dd_df.plot(ax=plt.gca())
    plt.title("回撤对比")
    plt.ylabel("回撤")
    plt.xlabel("日期")
    plt.legend()
    plt.tight_layout()
    plt.savefig("drawdown_compare.png", dpi=150)            # 存图：回撤对比

    # ====== 输出结果 ======
    print("\n=== 绩效汇总（年化/波动/夏普/最大回撤/换手/成本/最终净值） ===")
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(stats_df)

    nav_df.to_csv("nav_compare.csv", encoding="utf-8-sig")  # 输出净值表（CSV）
    dd_df.to_csv("drawdown_compare.csv", encoding="utf-8-sig") # 输出回撤表（CSV）
    rp_wts.to_csv("rp_monthly_weights.csv", encoding="utf-8-sig") # 输出风险平价的月度权重

    print("\n文件已保存：nav_compare.csv, drawdown_compare.csv, "
          "rp_monthly_weights.csv, nav_compare.png, drawdown_compare.png")
