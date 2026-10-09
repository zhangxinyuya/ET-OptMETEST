import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# ============================================================
# 配置区域 - 根据需要修改
# ============================================================
EXCEL_LIT_FILE    = "Cgproduct.xlsx"
ET_ECOMP_BASE_DIR = "./result_ET_EComp"
ET_FSEOF_BASE_DIR = "./result_ET_FSEOF"
ECFBRIDGE_BASE_DIR = "./ecfb_results"
OUTPUT_DIR        = "./Fig_results"

LITERATURE_COL = "Gene in 14067"
PRODUCT_COL    = "Product"
STRATEGY_COL   = "Strategy"

CPD_FULLNAME_MAP = {}

# ============================================================
# 工具函数
# ============================================================

def extract_cpd_code(text):
    match = re.search(r'cpd(\d{5})', str(text).lower())
    return f"cpd{match.group(1)}" if match else None


def normalize_strategy(s):
    if pd.isna(s):
        return None
    s = str(s).strip().upper()
    if s in ['OE', 'OVEREXPRESSION', 'UP', 'AMPLIFICATION', 'OVEREXPR']:
        return 'OE'
    if s in ['KD', 'KO', 'KNOCKOUT', 'KNOCKDOWN', 'DOWN', 'ATTENUATION', 'DELETION']:
        return 'KD'
    return None


def split_genes(gene_str):
    if pd.isna(gene_str):
        return []
    gene_str = str(gene_str).strip()
    parts = re.split(r'\s+and\s+|;|,', gene_str, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


# ============================================================
# 读取文献数据
# ============================================================

def load_literature(excel_file):
    df = pd.read_excel(excel_file, sheet_name=0, header=0)
    df[PRODUCT_COL] = df[PRODUCT_COL].fillna(method='ffill')

    lit_dict = {}
    cpd_fullname = {}

    for product_name, grp in df.groupby(PRODUCT_COL):
        product_name = str(product_name).strip()
        cpd = extract_cpd_code(product_name)
        if cpd is None:
            continue

        cpd_fullname[cpd] = product_name
        targets = set()

        for _, row in grp.iterrows():
            gene_raw  = row.get(LITERATURE_COL)
            strat_raw = row.get(STRATEGY_COL)
            strat = normalize_strategy(strat_raw)
            if strat is None:
                continue
            for g in split_genes(gene_raw):
                targets.add((g, strat))

        if targets:
            lit_dict[cpd] = {'full_name': product_name, 'targets': targets}

    print(f"[文献] 共读取 {len(lit_dict)} 个产品，CPD编号: {sorted(lit_dict.keys())}")
    return lit_dict, cpd_fullname


# ============================================================
# 读取 ET-EComp
# ============================================================

def load_et_targets(base_dir, method_name="ET-EComp"):
    result = {}
    pattern = os.path.join(base_dir, "EX_cpd*_e", "results.xlsx")
    files = glob.glob(pattern)

    if not files:
        print(f"[{method_name}] 未找到文件，路径模式: {pattern}")
        return result

    for fpath in sorted(files):
        folder = Path(fpath).parent.name
        cpd = extract_cpd_code(folder)
        if cpd is None:
            continue

        try:
            xls = pd.ExcelFile(fpath)
            sheet_name = None
            for s in xls.sheet_names:
                if 'must' in s.lower():
                    sheet_name = s
                    break
            if sheet_name is None:
                sheet_name = 'MUST'
            df = pd.read_excel(fpath, sheet_name=sheet_name)
        except Exception as e:
            print(f"[{method_name}] 读取失败 {fpath}: {e}")
            continue

        gene_col = manip_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'gene' in col_lower and gene_col is None:
                gene_col = col
            if 'manipulation' in col_lower:
                manip_col = col

        if gene_col is None or manip_col is None:
            print(f"[{method_name}] 无法识别列名 {fpath}，列: {list(df.columns)}")
            continue

        targets = set()
        for _, row in df.iterrows():
            manip = str(row[manip_col]).strip() if pd.notna(row[manip_col]) else ''
            if manip.lower() in ['up', 'oe', 'overexpression']:
                strat = 'OE'
            elif manip.lower() in ['down', 'kd', 'ko', 'knockout', 'knockdown']:
                strat = 'KD'
            else:
                continue
            for g in split_genes(row[gene_col]):
                targets.add((g, strat))

        result[cpd] = targets
        print(f"[{method_name}] {cpd}: {len(targets)} 个靶点")

    return result


# ============================================================
# 读取 ET-FSEOF
# ============================================================

def load_et_fseof_targets(base_dir):
    method_name = "ET-FSEOF"
    result = {}
    pattern = os.path.join(base_dir, "EX_cpd*_e", "results.xlsx")
    files = glob.glob(pattern)

    if not files:
        print(f"[{method_name}] 未找到文件，路径模式: {pattern}")
        return result

    for fpath in sorted(files):
        folder = Path(fpath).parent.name
        cpd = extract_cpd_code(folder)
        if cpd is None:
            continue

        try:
            xls = pd.ExcelFile(fpath)
            sheet_name = None
            for s in xls.sheet_names:
                if s.lower() == 'test':
                    sheet_name = s
                    break
            if sheet_name is None:
                print(f"[{method_name}] 未找到 'test' sheet，跳过 {fpath}，可用sheets: {xls.sheet_names}")
                continue
            df = pd.read_excel(fpath, sheet_name=sheet_name, index_col=1)
        except Exception as e:
            print(f"[{method_name}] 读取失败 {fpath}: {e}")
            continue

        manip_col = None
        for col in df.columns:
            if 'manipulation' in str(col).lower():
                manip_col = col
                break

        if manip_col is None:
            print(f"[{method_name}] 未找到 'manipulations' 列，跳过 {fpath}，列: {list(df.columns)}")
            continue

        targets = set()
        for gene_name, row in df.iterrows():
            gene_name = str(gene_name).strip()
            if not gene_name:
                continue
            manip = str(row[manip_col]).strip().lower() if pd.notna(row[manip_col]) else ''
            if manip in ['up', 'oe', 'overexpression']:
                strat = 'OE'
            elif manip in ['down', 'kd', 'ko', 'knockout', 'knockdown']:
                strat = 'KD'
            else:
                continue
            targets.add((gene_name, strat))

        result[cpd] = targets
        print(f"[{method_name}] {cpd}: {len(targets)} 个靶点")

    return result


# ============================================================
# 读取 ecFbridge（新目录结构: base_dir/batch_*/cpd*/*_singletest_filtered.csv）
# ============================================================

def load_ecfbridge_targets(base_dir, improvement_threshold=0.1):
    result = {}
    pattern = os.path.join(base_dir, "batch_*", "cpd*", "*_singletest_filtered.csv")
    files = glob.glob(pattern)

    if not files:
        # 兜底：尝试直接搜索所有子目录中的 singletest_filtered 文件
        pattern_fallback = os.path.join(base_dir, "**", "*_singletest_filtered.csv")
        files = glob.glob(pattern_fallback, recursive=True)

    if not files:
        print(f"[ecFbridge] 未找到文件，路径模式: {pattern}")
        return result

    for fpath in sorted(files):
        # 优先从父目录名提取 cpd 编号（目录名即 cpd*）
        cpd = extract_cpd_code(Path(fpath).parent.name)
        if cpd is None:
            # 兜底：从文件名提取
            cpd = extract_cpd_code(Path(fpath).name)
        if cpd is None:
            continue

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"[ecFbridge] 读取失败 {fpath}: {e}")
            continue

        # 如果存在 improvement 列，进行阈值过滤
        if "production_improvement" in df.columns and "yield_improvement" in df.columns:
            df = df[(df["production_improvement"] > improvement_threshold) |
                    (df["yield_improvement"] > improvement_threshold)]

        # 自动识别 gene 列和 strategy/type 列
        gene_col = type_col = None
        for col in df.columns:
            cl = col.lower()
            if any(k in cl for k in ['gene_id', 'gene', 'target']) and gene_col is None:
                gene_col = col
            if any(k in cl for k in ['type', 'strategy', 'mod']) and type_col is None:
                type_col = col

        if gene_col is None:
            gene_col = df.columns[0]

        targets = set()
        for _, row in df.iterrows():
            gene_raw = row.get(gene_col)
            strat = normalize_strategy(row.get(type_col)) if type_col else None
            if strat is None:
                continue
            for g in split_genes(gene_raw):
                targets.add((g, strat))

        result[cpd] = targets
        print(f"[ecFbridge] {cpd}: {len(targets)} 个靶点 ({Path(fpath).name})")

    return result


# ============================================================
# 计算性能指标
# ============================================================

def compute_hit_rates(lit_dict, pred_dict_list, method_names):
    all_cpds = set(lit_dict.keys())
    for pred in pred_dict_list:
        all_cpds &= set(pred.keys())
    all_cpds = sorted(all_cpds)

    if not all_cpds:
        print("[警告] 没有共同的CPD产品")
        print("文献CPD:", sorted(lit_dict.keys()))
        for name, pred in zip(method_names, pred_dict_list):
            print(f"{name} CPD:", sorted(pred.keys()))
        return pd.DataFrame(), pd.DataFrame(), None

    print(f"\n[评估] 公共产品数量: {len(all_cpds)}, CPD: {all_cpds}")

    hit_rate_data  = {name: [] for name in method_names}
    coverage_data  = {name: [] for name in method_names}
    counts_data    = {name: [] for name in method_names}
    product_labels = []

    for cpd in all_cpds:
        lit_targets = lit_dict[cpd]['targets']
        full_name   = lit_dict[cpd]['full_name']
        label = re.sub(r'\s*\(cpd\d{5}\)', '', full_name).strip()
        product_labels.append(label)

        for name, pred in zip(method_names, pred_dict_list):
            pred_targets = pred.get(cpd, set())
            tp     = len(lit_targets & pred_targets)
            n_pred = len(pred_targets)
            n_lit  = len(lit_targets)

            hit_rate = (tp / n_pred * 100) if n_pred > 0 else 0.0
            coverage = (tp / n_lit  * 100) if n_lit  > 0 else 0.0

            hit_rate_data[name].append(round(hit_rate, 2))
            coverage_data[name].append(round(coverage, 2))
            counts_data[name].append(n_pred)

    df_hit    = pd.DataFrame(hit_rate_data, index=product_labels)
    df_cov    = pd.DataFrame(coverage_data, index=product_labels)
    counts_mat = np.array([counts_data[n] for n in method_names]).T

    return df_hit, df_cov, counts_mat


# ============================================================
# 合并组图：左侧柱状图 + 右侧两个箱线图（横向排列）
# ============================================================

def plot_combined_figure(df_hit, df_cov, counts_mat, method_names, product_labels, save_path):
    """
    横向组图：[柱状图] | [Hit Rate 箱线图] | [Coverage 箱线图]
    图例位于第一个箱线图内部左上角。
    """
    n_methods  = len(method_names)
    n_products = len(product_labels)

    palette = ['#4C72B0', '#55A868', '#C44E52', '#8172B2','#64B5CD']
    colors  = palette[:n_methods]

    # 3列布局，constrained_layout 自动处理间距
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(20, 6.5),
        constrained_layout=True
    )
    fig.patch.set_facecolor('white')

    # ── 左：柱状图 ──────────────────────────────────────────────────────────
    mean_counts = counts_mat.mean(axis=0)
    std_counts  = counts_mat.std(axis=0)
    x = np.arange(n_methods)

    bars = ax1.bar(x, mean_counts, color=colors, width=0.6,
                   alpha=0.82, edgecolor='white', linewidth=1.5, zorder=3)
    ax1.errorbar(x, mean_counts, yerr=std_counts,
                 fmt='none', color='#333333', capsize=6,
                 capthick=1.8, elinewidth=1.8, zorder=4)

    for bar, val, sd in zip(bars, mean_counts, std_counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + sd + max(mean_counts) * 0.015,
            f'{val:.1f}',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#333333'
        )

    for j in range(n_methods):
        jitter_x = np.random.normal(j, 0.06, size=n_products)
        ax1.scatter(jitter_x, counts_mat[:, j],
                    alpha=0.60, s=28, color='#2C3E50',
                    edgecolors='white', linewidth=0.5, zorder=5)

    y_top = (mean_counts + std_counts).max() * 1.3
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, fontsize=15, fontweight='bold',
                        rotation=30, ha='right')
    ax1.set_ylabel(f'Number of Predicted Targets',
                   fontsize=11, fontweight='bold')
    ax1.set_xlabel('Method', fontsize=15, fontweight='bold')
    ax1.set_title('Average Predicted\nTarget Count per Method',
                  fontsize=15, fontweight='bold', pad=10)
    ax1.set_ylim(0, max(y_top, 5))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_facecolor('white')

    # ── 右两列：箱线图 ───────────────────────────────────────────────────────
    box_configs = [
        (ax2, df_hit, 'Ratio of Validated Targets (%)',
         'Hit Rate Distribution'),
        (ax3, df_cov, 'Literature Coverage (%)',
         'Coverage Rate Distribution'),
    ]

    for ax, df, ylabel, title in box_configs:
        data_by_method = [df[col].values for col in method_names]

        bp = ax.boxplot(
            data_by_method,
            labels=method_names,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            widths=0.5,
        )

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        for meanline in bp['means']:
            meanline.set_color('red')
            meanline.set_linewidth(2)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1.2)
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1.2)
        for flier in bp['fliers']:
            flier.set(marker='o', color='gray', alpha=0.5, markersize=4)

        for i, data in enumerate(data_by_method):
            jx = np.random.normal(i + 1, 0.04, size=len(data))
            ax.scatter(jx, data, alpha=0.65, s=28, color='darkred',
                       edgecolors='black', linewidth=0.4, zorder=3)

        for edge in ['left', 'bottom']:
            ax.spines[edge].set_visible(True)
            ax.spines[edge].set_color('black')
            ax.spines[edge].set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(axis='both', which='major', direction='out',
                       length=6, width=1.5, colors='black', bottom=True, left=True)
        ax.set_xticklabels(method_names, rotation=35, ha='right',
                           fontsize=15, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=15, pad=10)
        ax.set_ylim(-5, 105)
        ax.grid(False)
        ax.set_facecolor('white')

    # ── 图例：放在第一个箱线图（ax2）内部左上角 ──────────────────────────────
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgray', alpha=0.7, edgecolor='black',
              label='Interquartile range'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
        plt.Line2D([0], [0], color='red',   linewidth=2, label='Mean'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='darkred', markersize=7,
                   alpha=0.65, label='Data Points'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left',
               fontsize=12, frameon=True, framealpha=0.9,
               edgecolor='#cccccc')

    fig.suptitle('Algorithm Performance Comparison',
                  fontsize=14, fontweight='bold')

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"组图已保存: {save_path}")
    plt.show()


# ============================================================
# 绘制合并热力图（保留，按需启用）
# ============================================================

def plot_combined_heatmap(df_hit, df_cov, save_path):
    n_rows = df_hit.shape[0]
    n_cols = df_hit.shape[1]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(16, n_cols * 3.2 + 6), max(6, n_rows * 0.82 + 3.5))
    )

    configs = [
        (df_hit, 'Minimal Precision (Hit Rate %)', 'YlOrRd'),
        (df_cov, 'Coverage Rate (Recall %)',        'YlGnBu'),
    ]

    for ax, (df, title, cmap) in zip(axes, configs):
        im = ax.imshow(df.values.astype(float), aspect='auto',
                       cmap=cmap, vmin=0, vmax=100)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(df.columns, fontsize=11, fontweight='bold',
                           rotation=20, ha='right')
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(df.index, fontsize=15)

        for i in range(n_rows):
            for j in range(n_cols):
                val = df.values[i, j]
                text_color = 'white' if val > 60 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        fontsize=15, color=text_color, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label('(%)', fontsize=11)
        cbar.ax.tick_params(labelsize=10)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Product', fontsize=12, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle('Algorithm Performance Comparison (5 Methods)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"合并热力图已保存: {save_path}")
    plt.show()


# ============================================================
# 主流程
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== 读取文献数据 ===")
    lit_dict, cpd_fullname = load_literature(EXCEL_LIT_FILE)

    print("\n=== 读取 ET-EComp 预测结果 ===")
    et_ecomp_preds = load_et_targets(ET_ECOMP_BASE_DIR, method_name="ET-EComp")

    print("\n=== 读取 ET-FSEOF 预测结果 ===")
    et_fseof_preds = load_et_fseof_targets(ET_FSEOF_BASE_DIR)

    print("\n=== 计算 ET-optME 交集 & 并集 ===")
    common_et_cpds = set(et_ecomp_preds.keys()) & set(et_fseof_preds.keys())
    intersection_preds = {cpd: et_ecomp_preds[cpd] & et_fseof_preds[cpd] for cpd in common_et_cpds}
    union_preds        = {cpd: et_ecomp_preds[cpd] | et_fseof_preds[cpd] for cpd in common_et_cpds}
    for cpd in sorted(common_et_cpds):
        print(f"  {cpd}  Intersection={len(intersection_preds[cpd])}  Union={len(union_preds[cpd])}")

    print("\n=== 读取 ecFbridge 预测结果 ===")
    ecfbridge_preds = load_ecfbridge_targets(ECFBRIDGE_BASE_DIR)

    method_names = [
        'ET-EComp',
        'ET-ESEOF',
        'ET-OptME\nIntersection',
        'ET-OptME\nUnion',
        'ecFB',
    ]
    pred_dict_list = [
        et_ecomp_preds,
        et_fseof_preds,
        intersection_preds,
        union_preds,
        ecfbridge_preds,
    ]

    print("\n=== 计算性能指标 ===")
    df_hit, df_cov, counts_mat = compute_hit_rates(lit_dict, pred_dict_list, method_names)

    print("\n--- 命中率 (Minimal Precision) ---")
    print(df_hit.to_string())
    print("\n--- 覆盖率 (Coverage/Recall) ---")
    print(df_cov.to_string())

    df_hit.to_csv(os.path.join(OUTPUT_DIR, 'hit_rate.csv'))
    df_cov.to_csv(os.path.join(OUTPUT_DIR, 'coverage_rate.csv'))
    print(f"\nCSV 已保存至 {OUTPUT_DIR}/")

    # 热力图（按需启用）
    # plot_combined_heatmap(
    #     df_hit, df_cov,
    #     save_path=os.path.join(OUTPUT_DIR, 'heatmap_combined.png')
    # )

    # 组图：柱状图 + 两个箱线图横向排列
    plot_combined_figure(
        df_hit, df_cov, counts_mat,
        method_names=method_names,
        product_labels=list(df_hit.index),
        save_path=os.path.join(OUTPUT_DIR, 'bar_boxplot_combined.png')
    )

    print("\n全部完成！结果保存在:", OUTPUT_DIR)

    print("\n=== 计算性能指标 ===")
    df_hit, df_cov, counts_mat = compute_hit_rates(lit_dict, pred_dict_list, method_names)

    print("\n--- 命中率 (Minimal Precision) ---")
    print(df_hit.to_string())

    # 打印命中率的平均值
    avg_hit_rate = df_hit.mean(axis=0)
    print("\n--- 平均命中率 ---")
    print(avg_hit_rate)

    print("\n--- 覆盖率 (Coverage/Recall) ---")
    print(df_cov.to_string())

    # 打印覆盖率的平均值
    avg_coverage_rate = df_cov.mean(axis=0)
    print("\n--- 平均覆盖率 ---")
    print(avg_coverage_rate)


if __name__ == '__main__':
    main()