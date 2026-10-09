import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PATH']            += ':/opt/gurobi1200/linux64/bin'
os.environ['GUROBI_HOME']      = '/opt/gurobi1200/linux64'
os.environ['LD_LIBRARY_PATH']  = '/opt/gurobi1200/linux64/lib'
import sys
sys.path.append('./script')
from pyomo_solving import *
from ET_optme import *
import pandas as pd
import cobra
import json
import os

# ============================================================
# 批量预测配置
# 在这里填写你要预测的所有产品 Exchange reaction ID
# ============================================================
PRODUCT_LIST = [
    "EX_cpd00039_e", "EX_cpd00322_e", "EX_cpd00107_e",
    "EX_cpd00156_e", "EX_cpd00161_e", "EX_cpd00227_e", "EX_cpd00129_e",
    "EX_cpd00051_e", "EX_cpd00064_e", "EX_cpd00274_e", "EX_cpd00069_e",
    "EX_cpd00065_e", "EX_cpd00054_e", "EX_cpd00084_e", "EX_cpd00119_e",
    "EX_cpd00066_e",
]

# ============================================================
# 文件路径配置（与原脚本保持一致，按需修改）
# ============================================================
model_file              = "./file/iCZ870.json"
model0_file             = "./file/iCZ870.json"
reaction_kcat_MW_file   = './file/reaction_kcat_MW_modify_n.csv'
dictionarymodel_path    = './results/etcz870.json'
reaction_g0_file        = './file/iCZ870_g0_n.csv'
metabolites_lnC_file    = './file/iCZ870_met.txt'
BASE_SAVE_DIR           = './result_ET_FSEOF'   # 批量结果根目录

# ============================================================
# 字典模型只加载一次，所有产品共用
# ============================================================
print("=" * 60)
print("[Init] 加载 dictionary_model...")
dictionary_model = json_load(dictionarymodel_path)
print("[Init] 加载完成")


def build_concretemodel_data() -> dict:
    """每个产品独立构建 Concretemodel_Need_Data，避免对象复用污染"""
    Concretemodel_Need_Data = Get_Concretemodel_Need_Data(model_file)

    rname3 = []
    get_dictionarymodel_data2(dictionary_model, Concretemodel_Need_Data, rname3)
    Get_Concretemodel_Need_Data_g0(
        Concretemodel_Need_Data,
        reaction_g0_file,
        metabolites_lnC_file,
        reaction_kcat_MW_file
    )

    # 替换 g0 中的 0 为 NaN 并去除
    Concretemodel_Need_Data['reaction_g0']['g0'] = (
        Concretemodel_Need_Data['reaction_g0']['g0'].replace(0, np.nan)
    )
    Concretemodel_Need_Data['reaction_g0'].dropna(subset=['g0'], inplace=True)

    # 补全代谢物浓度边界
    Inc = Concretemodel_Need_Data['metabolites_lnC']
    for i in Concretemodel_Need_Data['metabolite_list']:
        if i not in Inc.index:
            Inc.loc[i, 'lnClb'] = -14.508658
            Inc.loc[i, 'lnCub'] = -3.912023
    Concretemodel_Need_Data['metabolites_lnC'] = Inc

    return Concretemodel_Need_Data


def run_one_product(product_id: str):
    """对单个产品运行完整 ET-FSEOF 流程"""
    print(f"\n{'=' * 60}")
    print(f"[Task] 开始预测产品: {product_id}")
    print(f"{'=' * 60}")

    # 每次独立加载 cobra 模型（避免 bound 修改互相污染）
    model  = cobra.io.load_json_model(model_file)
    model0 = cobra.io.load_json_model(model0_file)

    # 本次任务的 inputdic（只改 product，其余不变）
    inputdic = {
        "model":       "iCZ870.json",
        "substrate":   "EX_cpd00027_e_reverse",
        "biomass":     "EX_biomass_c",
        "product":     product_id,
        "taskname":    "ET-FSEOF",
        "mode":        "SET",
        "oxygenstate": "aerobic"
    }

    # 结果保存目录：result_ET_FSEOF/<product_id>/
    save_dir = os.path.join(BASE_SAVE_DIR, product_id)
    os.makedirs(save_dir, exist_ok=True)

    # 保存本次 task.json
    task_json_path = os.path.join(save_dir, 'task_fseof.json')
    with open(task_json_path, 'w') as f:
        json.dump(inputdic, f, indent=4)
    print(f"[Info] task.json 已保存至: {task_json_path}")

    # 设置氧气条件
    oxygen_state = inputdic['oxygenstate']
    if oxygen_state == 'aerobic':
        model.reactions.get_by_id('EX_cpd00007_e_reverse').upper_bound = 1000
    elif oxygen_state == 'micro_aerobic':
        model.reactions.get_by_id('EX_cpd00007_e_reverse').upper_bound = 2
    elif oxygen_state == 'anaerobic':
        model.reactions.get_by_id('EX_cpd00007_e_reverse').upper_bound = 0

    # 每个产品独立构建热力学数据
    Concretemodel_Need_Data = build_concretemodel_data()

    try:
        # Step 1: 计算最大产率
        print(f"[Step 1] 计算最大产率...")
        product_val, objvalue2 = calculate_product_fseof(
            Concretemodel_Need_Data, inputdic, model
        )

        # Step 2: 在最大产率约束下计算生长速率与 FSEOF 扫描结果
        print(f"[Step 2] 计算生长速率与 FSEOF 扫描结果...")
        FSEOFdf, reactiondf = biomass(
            product_val, inputdic, Concretemodel_Need_Data,
            objvalue2, model, model0
        )

        # Step 3: 结果整理
        print(f"[Step 3] 整理结果表格...")
        columns = FSEOFdf.columns
        sorted_columns = sorted(columns, key=lambda x: get_sort_key(x, FSEOFdf))
        sorted_FSEOFdf  = FSEOFdf[sorted_columns]
        FSEOFdf = sorted_FSEOFdf[['gene'] + [col for col in sorted_columns if col != 'gene']]
        FSEOFdf = detail(FSEOFdf)
        FSEOFdf = result(FSEOFdf)

        # Step 4: 保存结果到 save_dir
        output_fseof(save_dir, inputdic, reactiondf, FSEOFdf)

        print(f"[Done] {product_id} 预测完成，结果已保存至: {save_dir}")
        print(f"[Key Results]\n{FSEOFdf[['gene', 'manipulations']].to_string(index=False)}")
        return product_id, True, None

    except Exception as e:
        print(f"[Error] {product_id} 预测失败: {e}")
        error_log = os.path.join(save_dir, 'error.log')
        with open(error_log, 'w') as f:
            f.write(str(e))
        return product_id, False, str(e)


# ============================================================
# 主循环：依次运行每个产品
# ============================================================
if __name__ == '__main__':
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    summary = []

    for product_id in PRODUCT_LIST:
        pid, success, err = run_one_product(product_id)
        summary.append({
            "product": pid,
            "success": success,
            "error":   err if err else ""
        })

    # 打印并保存汇总报告
    print("\n" + "=" * 60)
    print("[Summary] 全部批量预测完成")
    print("=" * 60)
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(BASE_SAVE_DIR, 'batch_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Info] 汇总报告已保存至: {summary_path}")