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
import multiprocessing
import os
import copy
from ET_optme import _find_feasible_product_coeff, _find_feasible_biomass_coeff
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
# 文件路径配置（按需修改）
# ============================================================
model_file              = "./file/iCZ870.json"
model0_file             = "./file/iCZ870.json"
reaction_kcat_MW_file   = './file/reaction_kcat_MW_modify_n.csv'
dictionarymodel_path    = './results/etcz870.json'
reaction_g0_file        = './file/iCZ870_g0_n.csv'
metabolites_lnC_file    = './file/iCZ870_met.txt'
BASE_SAVE_DIR           = './result_ET_EComp'   # 批量结果根目录
path_strain             = 'iCW'

# ============================================================
# dictionary_model 只加载一次，所有产品共用
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
    """对单个产品运行完整 ET-EComp 流程"""
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
        "taskname":    "ET-EComp",
        "mode":        "SET",
        "oxygenstate": "aerobic"
    }

    # 结果保存目录：result_ET_EComp/<product_id>/
    save_dir = os.path.join(BASE_SAVE_DIR, product_id)
    os.makedirs(save_dir, exist_ok=True)

    # 保存本次 task.json
    task_json_path = os.path.join(save_dir, 'task_ecomp.json')
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
        # ── Step 1: 计算最大生长速率 ──────────────────────────────
        print(f"[Step 1] 计算最大生长速率...")
        B_value1, v0_biomass, bio, totalE1, objvalue2 = calculate_biomass(
            Concretemodel_Need_Data, inputdic, model, path_strain
        )
        print(f"[Step 1] v0_biomass={v0_biomass:.4f}, totalE1={totalE1}, B_value1={B_value1}")

        # ── Step 2: 计算野生型酶浓度范围（并行）────────────────────
        print(f"[Step 2] 计算野生型酶浓度范围（并行）...")

        enzyme_list = list(Concretemodel_Need_Data['mw_dict'].keys())
        if inputdic['mode'] == 'SET':
            test_genes = list(Concretemodel_Need_Data['enzyme_rxns_dict'].keys())[:3]
            Concretemodel_Need_Data['E_total'] = totalE1
            Concretemodel_Need_Data['B_value'] = B_value1 * 0.98
            biomass_coeff = _find_feasible_biomass_coeff(
                Concretemodel_Need_Data,
                {'fix_reactions': {}, 'substrate_constrain': (inputdic['substrate'], objvalue2)},
                inputdic, v0_biomass, test_genes)
            if biomass_coeff is None:
                raise ValueError("[SET] biomass_constrain 系数从0.95降至0.90仍无解")
            Concretemodel_Need_Data['biomass_coeff'] = biomass_coeff
        else:
            biomass_coeff = None

        clean_data_wild = copy.deepcopy(Concretemodel_Need_Data)

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results_wild = pool.starmap(
            calculate_wildrange,
            [(clean_data_wild, obj_name, inputdic, objvalue2,  # ← 改这里
              v0_biomass, totalE1, path_strain, B_value1, biomass_coeff)
             for obj_name in enzyme_list]
        )
        pool.close()
        pool.join()

        final_wild = {}
        for r in results_wild:
            final_wild.update(r)

        wild_file_path = os.path.join(save_dir, 'wild-enzyme.json')
        with open(wild_file_path, 'w') as f:
            json.dump(final_wild, f, indent=4)
        print(f"[Step 2] 野生型结果已保存: {wild_file_path}")

        # ── Step 3: 计算最大产品速率 ─────────────────────────────
        print(f"[Step 3] 计算最大产品速率...")
        B_value2, v1_product_max, pro, totalE2, EcoECM_FBA_protainmodel_B2 = calculate_product(
            Concretemodel_Need_Data, inputdic, objvalue2, path_strain, v0_biomass
        )
        print(f"[Step 3] v1_product_max={v1_product_max:.4f}, totalE2={totalE2}, B_value2={B_value2}")

        # ── Step 4: 计算过表达酶浓度范围（并行）────────────────────
        print(f"[Step 4] 计算过表达酶浓度范围（并行）...")
        if inputdic['mode'] == 'SET':
            test_genes = list(Concretemodel_Need_Data['enzyme_rxns_dict'].keys())[:3]
            Concretemodel_Need_Data['E_total'] = totalE2 * 1.01
            Concretemodel_Need_Data['B_value'] = B_value2 * 0.99
            product_coeff = _find_feasible_product_coeff(
                Concretemodel_Need_Data,
                {'fix_reactions': {},
                 'substrate_constrain': (inputdic['substrate'], objvalue2),
                 'biomass_constrain': (inputdic['biomass'], v0_biomass * 0.1)},
                inputdic, v1_product_max, test_genes)
            if product_coeff is None:
                raise ValueError("[SET] product_constrain 系数从0.95降至0.90仍无解")
            Concretemodel_Need_Data['product_coeff'] = product_coeff
        else:
            product_coeff = None

        clean_data_over = copy.deepcopy(Concretemodel_Need_Data)
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results_over = pool.starmap(
            calculate_over,
            [(clean_data_over, obj_name, inputdic, objvalue2,  # ← 改这里
              v0_biomass, v1_product_max, totalE2, path_strain, B_value2, product_coeff)
             for obj_name in enzyme_list]
        )
        pool.close()
        pool.join()

        final_over = {}
        for r in results_over:
            final_over.update(r)

        over_file_path = os.path.join(save_dir, 'over-enzyme.json')
        with open(over_file_path, 'w') as f:
            json.dump(final_over, f, indent=4)
        print(f"[Step 4] 过表达结果已保存: {over_file_path}")

        # ── Step 5: 结果处理 ──────────────────────────────────────
        print(f"[Step 5] 处理结果...")

        # 读取刚保存的结果（统一走 read_file 接口）
        enzyme_results_data, enzyme_overresults2_data = read_file(save_dir)

        # 筛选改造靶点
        ko_data, up_data, down_data = compare_results(
            enzyme_results_data, enzyme_overresults2_data, inputdic
        )

        # 反应方程字典
        reaction_list  = [rea.id for rea in model.reactions]
        equation_dict  = gene_reaction_map1(reaction_list, model)

        # 野生型酶浓度与反应通量
        E_refdict, mw_dict, reaction_dict_bio, kcat_dict = ref_e_con(
            inputdic, Concretemodel_Need_Data, totalE1, path_strain, objvalue2
        )

        # 野生型基因-反应映射
        new_gene_reaction_mapping_bio, gene_reaction_mapping, gene_kcat_mapping = gene_reaction_map(
            enzyme_list, reaction_dict_bio, kcat_dict, Concretemodel_Need_Data, model
        )

        # 过表达状态的反应通量
        reaction_dict, kcat_dict, EcoECM_FBA_protainmodel_pro_max = reaction_flux(
            inputdic, Concretemodel_Need_Data, totalE2,
            path_strain, totalE1, objvalue2, v0_biomass
        )

        # 过表达基因-反应映射
        new_gene_reaction_mapping, gene_reaction_mapping, gene_kcat_mapping = gene_reaction_map(
            enzyme_list, reaction_dict, kcat_dict, Concretemodel_Need_Data, model
        )

        # 酶用量比例
        normalized_E_dict, E_dict, e_dict = enzyme_usage_over(
            EcoECM_FBA_protainmodel_pro_max, Concretemodel_Need_Data
        )

        # 折叠变化
        Fold_change = fold_change(E_dict, E_refdict)

        # 生成结果表
        meged_df = must_df(
            inputdic, equation_dict,
            new_gene_reaction_mapping_bio, new_gene_reaction_mapping,
            E_refdict, E_dict, Fold_change, normalized_E_dict, e_dict,
            gene_kcat_mapping, enzyme_results_data, enzyme_overresults2_data,
            ko_data, up_data, down_data, Concretemodel_Need_Data
        )

        # 通量表
        bio_df, pro_df, KeyInfo = basic(
            bio, pro, B_value1, v0_biomass,
            B_value2, v1_product_max, totalE1, totalE2, inputdic
        )

        # 热力学瓶颈反应
        bottleneck_reactions_list, Df = bottleneck_reactions(
            EcoECM_FBA_protainmodel_B2, model
        )

        # ── Step 6: 保存 Excel ────────────────────────────────────
        output(KeyInfo, save_dir, bio_df, pro_df, meged_df, inputdic, Df)
        print(f"[Done] {product_id} 预测完成，结果已保存至: {save_dir}")

        # 打印关键改造靶点摘要
        print(f"\n[Key Targets]")
        print(f"  KO targets  ({len(ko_data)}):   {[x[0] for x in ko_data]}")
        print(f"  UP targets  ({len(up_data)}):   {[x[0] for x in up_data]}")
        print(f"  DOWN targets({len(down_data)}): {[x[0] for x in down_data]}")

        return product_id, True, None

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[Error] {product_id} 预测失败:\n{err_msg}")
        error_log = os.path.join(save_dir, 'error.log')
        with open(error_log, 'w') as f:
            f.write(err_msg)
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