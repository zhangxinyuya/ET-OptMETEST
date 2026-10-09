"""
g0_fix_v4.py
------------
Iterative two-phase thermodynamic g0 correction.

Algorithm per outer iteration:
  Phase 1  Batch-correct reactions (set g0 -> NaN) until B >= target_B.
  Phase 2  Roll back to round start. Individually add back candidates
           from the critical batch. Identify which ones significantly
           improve B — these are "essential". Commit them as NaN.

  If Phase 2 reaches feasibility → great, check if more iterations needed.
  If Phase 2 does NOT reach feasibility but found reactions that improved B
  → commit those improvements, start a new outer iteration to find more.

This ensures only truly necessary reactions are removed from the g0 table,
and the process iterates until feasibility is achieved.
"""

import os, re, sys, json

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
print(f"Working directory set to: {project_root}")

os.environ['PATH']            += ':/opt/gurobi1200/linux64/bin'
os.environ['GUROBI_HOME']      = '/opt/gurobi1200/linux64'
os.environ['LD_LIBRARY_PATH']  = '/opt/gurobi1200/linux64/lib'

sys.path.append('/home/sun/ETGEMS-10.20/')
import script.pyomo_solving as etgf
from script.pyomo_solving import *

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family']       = 'sans-serif'
matplotlib.rcParams['font.sans-serif']   = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pyomo.environ as pyo
from pyomo.environ import value
from pyomo.opt import SolverStatus, TerminationCondition
import cobra


# ======================================================================
# g0 DataFrame helpers
# ======================================================================

def _set_g0_nan(reaction_g0: pd.DataFrame, rxn_id: str) -> bool:
    if rxn_id not in reaction_g0.index:
        return False
    if pd.isna(reaction_g0.loc[rxn_id, 'g0']):
        return False
    reaction_g0.loc[rxn_id, 'g0'] = np.nan
    return True


def _g0_is_active(reaction_g0: pd.DataFrame, rxn_id: str) -> bool:
    if rxn_id not in reaction_g0.index:
        return False
    return not pd.isna(reaction_g0.loc[rxn_id, 'g0'])


def _active_g0(rg0: pd.DataFrame) -> pd.DataFrame:
    """Return copy with NaN rows dropped — ready for the solver."""
    return rg0.dropna(subset=['g0'])


# ======================================================================
# Misc helpers
# ======================================================================

def json_load(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_pair_rxn(rxn_id: str) -> str:
    """Return the paired forward/reverse reaction ID."""
    if '_reverse' in rxn_id:
        return rxn_id.replace('_reverse', '', 1)
    m = re.search(r'(_num\d+)$', rxn_id)
    if m:
        return rxn_id[:m.start()] + '_reverse' + m.group(1)
    return rxn_id + '_reverse'


# ======================================================================
# Solver wrappers
# ======================================================================

def _solve_two_step(Concretemodel_Need_Data, inputdic, objvalue2,
                    B_value, biomass_ratio, opt):
    Concretemodel_Need_Data['B_value'] = B_value
    Concretemodel_Need_Data['K_value'] = 1249

    constr_bio = {
        'fix_reactions'      : {},
        'substrate_constrain': (inputdic['substrate'], objvalue2),
    }
    bio_model = FBA_template2(
        set_obj_value=True, obj_name=inputdic['biomass'], obj_target='maximize',
        mode='ST', constr_coeff=constr_bio,
        Concretemodel_Need_Data=Concretemodel_Need_Data,
    )
    res_bio = opt.solve(bio_model)
    if res_bio.solver.termination_condition != TerminationCondition.optimal:
        return None, 0, -99999, False
    v0_biomass = value(bio_model.reaction[inputdic['biomass']])
    if v0_biomass < 1e-6:
        return None, 0, -99999, False

    constr_B = {
        'fix_reactions'      : {inputdic['biomass']: [v0_biomass * biomass_ratio,
                                                       float('inf')]},
        'substrate_constrain': (inputdic['substrate'], objvalue2),
    }
    diag_model = FBA_template2(
        set_obj_B_value=True, obj_name=inputdic['biomass'], obj_target='maximize',
        mode='ST', constr_coeff=constr_B,
        Concretemodel_Need_Data=Concretemodel_Need_Data,
    )
    res_B = opt.solve(diag_model)
    if res_B.solver.termination_condition != TerminationCondition.optimal:
        return None, v0_biomass, -99999, False

    return diag_model, v0_biomass, value(diag_model.B), True


def _check_feasible_at_target(Concretemodel_Need_Data, inputdic, objvalue2,
                               target_B, biomass_ratio, opt):
    _, v0_biomass, current_B, success = _solve_two_step(
        Concretemodel_Need_Data, inputdic, objvalue2, target_B, biomass_ratio, opt
    )
    if not success or v0_biomass < 1e-6:
        return False, None, None
    return True, current_B, v0_biomass


# ======================================================================
# B-value history plot  (iteration-aware)
# ======================================================================

def _plot_B_history(B_history: list, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    if not B_history:
        return

    labels = [h['label']     for h in B_history]
    B_vals = [h['B']         for h in B_history]
    phases = [h['phase']     for h in B_history]
    iters  = [h['iteration'] for h in B_history]

    color_map = {
        'phase1'          : '#4C8CBF',
        'phase2_rollback' : '#E87B2A',
        'phase2'          : '#F5C542',
        'phase2_essential': '#E53935',
        'phase2_no_help'  : '#AAAAAA',
    }

    fig, (ax_main, ax_delta) = plt.subplots(
        2, 1, figsize=(max(16, len(B_history) * 0.75), 9),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True
    )

    xs         = list(range(len(B_vals)))
    bar_colors = [color_map.get(p, '#888') for p in phases]

    ax_main.bar(xs, B_vals, color=bar_colors, alpha=0.5, zorder=2)
    ax_main.plot(xs, B_vals, 'k-o', markersize=4, linewidth=1.2, zorder=3)
    ax_main.axhline(y=0, color='red', linestyle='--', linewidth=2, zorder=4)

    for i, h in enumerate(B_history):
        if h['phase'] == 'phase2_rollback':
            ax_main.axvline(x=i, color='#E87B2A', linestyle=':',
                            linewidth=1.2, alpha=0.8)
        if h['phase'] == 'phase2_essential':
            ax_main.bar(i, B_vals[i], color='#E53935', alpha=0.85, zorder=5)

    seen_iters = set()
    for i, it in enumerate(iters):
        if it not in seen_iters and i > 0:
            ax_main.axvline(x=i - 0.5, color='black', linestyle='-',
                            linewidth=1.5, alpha=0.4)
            ax_main.text(i, ax_main.get_ylim()[1] * 0.97,
                         f'Iter {it}', fontsize=8, ha='left',
                         color='black', alpha=0.6)
        seen_iters.add(it)

    ax_main.set_ylabel('MDF (B)  kJ/mol', fontsize=11)
    ax_main.set_title(
        'g0 Iterative Correction -- MDF (B) over steps\n'
        '(blue=Phase1, orange=rollback, yellow=Phase2, '
        'red=essential, grey=no help)',
        fontsize=12, fontweight='bold'
    )
    legend_els = [
        Patch(facecolor='#4C8CBF', alpha=0.7, label='Phase1 batch'),
        Patch(facecolor='#E87B2A', alpha=0.7, label='Phase2 rollback'),
        Patch(facecolor='#F5C542', alpha=0.7, label='Phase2 individual'),
        Patch(facecolor='#E53935', alpha=0.7, label='Phase2 essential'),
        Patch(facecolor='#AAAAAA', alpha=0.7, label='Phase2 no help'),
        plt.Line2D([0], [0], color='red',   linestyle='--', label='Target B=0'),
        plt.Line2D([0], [0], color='black', linestyle='-',  alpha=0.4,
                   label='New iteration'),
    ]
    ax_main.legend(handles=legend_els, fontsize=9, loc='lower right')

    delta_B = [0] + [B_vals[i] - B_vals[i-1] for i in range(1, len(B_vals))]
    delta_colors = ['#2ca02c' if d >= 0 else '#d62728' for d in delta_B]
    ax_delta.bar(xs, delta_B, color=delta_colors, alpha=0.7)
    ax_delta.axhline(y=0, color='black', linewidth=0.8)
    ax_delta.set_ylabel('Delta B  kJ/mol', fontsize=9)
    ax_delta.set_title('B increment per step (green=improvement, red=decline)',
                       fontsize=9)
    ax_delta.set_xticks(xs)
    ax_delta.set_xticklabels(labels, rotation=75, fontsize=7, ha='right')

    plt.tight_layout()
    fig_path = os.path.join(output_path, 'B_value_history.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] B-value history plot -> {fig_path}")


# ======================================================================
# Single two-phase round  (one call = one outer iteration)
# ======================================================================

B_IMPROVE_THRESHOLD = 1.0   # kJ/mol — minimum improvement to count as "helpful"

def _one_round(
    reaction_g0,
    Concretemodel_Need_Data,
    inputdic,
    opt,
    objvalue2,
    diagnose_B,
    target_B,
    biomass_ratio,
    batch_size,
    max_iter,
    iteration_no,
    B_history,
    fix_log,
):
    """
    Run one full two-phase correction round on the current reaction_g0.

    Phase 2 logic (FIXED):
      - Roll back to round start (clean g0).
      - Individually test each candidate from the critical batch.
      - If adding a reaction improves B by >= threshold → mark as essential.
      - If adding a reaction does NOT improve B → revert it (not essential).
      - After testing all candidates:
        - If feasible → done, commit essential reactions.
        - If NOT feasible but found essential reactions → commit them,
          return to outer loop for another iteration.
        - If NOT feasible and NO reaction helped → stop (stuck).

    Returns
    -------
    essential_rxns   : list of reaction IDs committed as NaN this round
    already_feasible : True if the model was feasible before any work
    phase2_found_any : True if Phase 2 found at least one essential reaction
    reached_feasible : True if this round achieved full feasibility
    """

    prefix = f"I{iteration_no}"

    # Save full copy at round start — we will restore to this state
    g0_round_start = reaction_g0.copy()

    def measure_B(label: str, phase: str) -> float:
        Concretemodel_Need_Data['reaction_g0'] = _active_g0(reaction_g0)
        _, _, B_val, ok = _solve_two_step(
            Concretemodel_Need_Data, inputdic, objvalue2,
            diagnose_B, biomass_ratio, opt
        )
        B = float(B_val) if ok else -99999.0
        B_history.append({'label': f"{prefix}-{label}", 'B': B,
                          'phase': phase, 'iteration': iteration_no})
        return B

    def is_feasible() -> bool:
        Concretemodel_Need_Data['reaction_g0'] = _active_g0(reaction_g0)
        ok, _, _ = _check_feasible_at_target(
            Concretemodel_Need_Data, inputdic, objvalue2,
            target_B, biomass_ratio, opt
        )
        return ok

    def get_candidates(diag_model):
        Df_d   = {x: value(diag_model.Df[x])       for x in diag_model.Df}
        flux_d = {x: value(diag_model.reaction[x]) for x in diag_model.reaction}

        def filt(need_flux):
            return {
                r: df for r, df in Df_d.items()
                if df is not None
                and df < target_B
                and _g0_is_active(reaction_g0, r)
                and ((flux_d.get(r) or 0) > 1e-6 if need_flux else True)
            }

        d = filt(True)
        if not d:
            print("  No active negative-Df reactions with flux. "
                  "Broadening search (ignoring flux)...")
            d = filt(False)
        return sorted(d.items(), key=lambda kv: kv[1])

    # ── Check feasibility before doing any work ──
    B0 = measure_B('start', 'phase1')
    print(f"\n  [Iter {iteration_no}] Start B = {B0:.4f}")

    if is_feasible():
        print(f"  [Iter {iteration_no}] Already feasible — stopping outer loop.")
        return [], True, False, True

    # ==============================================================
    # Phase 1  — batch removal until feasible
    # ==============================================================
    print(f"\n  {'='*54}")
    print(f"  Phase 1  (iter={iteration_no}, batch_size={batch_size})")
    print(f"  {'='*54}")

    snapshots          = []
    critical_batch_idx = None

    for batch_no in range(max_iter):
        Concretemodel_Need_Data['reaction_g0'] = _active_g0(reaction_g0)
        diag_model, _, cur_B, ok = _solve_two_step(
            Concretemodel_Need_Data, inputdic, objvalue2,
            diagnose_B, biomass_ratio, opt
        )
        if not ok:
            print("  [Phase1] diagnose_B solve failed; stopping.")
            break

        sorted_probs = get_candidates(diag_model)
        if not sorted_probs:
            print("  [Phase1] No candidate reactions; stopping.")
            break

        batch_rxns = [r for r, _ in sorted_probs[:batch_size]]
        df_map     = dict(sorted_probs)
        g0_before  = reaction_g0.copy()

        print(f"\n  -- Batch {batch_no+1}  (B = {cur_B:.4f}) --")

        fixed_this = []
        for rxn_id in batch_rxns:
            for cand in [rxn_id, get_pair_rxn(rxn_id)]:
                if _g0_is_active(reaction_g0, cand):
                    orig = float(reaction_g0.loc[cand, 'g0'])
                    _set_g0_nan(reaction_g0, cand)
                    fixed_this.append(cand)
                    print(f"    {cand:45s}  orig g0 = {orig:>10.4f} -> NaN")

        snapshots.append({
            'batch_no'  : batch_no + 1,
            'g0_before' : g0_before,
            'batch_rxns': batch_rxns,
            'fixed'     : fixed_this,
            'df_map'    : df_map,
        })

        B_after = measure_B(f'P1-b{batch_no+1}', 'phase1')
        print(f"  -> B after = {B_after:.4f}")

        if is_feasible():
            print(f"\n  [OK] Feasible after batch {batch_no+1}.")
            critical_batch_idx = len(snapshots) - 1
            break
    else:
        print(f"\n  Reached max_iter={max_iter} without feasibility.")

    if critical_batch_idx is None:
        # Phase 1 never reached feasibility — nothing to do
        print("  [Phase1] Could not reach feasibility. Restoring g0.")
        for idx in g0_round_start.index:
            reaction_g0.loc[idx, 'g0'] = g0_round_start.loc[idx, 'g0']
        return [], False, False, False

    # ==============================================================
    # Phase 2  — roll back to round start, test each candidate
    #            INDIVIDUALLY to find which ones actually help
    # ==============================================================
    crit = snapshots[critical_batch_idx]
    print(f"\n  {'='*54}")
    print(f"  Phase 2  (iter={iteration_no}, "
          f"critical batch {crit['batch_no']})")
    print(f"  Candidates: {crit['batch_rxns']}")
    print(f"  {'='*54}")

    # Roll back to the state at the START of this entire round
    for idx in g0_round_start.index:
        reaction_g0.loc[idx, 'g0'] = g0_round_start.loc[idx, 'g0']

    B_rb = measure_B('P2-rollback', 'phase2_rollback')
    print(f"  B after rollback (to round start) = {B_rb:.4f}")

    essential_rxns   = []
    reached_feasible = False
    prev_B           = B_rb

    for i, rxn_id in enumerate(crit['batch_rxns']):
        pair = get_pair_rxn(rxn_id)

        # Save state before this candidate
        g0_before_candidate = reaction_g0.copy()

        changed = []
        for cand in [rxn_id, pair]:
            if _g0_is_active(reaction_g0, cand):
                _set_g0_nan(reaction_g0, cand)
                changed.append(cand)

        B_cur    = measure_B(f'P2-{i+1}:{rxn_id[:18]}', 'phase2')
        feasible = is_feasible()
        delta    = B_cur - prev_B

        # Decide: did this reaction help?
        is_helpful = (delta > B_IMPROVE_THRESHOLD)

        if feasible:
            status = "[OK] FEASIBLE"
            is_helpful = True   # always keep if it makes us feasible
        elif is_helpful:
            status = f"[++] B improved by {delta:.4f}"
        else:
            status = f"[--] no significant help (delta={delta:.4f})"

        print(f"  [{i+1:02d}/{len(crit['batch_rxns'])}] "
              f"+{rxn_id} (+ reverse)  B={B_cur:.4f}  {status}")

        if is_helpful:
            # Keep this reaction as NaN — it's essential
            essential_rxns.extend(changed)
            prev_B = B_cur

            # Update B_history tag
            B_history[-1]['phase'] = 'phase2_essential'

            # Record in fix_log
            for cand in changed:
                orig_g0 = g0_round_start.loc[cand, 'g0'] if cand in g0_round_start.index else np.nan
                fix_log[cand] = {
                    'original_g0': float(orig_g0) if not pd.isna(orig_g0) else 'N/A',
                    'Df_at_fix'  : crit['df_map'].get(cand,
                                   crit['df_map'].get(rxn_id, 0.0)),
                    'phase'      : 'phase2_essential',
                    'batch_no'   : crit['batch_no'],
                    'iteration'  : iteration_no,
                }

            if feasible:
                reached_feasible = True
                print(f"\n  [FEASIBLE] Reached feasibility!")
                break
        else:
            # Revert — this reaction is NOT essential
            for idx in g0_before_candidate.index:
                reaction_g0.loc[idx, 'g0'] = g0_before_candidate.loc[idx, 'g0']
            B_history[-1]['phase'] = 'phase2_no_help'

    # Summary
    phase2_found_any = len(essential_rxns) > 0

    if reached_feasible:
        print(f"\n  Essential set this round ({len(essential_rxns)}):")
        for r in essential_rxns:
            print(f"    {r}")
    elif phase2_found_any:
        print(f"\n  [Phase2] Not yet feasible, but found {len(essential_rxns)} "
              f"essential reaction(s) this round:")
        for r in essential_rxns:
            print(f"    {r}")
        print(f"  Will commit these and start a new outer iteration.")
    else:
        print(f"\n  [Phase2] No reaction in the critical batch improved B. "
              f"Stuck — consider increasing batch_size or checking the model.")

    return essential_rxns, False, phase2_found_any, reached_feasible


# ======================================================================
# Outer iterative loop
# ======================================================================

def fix_g0_iterative_loop(
    Concretemodel_Need_Data,
    inputdic,
    reaction_g0_init,
    model,
    solver        = 'gurobi',
    diagnose_B    = -5000,
    target_B      = 0,
    max_iter      = 200,
    biomass_ratio = 0.9,
    batch_size    = 10,
    max_rounds    = 50,
    output_path   = './results/g0_fix',
):
    """
    Repeatedly run two-phase g0 correction until the model is feasible.

    Each round:
      1. Phase 1: batch-remove g0 until feasible.
      2. Phase 2: roll back to clean state, individually test candidates
         from the critical batch, keep only those that improve B.
      3. If feasible → check next round (might already be done).
         If not feasible but found essentials → commit and iterate.
         If stuck → stop.

    Returns
    -------
    reaction_g0   : final corrected g0 DataFrame (NaN rows dropped)
    fix_log       : dict of all essential modifications
    log_df        : pd.DataFrame summary
    B_history     : list of dicts
    round_summary : list of dicts
    """
    reaction_g0 = reaction_g0_init.copy()
    opt         = pyo.SolverFactory(solver)
    objvalue2   = Concretemodel_Need_Data.get('objvalue2', 10)

    B_history     = []
    fix_log       = {}
    round_summary = []

    print(f"\n{'#'*62}")
    print(f"  Iterative g0 fix  (max_rounds={max_rounds})")
    print(f"{'#'*62}")

    for iteration_no in range(1, max_rounds + 1):
        print(f"\n{'#'*62}")
        print(f"  OUTER ITERATION {iteration_no}")
        print(f"{'#'*62}")

        essential_rxns, already_feasible, found_any, reached_feasible = _one_round(
            reaction_g0             = reaction_g0,
            Concretemodel_Need_Data = Concretemodel_Need_Data,
            inputdic                = inputdic,
            opt                     = opt,
            objvalue2               = objvalue2,
            diagnose_B              = diagnose_B,
            target_B                = target_B,
            biomass_ratio           = biomass_ratio,
            batch_size              = batch_size,
            max_iter                = max_iter,
            iteration_no            = iteration_no,
            B_history               = B_history,
            fix_log                 = fix_log,
        )

        round_summary.append({
            'iteration'       : iteration_no,
            'essential_rxns'  : list(essential_rxns),
            'n_essential'     : len(essential_rxns),
            'already_feasible': already_feasible,
            'found_any'       : found_any,
            'reached_feasible': reached_feasible,
        })

        if already_feasible:
            print(f"\n[Loop] Model feasible at start of iteration {iteration_no}.")
            print(f"[Loop] Outer loop complete after {iteration_no - 1} "
                  f"correction round(s).")
            break

        if reached_feasible:
            # Phase 2 reached feasibility — but we should verify at the
            # start of next iteration (the outer loop will do this).
            print(f"\n[Loop] Iteration {iteration_no}: reached feasibility "
                  f"with {len(essential_rxns)} essential reaction(s). "
                  f"Verifying in next iteration...")
            continue

        if found_any and not reached_feasible:
            # Phase 2 found essential reactions but not enough for feasibility.
            # Commit them and try again.
            print(f"\n[Loop] Iteration {iteration_no}: committed "
                  f"{len(essential_rxns)} essential reaction(s) as NaN. "
                  f"B still < 0, starting iteration {iteration_no + 1}.")
            continue

        if not found_any:
            print(f"\n[Loop] Phase 2 found no helpful reactions in iteration "
                  f"{iteration_no}. Stopping.")
            break

    else:
        print(f"\n[Loop] Reached max_rounds={max_rounds}. Stopping.")

    # ==================================================================
    # Final summary
    # ==================================================================
    print(f"\n{'='*62}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*62}")
    print(f"  Total outer iterations run : {len(round_summary)}")
    print(f"  Total essential reactions   : {len(fix_log)}")

    total_essential = sum(s['n_essential'] for s in round_summary)

    for s in round_summary:
        flags = []
        if s['already_feasible']:
            flags.append("already feasible")
        if s['reached_feasible']:
            flags.append("reached feasible")
        if not s['found_any'] and not s['already_feasible']:
            flags.append("stuck")
        flag_str = f"  ({', '.join(flags)})" if flags else ""
        print(f"    Iter {s['iteration']:>2d}: {s['n_essential']:>3d} essential"
              f"{flag_str}")

    # Drop NaN rows for downstream use
    n_before = len(reaction_g0)
    reaction_g0.dropna(subset=['g0'], inplace=True)
    n_after  = len(reaction_g0)
    n_dropped = n_before - n_after

    print(f"\n  Rows in original g0 : {len(reaction_g0_init)}")
    print(f"  Rows in fixed g0    : {n_after}")
    print(f"  Rows removed (NaN)  : {n_dropped}")
    print(f"  Expected (essential): {total_essential}")
    if n_dropped != total_essential:
        print(f"  WARNING: mismatch between dropped rows and essential count!")

    if fix_log:
        log_df = pd.DataFrame(fix_log).T.reset_index()
        log_df.columns = ['reaction', 'original_g0', 'Df_at_fix',
                          'phase', 'batch_no', 'iteration']
        essential_df = log_df[log_df['phase'] == 'phase2_essential']
        if len(essential_df):
            print(f"\n  All essential reactions ({len(essential_df)}):")
            print(essential_df[['reaction', 'original_g0',
                                 'Df_at_fix', 'iteration']].to_string(index=False))
    else:
        log_df = pd.DataFrame(
            columns=['reaction', 'original_g0', 'Df_at_fix',
                     'phase', 'batch_no', 'iteration']
        )

    _plot_B_history(B_history, output_path)

    return reaction_g0, fix_log, log_df, B_history, round_summary


# ======================================================================
# Validate and save
# ======================================================================

def validate_and_save_fixed_g0(
    fixed_reaction_g0,
    log_df,
    Concretemodel_Need_Data,
    inputdic,
    output_path,
    model,
    solver        = 'gurobi',
    target_B      = 0,
    biomass_ratio = 0.9,
):
    os.makedirs(output_path, exist_ok=True)
    opt       = pyo.SolverFactory(solver)
    objvalue2 = Concretemodel_Need_Data.get('objvalue2', 10)

    Concretemodel_Need_Data['reaction_g0'] = fixed_reaction_g0

    diag_model, v0_biomass, final_B, success = _solve_two_step(
        Concretemodel_Need_Data, inputdic, objvalue2, target_B, biomass_ratio, opt
    )

    print(f"\n[Validation] Two-step solve: {'SUCCESS' if success else 'FAILED'}")

    if not success or v0_biomass < 1e-6:
        print("[Validation] FAIL: model cannot grow under B=0.")
        _save_partial(fixed_reaction_g0, log_df, output_path)
        return None, None, None

    print(f"[Validation] B = {final_B:.4f},  biomass = {v0_biomass:.6f}")
    print("[Validation] PASS!" if final_B >= target_B else
          "[Validation] FAIL: B below target.")

    fixed_g0_path = os.path.join(output_path, 'reaction_g0_fixed.txt')
    fixed_reaction_g0.to_csv(fixed_g0_path, sep='\t')
    print(f"[Saved] Fixed g0 -> {fixed_g0_path}")

    log_path = os.path.join(output_path, 'g0_fix_log.xlsx')
    log_df.to_excel(log_path, index=False)
    print(f"[Saved] Fix log  -> {log_path}")

    if len(log_df) > 0:
        valid_log = log_df[log_df['original_g0'] != 'N/A'].copy()
        valid_log['original_g0'] = valid_log['original_g0'].astype(float)
        valid_log['Df_at_fix']   = valid_log['Df_at_fix'].astype(float)

        phase_colors = {
            'phase1'          : '#4C8CBF',
            'phase2'          : '#F5C542',
            'phase2_essential': '#E53935',
        }
        colors = valid_log['phase'].map(
            lambda p: phase_colors.get(p, '#888888')
        ).tolist()

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle('g0 Fix Analysis', fontsize=14, fontweight='bold')

        axes[0].bar(range(len(valid_log)), valid_log['original_g0'],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.4)
        axes[0].axhline(y=0, color='steelblue', linestyle='--', linewidth=1.5,
                        label='Fixed g0 = NaN (removed)')
        axes[0].set_title('Original g0 per Reaction (color = phase)')
        axes[0].set_ylabel('g0 (kJ/mol)')
        axes[0].set_xticks([])
        axes[0].legend(handles=[
            Patch(facecolor='#E53935', label='Essential (removed)'),
        ], fontsize=9)

        axes[1].barh(valid_log['reaction'], valid_log['Df_at_fix'],
                     color=colors, alpha=0.8)
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        axes[1].set_title('Df Value at Fix (kJ/mol)')
        axes[1].set_xlabel('Df')
        axes[1].tick_params(axis='y', labelsize=6)

        plt.tight_layout()
        fig_path = os.path.join(output_path, 'g0_fix_visualization.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Saved] Visualization -> {fig_path}")

    return diag_model, final_B, v0_biomass


def _save_partial(fixed_reaction_g0, log_df, output_path):
    os.makedirs(output_path, exist_ok=True)
    fixed_reaction_g0.to_csv(
        os.path.join(output_path, 'reaction_g0_partial.txt'), sep='\t'
    )
    log_df.to_excel(os.path.join(output_path, 'g0_fix_log.xlsx'), index=False)
    print(f"[Saved] Partial g0 -> {output_path}/reaction_g0_partial.txt")
    print(f"[Saved] Fix log    -> {output_path}/g0_fix_log.xlsx")


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':

    model_file            = "./file/iCZ870.json"
    reaction_g0_file      = './file/iCZ870_g0_n.csv'
    metabolites_lnC_file  = './file/iCZ870_met.txt'
    reaction_kcat_MW_file = './file/reaction_kcat_MW_modify_n.csv'
    dictionarymodel_path  = "./results/etcz870.json"
    OUTPUT_PATH           = './results/g0_fix'

    inputdic = {
        "model"      : 'iCZ870.json',
        "substrate"  : "EX_cpd00027_e_reverse",
        "biomass"    : "EX_biomass_c",
        "product"    : "EX_cpd00039_e",
        "mode"       : "SET",
        "oxygenstate": "aerobic",
    }

    DIAGNOSE_B    = -5000
    TARGET_B      = 0
    BIOMASS_RATIO = 0.9
    BATCH_SIZE    = 10
    MAX_ITER      = 200
    MAX_ROUNDS    = 50

    # -- Load model --
    model  = cobra.io.load_json_model(model_file)
    Concretemodel_Need_Data = Get_Concretemodel_Need_Data(model_file)
    dictionary_model = json_load(dictionarymodel_path)
    print(f"[DictModel] keys (first 5): {list(dictionary_model.keys())[:5]} ...")

    rname3 = []
    get_dictionarymodel_data2(dictionary_model, Concretemodel_Need_Data, rname3)
    Get_Concretemodel_Need_Data_g0(
        Concretemodel_Need_Data, reaction_g0_file,
        metabolites_lnC_file, reaction_kcat_MW_file,
    )

    Concretemodel_Need_Data['reaction_g0']['g0'] = (
        Concretemodel_Need_Data['reaction_g0']['g0'].replace(0, np.nan)
    )
    Concretemodel_Need_Data['reaction_g0'].dropna(subset=['g0'], inplace=True)

    Inc = Concretemodel_Need_Data['metabolites_lnC']
    for i in Concretemodel_Need_Data['metabolite_list']:
        if i not in Inc.index:
            Inc.loc[i, 'lnClb'] = -14.508658
            Inc.loc[i, 'lnCub'] = -3.912023
    Concretemodel_Need_Data['metabolites_lnC'] = Inc

    if inputdic['oxygenstate'] == 'aerobic':
        model.reactions.get_by_id('EX_cpd00007_e_reverse').upper_bound = 1000

    model.objective = inputdic['substrate']
    objvalue2 = model.optimize().objective_value
    Concretemodel_Need_Data['objvalue2'] = objvalue2

    reaction_g0 = Concretemodel_Need_Data['reaction_g0'].copy()
    print(f"[g0] rows={len(reaction_g0)}, "
          f"range=[{reaction_g0['g0'].min():.2f}, {reaction_g0['g0'].max():.2f}]")
    print(f"[Substrate] objvalue2 = {objvalue2:.4f}")

    # -- Step 2: Iterative two-phase correction --
    print("\n>>> Step 2: Iterative two-phase g0 fix")

    fixed_g0, fix_log, log_df, B_history, round_summary = fix_g0_iterative_loop(
        Concretemodel_Need_Data = Concretemodel_Need_Data,
        inputdic                = inputdic,
        reaction_g0_init        = reaction_g0,
        model                   = model,
        solver                  = 'gurobi',
        diagnose_B              = DIAGNOSE_B,
        target_B                = TARGET_B,
        max_iter                = MAX_ITER,
        biomass_ratio           = BIOMASS_RATIO,
        batch_size              = BATCH_SIZE,
        max_rounds              = MAX_ROUNDS,
        output_path             = OUTPUT_PATH,
    )

    # -- Step 3: Validate and save --
    print("\n>>> Step 3: Validate and save")

    val_model, final_B, final_biomass = validate_and_save_fixed_g0(
        fixed_reaction_g0       = fixed_g0,
        log_df                  = log_df,
        Concretemodel_Need_Data = Concretemodel_Need_Data,
        inputdic                = inputdic,
        output_path             = OUTPUT_PATH,
        model                   = model,
        solver                  = 'gurobi',
        target_B                = TARGET_B,
        biomass_ratio           = BIOMASS_RATIO,
    )

    print("\n>>> Pipeline complete")
    if final_B is not None:
        print(f"    Final MDF (B)           = {final_B:.4f}")
        print(f"    Final biomass           = {final_biomass:.6f}")
        print(f"    Total essential rxns     = {len(fix_log)}")
        print(f"    Outer iterations run    = {len(round_summary)}")
        essential_all = list(fix_log.keys())
        if essential_all:
            print(f"    Essential reactions     = {essential_all}")
    print(f"    Results saved to        = {OUTPUT_PATH}")