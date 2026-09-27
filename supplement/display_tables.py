"""Numerical display-table transformations extracted from the original report builder.

Consumes completed analysis tables only; performs no fitting or model selection.
The private report application and its delivery metadata are not required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASETS = ['bike', 'diamond', 'SGEMM', 'Transcoding']
LABELS = {'bike': 'Bike', 'diamond': 'Diamonds', 'SGEMM': 'SGEMM', 'Transcoding': 'Transcoding'}
METHODS = ['Fixed-Supervised', 'Fixed-PPI', 'raw_pinball', 'calibrated_pinball',
           'coverage_beta0', 'coverage_beta1', 'Extra-Supervised']


def build_tables(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    def read(name):
        return pd.read_csv(analysis_dir / f'{name}.csv')

    summary = read('tabular_summary')
    metric_table = []
    for dataset in DATASETS:
        for method in METHODS:
            row = summary[(summary.dataset == dataset) & (summary.method == method)].iloc[0]
            metric_table.append({
                'dataset': LABELS[dataset], 'method': method,
                'MSCE_K30': f'{row.group_raw_k30_mean:.6f} ± {row.group_raw_k30_std:.6f}',
                'K10_cross': f'{row.group_cross_product_k10_mean:.6f} ± {row.group_cross_product_k10_std:.6f}',
                'WSC': f'{row.wsc_mean:.5f} ± {row.wsc_std:.5f}',
                'ERT_L1': f'{row.ert_l1_gain_mean:.5f} ± {row.ert_l1_gain_std:.5f}',
                'coverage': f'{row.coverage_mean:.5f} ± {row.coverage_std:.5f}',
                'log_volume': f'{row.mean_log_volume_mean:.4f} ± {row.mean_log_volume_std:.4f}'})

    roles = read('roles').groupby('dataset', sort=False).first().reset_index()
    roles['selector_threshold_true_labels'] = roles.learning + roles.development
    roles['extra_supervised_training_labels'] = roles.learning + roles.development
    frequencies = read('selections').groupby(['dataset', 'method', 'candidate']).size().rename('count').reset_index()
    headroom = read('headroom').query('metric == "group_raw_k30"').groupby('dataset')[['fixed_ppi', 'evaluation_oracle', 'headroom']].mean().reset_index()
    headroom['relative_optimistic_headroom'] = headroom.headroom / headroom.fixed_ppi
    invariants = read('invariants').groupby('dataset').agg(
        contrasts=('shift', 'size'), raw_selection_changed=('raw_selection_changed', 'sum'),
        raw_order_changed=('raw_order_changed', 'sum'),
        max_calibrated_pinball_error=('calibrated_pinball', 'max'),
        max_beta0_error=('coverage_beta0', 'max'), max_beta1_error=('coverage_beta1', 'max'),
        max_deployed_threshold_error=('final_threshold_max_error', 'max')).reset_index()
    ratios = read('statistical').pivot(index=['case', 'n'], columns='beta', values='variance_total').reset_index()
    ratios['variance_ratio_beta1_beta0'] = ratios[1.] / ratios[0.]
    ratios = ratios.rename(columns={0.: 'variance_beta0', 1.: 'variance_beta1'})
    return {'main_display': pd.DataFrame(metric_table), 'resource_display': roles,
            'selection_frequencies': frequencies, 'headroom_display': headroom,
            'invariance_display': invariants, 'variance_ratios': ratios}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, default=ROOT / 'outputs/analysis')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    tables = build_tables(args.analysis_dir)
    output = args.output_dir or args.analysis_dir
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    print(f'Wrote {len(tables)} display tables.')


if __name__ == '__main__':
    main()
