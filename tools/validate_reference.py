"""Check shipped reference evidence without claiming to rerun experiments."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    ref=ROOT/'reference_results'
    main_rows=pd.read_csv(ref/'benchmark/task_method_rows.csv')
    assert len(main_rows)==6900
    assert main_rows['task_id'].nunique()==740
    assert main_rows.dataset.nunique()==8
    assert not main_rows.duplicated(['task_id','method']).any()
    followup=pd.read_csv(ref/'theory_followup/three_resource_task_method_rows.csv')
    assert len(followup)==540
    methods=pd.read_csv(ref/'supplement/methods.csv')
    candidates=pd.read_csv(ref/'supplement/candidates.csv')
    helpsteer=pd.read_csv(ref/'supplement/helpsteer_metrics.csv')
    stats=pd.read_csv(ref/'supplement/statistical.csv')
    assert len(methods)==len(candidates)==420
    assert len(methods[['dataset','seed']].drop_duplicates())==60
    assert not methods.duplicated(['dataset','seed','method']).any()
    assert len(helpsteer)==60 and not helpsteer.duplicated(['seed','method']).any()
    assert len(stats)==32
    assert np.isfinite(methods.group_raw_k30).all()
    assert stats.bias_z.abs().max()<4 and stats.variance_z.abs().max()<4
    h=pd.read_parquet(ROOT/'llm/helpsteer/data/helpsteer2.parquet')
    labels=pd.read_parquet(ROOT/'llm/helpsteer/results/pilot_predictions.parquet')
    emb=np.load(ROOT/'llm/helpsteer/data/helpsteer2_embeddings.npz',allow_pickle=False)
    assert h.row_id.is_unique and labels.row_id.is_unique and len(labels)==500
    assert emb['features'].shape==(2839,768)
    assert np.isfinite(emb['features']).all()
    assert set(h.loc[h.role.isin(['base_train','pilot','conformal','test']),'row_id'])==set(emb['row_ids'])
    assert not {'prompt','response'} & set(h.columns)
    print(json.dumps({'status':'PASS','benchmark_tasks':740,'benchmark_method_rows':6900,
                      'followup_method_rows':540,'supplement_tasks':60,'supplement_method_rows':420,
                      'supplement_helpsteer_rows':60,'statistical_settings':32,
                      'helpsteer_cached_features':[2839,768]},indent=2))


if __name__=='__main__':
    main()
