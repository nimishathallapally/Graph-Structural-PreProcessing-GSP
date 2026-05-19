import json, os

datasets = {
    'ML-1M':  'output/sweep_ml1m',
    'ML-25M': 'output/sweep_ml25m_ordered',
    'Yelp':   'output/sweep_yelp',
}

MODELS = ['lightgcn', 'gcn', 'graphsage', 'gat']
MNAMES = {'lightgcn': 'LightGCN', 'gcn': 'GCN', 'graphsage': 'GraphSAGE', 'gat': 'GAT'}
GSP_KEY = 'gsp_projected'

for ds_name, ds_dir in datasets.items():
    print(f'\n========== {ds_name} ==========')
    for mode in ['forman_ricci', 'cosine']:
        for frac in ['025', '05', '075', '10']:
            for ms in ['1', '3', '5']:
                folder = f'{ds_dir}/{mode}_frac{frac}_ms{ms}'
                fpath = f'{folder}/full_results.json'
                if not os.path.exists(fpath):
                    continue
                with open(fpath) as f:
                    data = json.load(f)
                gsp = data.get('gsp', {})
                node_red = gsp.get('compression_ratio_pct', 0)
                edge_red = gsp.get('bipartite_edge_reduction_pct', 0)
                prepro   = gsp.get('gsp_preprocessing_time_s', 0)
                label = f'{mode}|frac{frac}|ms{ms}'
                print(f'\n  [{label}]  nodes:{node_red:.2f}%  edges:{edge_red:.2f}%  prepro:{prepro:.1f}s')

                metrics_by = {}
                for m in data.get('metrics', []):
                    key = (m.get('model'), m.get('run_type'))
                    metrics_by[key] = m

                for model in MODELS:
                    base  = metrics_by.get((model, 'baseline'), {})
                    gsp_m = metrics_by.get((model, GSP_KEY), {})
                    if not base and not gsp_m:
                        continue
                    mn = MNAMES[model]
                    b_n = base.get('NDCG@10', float('nan'))
                    g_n = gsp_m.get('NDCG@10', float('nan'))
                    b_r = base.get('Recall@10', float('nan'))
                    g_r = gsp_m.get('Recall@10', float('nan'))
                    b_p = base.get('Precision@10', float('nan'))
                    g_p = gsp_m.get('Precision@10', float('nan'))
                    b_t = base.get('training_time_s', float('nan'))
                    g_t = gsp_m.get('training_time_s', float('nan'))
                    print(f'    {mn:12s}  base: NDCG={b_n:.4f} Rec={b_r:.4f} Prec={b_p:.4f} time={b_t:.1f}s')
                    print(f'    {mn:12s}   gsp: NDCG={g_n:.4f} Rec={g_r:.4f} Prec={g_p:.4f} time={g_t:.1f}s')

