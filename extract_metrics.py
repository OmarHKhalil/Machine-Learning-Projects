import json
import re
from pathlib import Path
root = Path('e:/Machine-Learning-Projects')
metric_re = re.compile(r'(?i)(accuracy|precision|recall|f1-score|f1|rmse|r2|silhouette|mae|mse)')
projects = [p for p in sorted(root.iterdir()) if p.is_dir() and (p/'README.md').exists()]
all_metrics = {}
for proj in projects:
    recs = []
    for nb_path in sorted(proj.glob('*.ipynb')):
        data = json.loads(nb_path.read_text(encoding='utf-8'))
        for idx, cell in enumerate(data.get('cells', [])):
            if cell.get('cell_type') != 'code':
                continue
            out_texts = []
            for out in cell.get('outputs', []):
                if 'text' in out:
                    out_texts.append(''.join(out['text']))
                elif 'data' in out and isinstance(out['data'], dict):
                    tp = out['data'].get('text/plain')
                    if tp is not None:
                        out_texts.append(''.join(tp) if isinstance(tp, (list, tuple)) else tp)
            out = '\n'.join(out_texts).replace('\r', '')
            if not out.strip():
                continue
            if metric_re.search(out):
                recs.append({'notebook': nb_path.name, 'cell': idx, 'out': out})
    all_metrics[proj.name] = recs
with open(root/'metric_extraction.json', 'w', encoding='utf-8') as f:
    json.dump(all_metrics, f, ensure_ascii=False, indent=2)
print('wrote', root/'metric_extraction.json')
