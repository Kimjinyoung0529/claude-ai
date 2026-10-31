# 후보 모델: transformers 가 구조를 알고 있는지, optimum ONNX exporter 가 지원하는지 실제로 확인.
import json, sys
from transformers import AutoConfig
import optimum.exporters.onnx.model_configs
from optimum.exporters.tasks import TasksManager
ids = json.load(open('reports/model-audit.json'))
out = []
for g in ('experimental', 'korean', 'base'):
    for m in ids['groups'][g]:
        r = {'id': m['id'], 'group': g, 'modelType': m.get('modelType')}
        try:
            c = AutoConfig.from_pretrained(m['id'], trust_remote_code=False)
            r['transformersConfig'] = type(c).__name__; r['hidden'] = getattr(c, 'hidden_size', None) or getattr(getattr(c, 'text_config', None), 'hidden_size', None)
        except Exception as e:
            r['transformersConfig'] = None; r['configError'] = str(e)[:160]
        mt = (r['modelType'] or '').replace('_', '-')
        try:
            TasksManager.get_exporter_config_constructor('onnx', task='text-generation-with-past', model_type=mt, library_name='transformers'); r['optimumOnnx'] = True
        except Exception as e:
            r['optimumOnnx'] = False; r['exporterError'] = str(e)[:140]
        out.append(r); print(r['id'], r['modelType'], 'cfg=' + str(bool(r['transformersConfig'])), 'optimum=' + str(r['optimumOnnx']), flush=True)
json.dump(out, open('reports/model-probe.json', 'w'), ensure_ascii=False, indent=1)
