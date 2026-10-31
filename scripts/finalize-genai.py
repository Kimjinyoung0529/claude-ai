# genai 빌더 출력을 Transformers.js 로컬 모델 구조로 정리: onnx/model_q4.onnx + model_q4.onnx_data (외부 데이터 위치 갱신)
import sys, os, shutil, json, onnx
src, hf_id, dst = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dst + '/onnx', exist_ok=True)
m = onnx.load(src + '/model.onnx', load_external_data=False)
n = 0
for t in m.graph.initializer:
    for e in t.external_data:
        if e.key == 'location': e.value = 'model_q4.onnx_data'; n += 1
onnx.save(m, dst + '/onnx/model_q4.onnx')
shutil.copy(src + '/model.onnx.data', dst + '/onnx/model_q4.onnx_data')
from huggingface_hub import hf_hub_download
for f in ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja']:
    for base in (src, None):
        try:
            p = src + '/' + f if base else hf_hub_download(hf_id, f)
            if os.path.exists(p): shutil.copy(p, dst + '/' + f); break
        except Exception: pass
print('external refs patched:', n, os.listdir(dst), os.listdir(dst + '/onnx'))
