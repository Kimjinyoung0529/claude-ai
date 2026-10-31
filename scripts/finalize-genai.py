# genai 빌더 출력을 Transformers.js 로컬 모델 구조로 정리: onnx/model_q4.onnx + model_q4.onnx_data (외부 데이터 위치 갱신)
import sys, os, shutil, json, onnx
src, hf_id, dst = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dst + '/onnx', exist_ok=True)
m = onnx.load(src + '/model.onnx', load_external_data=False)
n = 0
for t in m.graph.initializer:
    for e in t.external_data:
        if e.key == 'location': e.value = 'model_q4.onnx_data'; n += 1
# Transformers.js 4.x 는 KV 캐시 모양을 ONNX 입력 메타데이터에서 읽고 batch_size 외 기호 차원은 0으로 만든다 → 헤드 차원을 숫자로 고정
cfg = json.load(open(src + '/genai_config.json'))['model']['decoder']; hd = cfg['head_size']; fixed = 0
for v in list(m.graph.input) + list(m.graph.output):
    if v.name.startswith(('past_key_values.', 'present.')):
        d = v.type.tensor_type.shape.dim
        if len(d) == 4 and d[3].dim_param: d[3].ClearField('dim_param'); d[3].dim_value = hd; fixed += 1
print('kv head dim fixed:', fixed, '->', hd)
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
# Transformers.js 는 tokenizer_config.json 의 chat_template 을 읽는다(별도 .jinja 파일은 읽지 않음)
tc = json.load(open(dst + '/tokenizer_config.json')); j = dst + '/chat_template.jinja'
if 'chat_template' not in tc and os.path.exists(j): tc['chat_template'] = open(j).read(); json.dump(tc, open(dst + '/tokenizer_config.json', 'w'), ensure_ascii=False, indent=1); print('chat_template embedded')
# 브라우저(ORT Web)는 외부 데이터 파일을 명시해야 마운트한다: 청크 1개(<2GB)
cj = json.load(open(dst + '/config.json')); cj.setdefault('transformers.js_config', {})['use_external_data_format'] = {'model_q4.onnx': 1}
json.dump(cj, open(dst + '/config.json', 'w'), ensure_ascii=False, indent=1); print('use_external_data_format set')
