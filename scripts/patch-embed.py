# ORT Web WASM 에는 GatherBlockQuantized 커널이 없다 → 양자화 임베딩 조회를 원본 임베딩(fp16)의 Gather + Cast(fp32) 로 교체한다.
# 사용: patch-embed.py <dst_dir> <hf_id>   (finalize-genai.py 실행 후)
import sys, os, json, numpy as np, onnx
from onnx import helper, TensorProto
dst, hf_id = sys.argv[1], sys.argv[2]
from huggingface_hub import hf_hub_download
from safetensors import safe_open
try: idx = json.load(open(hf_hub_download(hf_id, 'model.safetensors.index.json'))); fn = idx['weight_map']['model.embed_tokens.weight']
except Exception: fn = 'model.safetensors'
with safe_open(hf_hub_download(hf_id, fn), 'pt') as f: w = f.get_tensor('model.embed_tokens.weight').float().numpy().astype(np.float16)
print('embedding', w.shape)
mp = dst + '/onnx/model_q4.onnx'; m = onnx.load(mp, load_external_data=False); g = m.graph
gbq = [n for n in g.node if n.op_type == 'GatherBlockQuantized']; assert len(gbq) == 1, len(gbq)
out = gbq[0].output[0]; reshape = [n for n in g.node if n.output[0] == gbq[0].input[0]][0]
g.node.remove(gbq[0]); g.node.remove(reshape)
data_path = dst + '/onnx/model_q4.onnx_data'; off = (os.path.getsize(data_path) + 4095) // 4096 * 4096
with open(data_path, 'r+b') as f: f.seek(off); f.write(w.tobytes())
t = onnx.TensorProto(); t.name = 'embed_tokens.weight_fp16'; t.data_type = TensorProto.FLOAT16; t.dims.extend(w.shape); t.data_location = onnx.TensorProto.EXTERNAL
for k, v in (('location', 'model_q4.onnx_data'), ('offset', str(off)), ('length', str(w.nbytes))): e = t.external_data.add(); e.key = k; e.value = v
g.initializer.append(t)
g.node.insert(0, helper.make_node('Gather', ['embed_tokens.weight_fp16', 'input_ids'], ['/model/embed_tokens/Gather/output_fp16'], name='/model/embed_tokens/Gather', axis=0))
g.node.insert(1, helper.make_node('Cast', ['/model/embed_tokens/Gather/output_fp16'], [out], name='/model/embed_tokens/Cast', to=TensorProto.FLOAT))
onnx.save(m, mp); print('patched; data bytes', os.path.getsize(data_path))
