# 직접 변환: HF 원본(safetensors) → ONNX(fp32, KV cache 포함 merged) → MatMulNBits 4bit. Transformers.js 가 읽는 구조로 저장.
# 사용: convert-llm.py <hf_id> <out_dir> [block_size]
import sys, os, json, time, shutil, gc
hf_id, out = sys.argv[1], sys.argv[2]; bs = int(sys.argv[3]) if len(sys.argv) > 3 else 32
os.makedirs(out + '/onnx', exist_ok=True); tmp = out + '/_fp32'; rep = {'id': hf_id, 'steps': {}}; t0 = time.time()
from huggingface_hub import snapshot_download
from transformers import AutoConfig
rev = snapshot_download(hf_id, allow_patterns=['*.json', '*.model', '*.txt', '*.py', 'tokenizer*', '*.jinja'], local_dir=out + '/_meta')
import optimum.exporters.onnx.model_configs
from optimum.exporters.onnx import main_export
main_export(model_name_or_path=hf_id, output=tmp, task='text-generation-with-past', library_name='transformers', do_validation=False, no_post_process=False, dtype='fp32', device='cpu')
rep['steps']['export_sec'] = round(time.time() - t0)
fp32 = [f for f in os.listdir(tmp) if f.endswith('.onnx')]; rep['exported'] = fp32
src = tmp + '/decoder_model_merged.onnx' if 'decoder_model_merged.onnx' in fp32 else tmp + '/model.onnx'
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
m = onnx.load(src, load_external_data=True); gc.collect()
q = MatMulNBitsQuantizer(m, bits=4, accuracy_level=4, algo_config=DefaultWeightOnlyQuantConfig(block_size=bs, is_symmetric=True)); q.process()
onnx.save(q.model.model, out + '/onnx/model_q4.onnx', save_as_external_data=True, all_tensors_to_one_file=True, location='model_q4.onnx_data')
rep['steps']['total_sec'] = round(time.time() - t0)
rep['sizeBytes'] = {f: os.path.getsize(out + '/onnx/' + f) for f in os.listdir(out + '/onnx')}
for f in os.listdir(out + '/_meta'):
    if f.endswith(('.json', '.model', '.txt', '.jinja')) or f.startswith('tokenizer'): shutil.copy(out + '/_meta/' + f, out + '/' + f)
shutil.rmtree(tmp, ignore_errors=True); json.dump(rep, open(out + '/conversion.json', 'w'), indent=1); print(json.dumps(rep, indent=1))
