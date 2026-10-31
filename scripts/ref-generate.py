# 원본(PyTorch fp32) greedy 생성 기준값. 같은 프롬프트로 ONNX 변환본과 비교한다.
import sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
hf_id, out = sys.argv[1], sys.argv[2]
P = ['대한민국의 수도는 어디인가요? 한 문장으로 답하세요.', '인공지능을 초등학생에게 두 문장으로 설명해 주세요.', '다음 문장을 존댓말로 바꿔 주세요: 밥 먹었어?', '1부터 5까지 더하면 얼마인가요?', '김치찌개 재료를 세 가지만 알려 주세요.', 'Translate to English: 오늘 날씨가 좋네요.']
tok = AutoTokenizer.from_pretrained(hf_id); model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32, low_cpu_mem_usage=True).eval()
res = []
for p in P:
    ids = tok.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True, return_tensors='pt', return_dict=True)
    with torch.no_grad(): o = model.generate(**ids, max_new_tokens=24, do_sample=False)
    gen = o[0][ids['input_ids'].shape[1]:].tolist(); res.append({'prompt': p, 'prompt_ids': ids['input_ids'][0].tolist(), 'gen_ids': gen, 'text': tok.decode(gen, skip_special_tokens=True)}); print(p, '->', res[-1]['text'], flush=True)
json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
