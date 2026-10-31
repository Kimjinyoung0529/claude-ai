// 변환본(ONNX int4)을 Transformers.js(onnxruntime-node CPU)로 greedy 생성해 원본 PyTorch fp32 기준값과 토큰 단위로 비교한다.
// 사용: node verify-onnx.mjs <modelDir(.cache/converted/xxx)> <ref.json> <report.json>
import { AutoTokenizer, AutoModelForCausalLM, env } from '@huggingface/transformers'; import { readFileSync, writeFileSync } from 'node:fs'; import { dirname, basename } from 'node:path';
const [dir, refFile, outFile] = process.argv.slice(2); env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = dirname(dir) + '/'; const name = basename(dir);
const ref = JSON.parse(readFileSync(refFile)); const tok = await AutoTokenizer.from_pretrained(name); const t0 = Date.now();
const model = await AutoModelForCausalLM.from_pretrained(name, { dtype: 'q4', device: 'cpu' }); const loadMs = Date.now() - t0; const rows = [];
for (const r of ref) {
  const inputs = tok.apply_chat_template([{ role: 'user', content: r.prompt }], { add_generation_prompt: true, return_dict: true });
  const idsMatch = JSON.stringify(Array.from(inputs.input_ids.data, Number)) === JSON.stringify(r.prompt_ids);
  const t1 = Date.now(); const out = await model.generate({ ...inputs, max_new_tokens: 24, do_sample: false }); const ms = Date.now() - t1;
  const all = Array.from(out.data ?? out[0].data, Number); const gen = all.slice(inputs.input_ids.dims[1]);
  let common = 0; while (common < Math.min(gen.length, r.gen_ids.length) && gen[common] === r.gen_ids[common]) common++;
  rows.push({ prompt: r.prompt, promptTokensIdentical: idsMatch, refText: r.text, onnxText: tok.decode(gen, { skip_special_tokens: true }), commonPrefixTokens: common, refLen: r.gen_ids.length, firstTokenMatch: gen[0] === r.gen_ids[0], ms });
  console.log(common + '/' + r.gen_ids.length, JSON.stringify(rows.at(-1).onnxText.slice(0, 60)), '| ref:', JSON.stringify(r.text.slice(0, 60)));
}
const sum = { model: name, loadMs, prompts: rows.length, firstTokenAgree: rows.filter(x => x.firstTokenMatch).length, meanCommonPrefixFrac: +(rows.reduce((s, x) => s + x.commonPrefixTokens / x.refLen, 0) / rows.length).toFixed(3), allPromptTokensIdentical: rows.every(x => x.promptTokensIdentical), rows };
writeFileSync(outFile, JSON.stringify({ note: 'Node onnxruntime-node CPU, greedy 24토큰, 기준=PyTorch fp32. 정확도 벤치마크가 아니라 변환 동등성 점검.', ...sum }, null, 2)); console.log(JSON.stringify({ ...sum, rows: undefined }));
