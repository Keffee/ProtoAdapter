# ProtoAdapter

**OpenOneRec is required.** Use an externally installed OpenOneRec environment and a local OpenOneRec checkpoint with its native SID tokenizer. Generic language-model checkpoints are unsupported. OpenOneRec source, weights, datasets, and data-preparation code are not included in this repository.

## Train

Python 3.10+, PyTorch 2.5+, Transformers 5.3.x and a CUDA GPU are required.

```bash
pip install -e '.[train]'

CUDA_VISIBLE_DEVICES=0 protoadapter-train \
  --model /path/to/external/OpenOneRec-checkpoint \
  --train-batches /path/to/external/train_batches.pt \
  --output-dir outputs/run \
  --max-steps 1000 \
  --learning-rate 1e-5 \
  --warmup-steps 50 \
  --beta 1.0 --gamma 0.0
```

Supply existing, tokenized and batched training inputs. `--train-batches` must point to a PyTorch file containing a nonempty list of dictionaries with:

- `input_ids`, `attention_mask`, `labels`: `torch.long` tensors of shape `[batch, sequence]`; right padding; unshifted labels with prompt/padding positions set to `-100`.
- `variants`: one `"direct"` or `"trace"` string per row; both variants must appear in each batch.
- `bridge_mask` (optional): a boolean tensor of the same shape marking supervised bridge tokens in trace rows; required when `--gamma` is positive.

No downloading, tokenization, data conversion, trace generation or dataset construction is performed. Training freezes the Transformer and ordinary vocabulary rows, and optimizes the item-token embedding/output rows. Each loss component is normalized by its supervised token count.

Use `--validate-only` to check the external inputs without training. For a one-step check, use `--max-steps 1 --warmup-steps 0 --no-save`. The output directory must be new or empty. Normal runs write `trainable.safetensors`, `metrics.jsonl`, and `summary.json`; the saved tensors must be applied to the same external base checkpoint.
