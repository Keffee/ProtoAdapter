"""Train with an external OpenOneRec checkpoint and ready-to-use tensor batches."""
import argparse
import json
import math
from pathlib import Path

from .training import EmbeddingRowGuard, component_loss, freeze_backbone


def validate_openonerec(tokenizer, vocab_size):
    tokens = ['<s_a_0>', '<s_b_0>', '<s_c_0>', '<|sid_begin|>', '<|sid_end|>']
    ids = []
    for token in tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if (len(encoded) != 1 or not 0 <= encoded[0] < vocab_size
                or tokenizer.convert_ids_to_tokens(encoded[0]) != token):
            raise ValueError('An external OpenOneRec checkpoint with native SID tokens is required')
        ids.append(encoded[0])
    if len(set(ids)) != len(ids) or min(ids) != ids[0]:
        raise ValueError('OpenOneRec SID vocabulary layout is incompatible')
    vocabulary = tokenizer.get_vocab()
    for block,part in enumerate('abc'):
        for offset in range(8192):
            expected = ids[0] + block * 8192 + offset
            if expected >= vocab_size or vocabulary.get(f'<s_{part}_{offset}>') != expected:
                raise ValueError('OpenOneRec requires three complete contiguous SID vocabulary blocks')
    if min(ids[-2:]) < ids[0] + 3 * 8192:
        raise ValueError('OpenOneRec SID wrappers must follow the item vocabulary')
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('OpenOneRec tokenizer must define padding and EOS tokens')
    return ids[0]


def validate_batch(batch, vocab_size, max_length):
    """Check already prepared tensors without tokenization or transformations."""
    import torch
    if not isinstance(batch, dict):
        raise ValueError('Each training batch must be a dictionary')
    allowed = {'input_ids','attention_mask','labels','variants','bridge_mask'}
    if set(batch) - allowed:
        raise ValueError('Batch contains unsupported metadata or fields')
    for key in ('input_ids', 'attention_mask', 'labels'):
        tensor = batch.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.dtype != torch.long:
            raise ValueError(f'{key} must be a two-dimensional torch.long tensor')
    ids, mask, labels = (batch[k] for k in ('input_ids', 'attention_mask', 'labels'))
    if ids.shape != mask.shape or ids.shape != labels.shape:
        raise ValueError('input_ids, attention_mask and labels must have the same shape')
    if ids.shape[0] < 2 or not 2 <= ids.shape[1] <= max_length:
        raise ValueError('Each batch needs at least two rows and a valid sequence length')
    if torch.any(ids < 0) or torch.any(ids >= vocab_size):
        raise ValueError('input_ids are outside the model vocabulary')
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('attention_mask must contain only zero or one')
    if not torch.all(mask[:,0] == 1) or torch.any(mask[:,1:] > mask[:,:-1]):
        raise ValueError('Only right padding is supported')
    supervised = labels != -100
    if torch.any(supervised & (mask == 0)):
        raise ValueError('All padding labels must be -100')
    if torch.any(supervised & (labels != ids)):
        raise ValueError('Unshifted supervised labels must match input_ids')
    if torch.any(labels[:,0] != -100) or not torch.all(supervised[:,1:].any(dim=1)):
        raise ValueError('Every row needs a masked prefix and supervised next-token labels')
    variants = batch.get('variants')
    if (not isinstance(variants, list) or len(variants) != ids.shape[0]
            or any(v not in ('direct','trace') for v in variants)
            or set(variants) != {'direct','trace'}):
        raise ValueError('Every batch must contain direct and trace variants, one label per row')
    bridge = batch.get('bridge_mask')
    if bridge is not None:
        if not isinstance(bridge, torch.Tensor) or bridge.shape != ids.shape or bridge.dtype != torch.bool:
            raise ValueError('bridge_mask must be a boolean tensor matching input_ids shape')
        trace_rows = torch.tensor([v == 'trace' for v in variants])[:,None]
        if torch.any(bridge & (~supervised | ~trace_rows)):
            raise ValueError('bridge_mask may only mark supervised trace tokens')


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='External OpenOneRec checkpoint directory')
    parser.add_argument('--train-batches', required=True, help='External .pt file with ready tensor batches')
    parser.add_argument('--output-dir', required=True, help='New or empty output directory')
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validate-only', action='store_true', help='Check external inputs without training')
    parser.add_argument('--no-save', action='store_true', help='Skip checkpoint writing for a short validation run')
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if (args.max_steps < 1 or args.max_length < 2 or args.warmup_steps < 0
            or args.warmup_steps > args.max_steps):
        parser.error('Invalid step or length settings')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('--learning-rate must be finite and positive')
    if not all(math.isfinite(x) and x >= 0 for x in (args.beta,args.gamma)):
        parser.error('Loss weights must be finite and nonnegative')
    model_dir = Path(args.model)
    if not model_dir.is_dir() or not (model_dir/'config.json').is_file():
        parser.error('--model must point to an existing external OpenOneRec checkpoint')
    if not Path(args.train_batches).is_file():
        parser.error('--train-batches must point to an existing tensor batch file')
    output = Path(args.output_dir)
    if not args.validate_only and output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('--output-dir must be new or empty')

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True,trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True,trust_remote_code=False)
    start_index = validate_openonerec(tokenizer,config.vocab_size)
    batches = torch.load(args.train_batches, map_location='cpu', weights_only=True)
    if not isinstance(batches,list) or not batches:
        raise ValueError('The batch file must contain a nonempty list of tensor dictionaries')
    for batch in batches:
        validate_batch(batch,config.vocab_size,args.max_length)
        if args.gamma > 0 and ('bridge_mask' not in batch or not batch['bridge_mask'].any()):
            raise ValueError('Positive gamma requires supervised bridge tokens in every batch')
    if args.validate_only:
        print(json.dumps({'status':'valid','batches':len(batches),
                          'start_optimize_embedding_index':start_index}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU and an external OpenOneRec checkpoint are required')
    model = AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0')
    freeze = freeze_backbone(model)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    row_guard = EmbeddingRowGuard(model,start_index)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=0,foreach=False)
    generator = torch.Generator().manual_seed(args.seed)
    order, position = [], 0
    output.mkdir(parents=True,exist_ok=True)
    updates_verified = 0
    with (output/'metrics.jsonl').open('w',encoding='utf-8') as log:
        for step in range(args.max_steps):
            if position == len(order):
                order = torch.randperm(len(batches),generator=generator).tolist()
                position = 0
            source = batches[order[position]]
            position += 1
            batch = {key:source[key].to('cuda:0') for key in ('input_ids','attention_mask','labels')}
            bridge = source.get('bridge_mask')
            if bridge is not None:
                bridge = bridge.to('cuda:0')
            if args.warmup_steps:
                optimizer.param_groups[0]['lr'] = args.learning_rate * min(1.,(step+1)/args.warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result = model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False)
            loss,parts = component_loss(result.logits,batch['labels'],source['variants'],bridge,
                beta=args.beta,gamma=args.gamma)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            row_guard.mask_gradients()
            norm = torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
            if not float(norm) > 0:
                raise RuntimeError('No trainable gradient')
            # Probe actual SID row updates, keeping raw inputs out of the logs.
            ids = torch.unique(batch['input_ids'][batch['input_ids'] >= start_index])
            if ids.numel() == 0:
                raise ValueError('Batch has no native item tokens')
            weights = model.get_input_embeddings().weight
            before = weights[ids].detach().clone()
            optimizer.step()
            row_guard.restore()
            delta = float((weights[ids].detach()-before).abs().max())
            updates_verified += int(delta > 0)
            metrics = {'step':step+1,'loss':float(loss.detach()),'grad_norm':float(norm),
                       'learning_rate':optimizer.param_groups[0]['lr'],
                       'embedding_update_max':delta,**parts}
            log.write(json.dumps(metrics)+'\n')
            log.flush()
            print(json.dumps(metrics),flush=True)
            del result,loss,batch,bridge,before
    if not updates_verified or not row_guard.unchanged():
        raise RuntimeError('Parameter update or frozen vocabulary verification failed')
    if not args.no_save:
        from safetensors.torch import save_file
        state = {name:p.detach().contiguous().cpu() for name,p in model.named_parameters() if p.requires_grad}
        save_file(state,str(output/'trainable.safetensors'))
    summary = {'steps':args.max_steps,'seed':args.seed,'beta':args.beta,'gamma':args.gamma,
               'freeze':freeze,'start_optimize_embedding_index':start_index,
               'embedding_updates_verified':updates_verified,
               'frozen_vocabulary_rows_unchanged':True,
               'embedding_output_tied':model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(),
               'trainable_dtype':'float32',
               'checkpoint_saved':not args.no_save}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
