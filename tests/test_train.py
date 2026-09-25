import importlib.util
import unittest

from protoadapter.train import validate_openonerec, validate_batch, argument_parser


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    tokens = {f'<s_{part}_{i}>':100 + block*8192 + i
              for block,part in enumerate('abc') for i in range(8192)}
    tokens.update({'<|sid_begin|>':24676,'<|sid_end|>':24677})

    def get_vocab(self):
        return dict(self.tokens)

    def encode(self, token, add_special_tokens=False):
        return [self.tokens[token]]

    def convert_ids_to_tokens(self, token_id):
        return next(k for k,v in self.tokens.items() if v == token_id)


class DependencyTests(unittest.TestCase):
    def test_requires_native_openonerec_tokens(self):
        self.assertEqual(validate_openonerec(Tokenizer(), 25000), 100)

    def test_generic_tokenizer_is_rejected(self):
        class Generic(Tokenizer):
            def encode(self, token, add_special_tokens=False):
                return [12, 13, 14]
        with self.assertRaisesRegex(ValueError, 'OpenOneRec'):
            validate_openonerec(Generic(), 25000)

    def test_incomplete_sid_vocabulary_is_rejected(self):
        class Incomplete(Tokenizer):
            def get_vocab(self):
                vocabulary = super().get_vocab()
                del vocabulary['<s_b_719>']
                return vocabulary
        with self.assertRaisesRegex(ValueError,'OpenOneRec'):
            validate_openonerec(Incomplete(),25000)

    def test_required_external_paths(self):
        args = argument_parser().parse_args(['--model','external/model',
            '--train-batches','external/batches.pt','--output-dir','outputs/run'])
        self.assertEqual(args.max_steps, 1000)
        self.assertFalse(args.validate_only)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is required')
class BatchTests(unittest.TestCase):
    def batch(self):
        import torch
        return {'input_ids': torch.tensor([[1,2,3],[1,4,5]]),
                'attention_mask': torch.ones(2,3,dtype=torch.long),
                'labels': torch.tensor([[-100,2,3],[-100,4,5]]),
                'variants': ['direct','trace']}

    def test_consumes_ready_tensors_without_transforming(self):
        import torch
        batch = self.batch()
        before = batch['labels'].clone()
        validate_batch(batch, 100, 64)
        self.assertTrue(torch.equal(before,batch['labels']))

    def test_supervised_labels_must_match_input(self):
        batch = self.batch()
        batch['labels'][0,1] = 9
        with self.assertRaisesRegex(ValueError,'labels'):
            validate_batch(batch,100,64)

    def test_padded_tokens_cannot_be_supervised(self):
        batch = self.batch()
        batch['attention_mask'][0,-1] = 0
        with self.assertRaisesRegex(ValueError,'padding'):
            validate_batch(batch,100,64)

    def test_zero_supervision_row_is_rejected(self):
        batch = self.batch()
        batch['labels'][1,:] = -100
        with self.assertRaisesRegex(ValueError,'supervised'):
            validate_batch(batch,100,64)

    def test_bridge_must_mark_trace_supervision(self):
        import torch
        batch = self.batch()
        batch['bridge_mask'] = torch.tensor([[0,1,0],[0,0,0]],dtype=torch.bool)
        with self.assertRaisesRegex(ValueError,'bridge'):
            validate_batch(batch,100,64)

    def test_wrong_shape_or_out_of_vocab_rejected(self):
        batch = self.batch()
        batch['input_ids'][0,0] = 101
        with self.assertRaisesRegex(ValueError,'vocabulary'):
            validate_batch(batch,100,64)
        batch = self.batch()
        batch['labels'] = batch['labels'][:,:-1]
        with self.assertRaisesRegex(ValueError,'shape'):
            validate_batch(batch,100,64)

    def test_both_objectives_required_per_batch(self):
        batch = self.batch()
        batch['variants'] = ['direct','direct']
        with self.assertRaisesRegex(ValueError,'direct and trace'):
            validate_batch(batch,100,64)


if __name__ == '__main__':
    unittest.main()
