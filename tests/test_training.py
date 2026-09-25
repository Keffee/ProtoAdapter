import math
import unittest
import importlib.util

from protoadapter.training import component_loss, freeze_backbone, EmbeddingRowGuard


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is required')
class LossTests(unittest.TestCase):
    def test_loss_normalized_per_component_with_shift(self):
        import torch
        logits = torch.zeros(2, 4, 3, requires_grad=True)
        labels = torch.tensor([[-100, 1, 2, -100], [-100, 1, 2, 0]])
        bridge = torch.tensor([[0,0,0,0], [0,0,1,0]], dtype=torch.bool)
        loss, parts = component_loss(logits, labels, ['direct','trace'], bridge, beta=2., gamma=.5)
        self.assertAlmostEqual(loss.item(), 3.5 * math.log(3), places=5)
        self.assertEqual(parts['direct_tokens'], 2)
        self.assertEqual(parts['trace_tokens'], 3)
        self.assertEqual(parts['bridge_tokens'], 1)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_no_supervised_tokens_fails(self):
        import torch
        with self.assertRaisesRegex(ValueError, 'supervised'):
            component_loss(torch.zeros(1,3,2), torch.full((1,3),-100), ['direct'])

    def test_nonuniform_logits_predict_next_token_not_same_token(self):
        import torch
        logits = torch.tensor([[[0., 4., 0.],[0., 0., 4.],[4., 0., 0.]]])
        labels = torch.tensor([[-100,1,2]])
        loss,_ = component_loss(logits,labels,['direct'])
        self.assertAlmostEqual(loss.item(),math.log(1+2*math.exp(-4)),places=6)

    def test_backbone_freeze_matches_native_parameter_names(self):
        import torch
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(5,3)
                self.layers = torch.nn.Linear(3,3)
                self.lm_head = torch.nn.Linear(3,5,bias=False)
        model = Toy()
        info = freeze_backbone(model)
        self.assertEqual(info['trainable_names'], ['embed_tokens.weight','lm_head.weight'])
        self.assertFalse(model.layers.weight.requires_grad)

    def test_frozen_vocabulary_rows_survive_optimizer_weight_decay(self):
        import torch
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(5,3)
                self.lm_head = torch.nn.Linear(3,5,bias=False)
                self.lm_head.weight = self.embed_tokens.weight
            def get_input_embeddings(self):
                return self.embed_tokens
            def get_output_embeddings(self):
                return self.lm_head
        model = Toy()
        guard = EmbeddingRowGuard(model, 3)
        before = model.embed_tokens.weight.detach().clone()
        opt = torch.optim.AdamW(model.parameters(),lr=.1,weight_decay=.1)
        model.embed_tokens.weight.sum().backward()
        guard.mask_gradients()
        self.assertTrue(torch.equal(model.embed_tokens.weight.grad[:3],torch.zeros(3,3)))
        opt.step()
        guard.restore()
        self.assertTrue(guard.unchanged())
        self.assertTrue(torch.equal(model.embed_tokens.weight[:3],before[:3]))
        self.assertFalse(torch.equal(model.embed_tokens.weight[3:],before[3:]))


if __name__ == '__main__':
    unittest.main()
