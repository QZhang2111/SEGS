import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from segs.view_utils import is_back_view, structural_guidance_weight

if torch is not None:
    from segs.structural_energy import mean_teacher_features, structural_energy


class StructuralEnergyTests(unittest.TestCase):
    def test_teacher_features_average_topk_preserving_tokens(self):
        if torch is None:
            self.skipTest("torch not installed in current interpreter")

        teacher = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
            ]
        )

        averaged = mean_teacher_features(teacher, topk=3, tokens_per_sample=2)

        expected = torch.tensor([[3.0, 30.0], [4.0, 40.0]])
        self.assertTrue(torch.equal(averaged, expected))

    def test_structural_energy_matches_pca_teacher_mean(self):
        if torch is None:
            self.skipTest("torch not installed in current interpreter")

        feature = torch.tensor([[[[2.0]], [[4.0]]]])
        mean = torch.tensor([1.0, 1.0])
        basis = torch.eye(2)
        teacher = torch.tensor(
            [
                [1.0, 3.0],
                [2.0, 4.0],
                [3.0, 5.0],
            ]
        )

        loss = structural_energy(
            feature,
            mean=mean,
            basis=basis,
            teacher_feature=teacher,
            topk=3,
        )

        self.assertEqual(loss.item(), 1.0)

    def test_back_view_threshold_matches_paper_bin(self):
        self.assertFalse(is_back_view(119.9))
        self.assertTrue(is_back_view(120.0))
        self.assertTrue(is_back_view(-150.0))
        self.assertEqual(structural_guidance_weight(90.0), 0.0)
        self.assertEqual(structural_guidance_weight(120.0), 0.5)
        self.assertEqual(structural_guidance_weight(180.0), 1.0)


if __name__ == "__main__":
    unittest.main()
