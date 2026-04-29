import torch

from pygoo.physics.forces import compute_spring_forces


def test_spring_force_pull_direction():
    pos = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    vel = torch.zeros_like(pos)
    edges = torch.tensor([[0, 1]], dtype=torch.int64)
    rest = torch.tensor([1.0], dtype=torch.float32)
    k = torch.tensor([10.0], dtype=torch.float32)

    f = compute_spring_forces(pos, vel, edges, rest, k, 0.0, True, False)
    assert f[0, 0] > 0
    assert f[1, 0] < 0
