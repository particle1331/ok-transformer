import math
from src.particles import Particle, ParticleSimulator, visualize_simulation


def test_evolve():
    particles = [
        Particle( 0.0,  0.0, +1.0),
        Particle( 0.3, -0.5,  0.0),
        Particle(+0.2, +0.2, +0.3)
    ]

    simulator = ParticleSimulator(particles)
    simulator.evolve(0.1)

    p0, p1, p2 = particles

    def fequal(a, b, eps=1e-6):
        return abs(a - b) < eps

    assert fequal(p0.x,  0.0)
    assert fequal(p0.y,  0.0)
    assert fequal(p1.x,  0.3)
    assert fequal(p1.y, -0.5)
    assert fequal(p2.x, (0.2 ** 2 + 0.2 ** 2) ** 0.5 * math.cos(math.pi / 4 + 0.1 * 0.3))
    assert fequal(p2.y, (0.2 ** 2 + 0.2 ** 2) ** 0.5 * math.sin(math.pi / 4 + 0.1 * 0.3))
