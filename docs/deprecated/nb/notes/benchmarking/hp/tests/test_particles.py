import math
from hp.particles import Particle, ParticleSimulator


def test_evolve():
    particles = [
        Particle( 0.000,  0.000, +1.0),
        Particle(+0.110, +0.205, +0.3),
        Particle(+0.230, -0.405,  0.0),
        Particle(+0.617, +0.330, -0.1),
    ]

    # Evolve system
    t = 0.1
    simulator = ParticleSimulator(particles)
    simulator.evolve(t)

    # Check expected positions
    def fequal(a, b, eps=1e-6):
        return abs(a - b) < eps

    p0, p1, p2, p3 = particles
    r1 = (0.110 ** 2 + 0.205 ** 2) ** 0.5
    r3 = (0.617 ** 2 + 0.330 ** 2) ** 0.5
    θ1 = math.atan(0.205 / 0.110)
    θ3 = math.atan(0.330 / 0.617)
    
    assert fequal(p0.x,  0.000)
    assert fequal(p0.y,  0.000)
    assert fequal(p1.x, r1 * math.cos(θ1 + 0.3 * t))
    assert fequal(p1.y, r1 * math.sin(θ1 + 0.3 * t))
    assert fequal(p2.x,  0.230)
    assert fequal(p2.y, -0.405)
    assert fequal(p3.x, r3 * math.cos(θ3 - 0.1 * t))
    assert fequal(p3.y, r3 * math.sin(θ3 - 0.1 * t))
