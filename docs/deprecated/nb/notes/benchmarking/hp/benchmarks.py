import numpy as np
from .particles import Particle, ParticleSimulator


def random_particles(n: int) -> list[Particle]:
    particles = []
    for _ in range(n):
        x, y, ω = np.random.random(3)
        particles.append(Particle(x, y, ω))
    return particles

def benchmark():
    particles = random_particles(100)
    simulator = ParticleSimulator(particles)
    simulator.evolve(1.0)

def benchmark_memory():
    particles = random_particles(1000000)
    simulator = ParticleSimulator(particles)
    simulator.evolve(0.001)
