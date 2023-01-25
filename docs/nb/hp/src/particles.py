import numpy as np


class Particle:
    __slots__ = ('x', 'y', 'ω')

    def __init__(self, x, y, angular_vel):
        self.x = x
        self.y = y
        self.ω = angular_vel


class ParticleSimulatorSlow:
    def __init__(self, particles: list[Particle], h=1e-5):
        self.h = h  # Euler-method increment 
        self.particles = particles

    def evolve(self, t: float):
        """Evolve system from t=0 to t=t."""
        
        n_steps = int(t / self.h)
        for _ in range(n_steps):
            for p in self.particles:
                self.update_particle(p)

    def update_particle(self, p: Particle):
        """Evolve particle with Δt = h."""

        vx = -p.y * p.ω
        vy =  p.x * p.ω
        dx = vx * self.h
        dy = vy * self.h
        p.x += dx
        p.y += dy


class ParticleSimulatorFast:
    def __init__(self, particles: list[Particle], h=1e-5):
        self.h = h  # Euler-method increment 
        self.particles = particles
        self.data = np.array([[p.x, p.y, p.ω] for p in particles], dtype=np.float64)

    def evolve(self, t: float):
        """Evolve system from t=0 to t=t."""

        n_steps = int(t / self.h)
        for _ in range(n_steps):
            self.update_data()

        for i, p in enumerate(self.particles):
            p.x, p.y = self.data[i, [0, 1]]

    def update_data(self):
        """Evolve particle with Δt = h."""

        x = self.data[:, [0]]
        y = self.data[:, [1]]
        ω = self.data[:, [2]]
        vx = -y * ω
        vy =  x * ω
        dx = vx * self.h
        dy = vy * self.h
        self.data[:, [0]] += dx
        self.data[:, [1]] += dy


# ParticleSimulator = ParticleSimulatorSlow
ParticleSimulator = ParticleSimulatorFast
