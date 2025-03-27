import numpy as np


class Particle:    
    def __init__(self, x, y, angular_vel):
        self.x = x
        self.y = y
        self.ω = angular_vel


class ParticleSimulator:
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
