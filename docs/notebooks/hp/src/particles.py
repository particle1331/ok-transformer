import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera


# class Particle:
#     def __init__(self, x, y, angular_vel):
#         self.x = x
#         self.y = y
#         self.ω = angular_vel


class Particle:
    __slots__ = ('x', 'y', 'ω') 

    def __init__(self, x, y, angular_vel):
        self.x = x
        self.y = y
        self.ω = angular_vel


# class ParticleSimulator:
#     def __init__(self, particles: list[Particle], h=1e-5):
#         self.h = h  # Euler-method increment 
#         self.particles = particles

#     def evolve(self, t: float):
#         """Evolve system from t=0 to t=t."""
        
#         n_steps = int(t / self.h)
#         for _ in range(n_steps):
#             for p in self.particles:
#                 self.update_particle(p)

#     def update_particle(self, p: Particle):
#         """Evolve particle with Δt = h."""

#         vx = -p.y * p.ω
#         vy =  p.x * p.ω
#         dx = vx * self.h
#         dy = vy * self.h
#         p.x += dx
#         p.y += dy


class ParticleSimulator:
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


def visualize_simulation(simulator, timesteps=10, fps=10, dpi=400, savename='particles.gif'):
    
    # Get true path radius for each particle
    paths = {id(p): (p.x ** 2 + p.y ** 2) ** 0.5 for p in simulator.particles}
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    camera = Camera(fig)

    for t in range(timesteps):
        
        ax.annotate(f"t={t}", [0.8, 0.8])
        X = [p.x for p in simulator.particles]
        Y = [p.y for p in simulator.particles]
        colors = [f"C{j}" for j in range(len(X))]
        ax.scatter(X, Y, edgecolors="k", color=colors, zorder=2);
        
        # plot true paths (divergence with large delta in Euler-method)
        for p in simulator.particles:
            r = paths[id(p)]
            c = plt.Circle((0, 0), r, facecolor='None', edgecolor='black', linestyle='dashed', zorder=1)
            ax.add_patch(c);
        
        camera.snap()
        simulator.evolve(1)  # => 1 time units

    animation = camera.animate()
    animation.save(savename, fps=fps)


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
    particles = random_particles(100000)
    simulator = ParticleSimulator(particles)
    simulator.evolve(0.001)
