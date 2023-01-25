from celluloid import Camera
import matplotlib.pyplot as plt


def visualize_simulation(
    simulator, savename='particles.gif', timesteps=10, fps=10, dpi=400):
    
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
