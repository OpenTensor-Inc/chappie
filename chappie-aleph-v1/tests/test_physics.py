# tests/test_physics.py
"""
Numerical Verification Suite for Chappie-Aleph-v1 Forward Pass.
Tests: Wave Propagation, Diffusion Smoothing, HH Excitability, Coupled Stability.
"""
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- Path Setup to import chappie_aleph ---
# اضافه کردن پوشه ریشه پروژه به sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from config import ModelConfig, CONFIG, SpatialConfig, PhysicsConfig, SolverConfig
from state import ChappieState, init_state
from solver import simulate_step, simulate_trajectory, inject_tokens, readout_logits
from physics.hodgkin_huxley import compute_ionic_currents, rush_larsen_update

# ---------------------------------------------------------
# 1. Helper: Visualization Utilities
# ---------------------------------------------------------
def plot_spacetime(V_history, dt, dx, title="Voltage Field V(x,t)", save_path=None):
    """Plot Space-Time diagram (Heatmap)."""
    V_arr = np.array(V_history) # (T, N)
    T, N = V_arr.shape
    t_max = T * dt
    x_max = N * dx
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(V_arr.T, aspect='auto', origin='lower', 
                   extent=[0, t_max, 0, x_max], cmap='RdBu_r', 
                   vmin=-80, vmax=50) # HH Voltage range
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Space (x)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='V (mV)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spacetime plot to {save_path}")
    plt.show()

def plot_snapshots(V_history, steps_to_plot, dx, title="Voltage Snapshots"):
    """Plot V(x) at specific time steps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(V_history[0].shape[0]) * dx
    for step in steps_to_plot:
        if step < len(V_history):
            ax.plot(x, V_history[step], label=f't={step*CONFIG.solver.dt:.3f}')
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('V (mV)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

def plot_gates_history(gates_history, gate_names, dt, title="Gating Variables Dynamics"):
    """Plot gating variables m, h, n over time at a specific node (e.g., center)."""
    center_idx = len(gates_history[0][gate_names[0]]) // 2
    T = len(gates_history)
    t_axis = np.arange(T) * dt
    
    fig, axes = plt.subplots(len(gate_names), 1, figsize=(10, 2*len(gate_names)), sharex=True)
    if len(gate_names) == 1: axes = [axes]
    
    for i, name in enumerate(gate_names):
        vals = np.array([g[name][center_idx] for g in gates_history])
        axes[i].plot(t_axis, vals, label=name)
        axes[i].set_ylabel(name)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (t)')
    fig.suptitle(title)
    plt.show()

def create_animation(V_history, dt, dx, save_path="chappie_wave.mp4"):
    """Create MP4 animation of V(x,t). Requires ffmpeg."""
    try:
        V_arr = np.array(V_history)
        T, N = V_arr.shape
        x = np.arange(N) * dx
        
        fig, ax = plt.subplots(figsize=(8, 4))
        line, = ax.plot(x, V_arr[0], 'b-', lw=2)
        ax.set_ylim(-90, 60)
        ax.set_xlim(0, N*dx)
        ax.set_xlabel('Space (x)')
        ax.set_ylabel('V (mV)')
        ax.grid(True, alpha=0.3)
        title = ax.set_title(f'Time: 0.000')
        
        def update(frame):
            line.set_ydata(V_arr[frame])
            title.set_text(f'Time: {frame*dt:.3f} | Step: {frame}')
            return line, title
        
        ani = FuncAnimation(fig, update, frames=T, interval=50, blit=True)
        writer = FFMpegWriter(fps=20, metadata=dict(artist='Chappie-Aleph'), bitrate=1800)
        ani.save(save_path, writer=writer)
        plt.close()
        print(f"Animation saved to {save_path}")
    except Exception as e:
        print(f"Could not create animation (ffmpeg missing?): {e}")

# ---------------------------------------------------------
# 2. Test Scenarios
# ---------------------------------------------------------

def run_test_wave_propagation():
    """Test 1: Pure Wave Equation (Hyperbolic) - No Diffusion, No HH."""
    print("\n" + "="*60)
    print("TEST 1: Pure Wave Propagation (c=1.0, D=0, HH=Off)")
    print("="*60)
    
    cfg = ModelConfig(
        spatial=SpatialConfig(n_nodes=1024, domain_length=10.0),
        physics=PhysicsConfig(
            D_base=0.0, 
            c_base=1.0, 
            channel_configs=() # No HH channels
        ),
        solver=SolverConfig(dt=0.005, n_steps=400, imex_scheme="ARS23") 
    )
    
    state = init_state(jax.random.PRNGKey(0), cfg)
    
    # Inject Gaussian Pulse at center
    N = cfg.spatial.n_nodes
    x = jnp.arange(N) * cfg.spatial.dx
    center = cfg.spatial.domain_length / 2.0  # FIX: cfg.spatial.domain_length
    pulse = 20.0 * jnp.exp(-((x - center)**2) / (2 * 0.1**2)) # Amplitude 20mV
    
    state = ChappieState(V=state.V + pulse, gates=state.gates, t=0.0, step=0)
    
    print(f"Running {cfg.solver.n_steps} steps (dt={cfg.solver.dt})...")
    start = time.time()
    V_hist = [np.array(state.V)]
    gates_hist = [state.gates]
    
    def scan_fn(carry, _):
        new_state = simulate_step(carry, cfg)
        return new_state, (np.array(new_state.V), new_state.gates)
    
    final_state, (V_hist_jax, gates_hist_jax) = jax.lax.scan(scan_fn, state, jnp.arange(cfg.solver.n_steps))
    V_hist = [np.array(state.V)] + list(np.array(V_hist_jax))
    gates_hist = [state.gates] + list(gates_hist_jax)
    
    print(f"Done in {time.time()-start:.2f}s. Final V range: [{final_state.V.min():.2f}, {final_state.V.max():.2f}]")
    
    plot_spacetime(V_hist, cfg.solver.dt, cfg.spatial.dx, "Test 1: Pure Wave Propagation")
    plot_snapshots(V_hist, [0, 100, 200, 300, 399], cfg.spatial.dx, "Wave Snapshots")
    create_animation(V_hist, cfg.solver.dt, cfg.spatial.dx, "test1_wave.mp4")

def run_test_diffusion_smoothing():
    """Test 2: Pure Diffusion (Parabolic) - No Wave, No HH."""
    print("\n" + "="*60)
    print("TEST 2: Pure Diffusion Smoothing (D=0.1, c=0, HH=Off)")
    print("="*60)
    
    cfg = ModelConfig(
        spatial=SpatialConfig(n_nodes=512, domain_length=5.0),
        physics=PhysicsConfig(
            D_base=0.1, 
            c_base=0.0, 
            channel_configs=()
        ),
        solver=SolverConfig(dt=0.01, n_steps=200)
    )
    
    state = init_state(jax.random.PRNGKey(1), cfg)
    
    # Sharp Square Pulse
    N = cfg.spatial.n_nodes
    V = state.V.at[N//4 : N//2].set(50.0) 
    state = ChappieState(V=V, gates=state.gates, t=0.0, step=0)
    
    print(f"Running {cfg.solver.n_steps} steps...")
    start = time.time()
    V_hist = [np.array(state.V)]
    def scan_fn(carry, _):
        new_state = simulate_step(carry, cfg)
        return new_state, np.array(new_state.V)
    final_state, V_hist_jax = jax.lax.scan(scan_fn, state, jnp.arange(cfg.solver.n_steps))
    V_hist += list(np.array(V_hist_jax))
    print(f"Done in {time.time()-start:.2f}s.")
    
    plot_spacetime(V_hist, cfg.solver.dt, cfg.spatial.dx, "Test 2: Diffusion Smoothing")
    plot_snapshots(V_hist, [0, 10, 50, 100, 199], cfg.spatial.dx, "Diffusion Snapshots")
    
    # Mass Conservation Check
    mass = [v.sum() * cfg.spatial.dx for v in V_hist]
    print(f"Mass Conservation: Initial={mass[0]:.4f}, Final={mass[-1]:.4f}, Drift={abs(mass[-1]-mass[0]):.4f}")

def run_test_hh_excitability():
    """Test 3: Single Node HH Dynamics (0D) - Spike Generation."""
    print("\n" + "="*60)
    print("TEST 3: Single Node HH Excitability (No Spatial Coupling)")
    print("="*60)
    
    cfg = ModelConfig(
        spatial=SpatialConfig(n_nodes=10, domain_length=1.0),
        physics=PhysicsConfig(
            D_base=0.0, 
            c_base=0.0,
        ),
        solver=SolverConfig(dt=0.001, n_steps=5000) # DT SMALLER FOR HH STABILITY (Forward Euler on gates)
    )
    
    state = init_state(jax.random.PRNGKey(2), cfg)
    
    # Inject Strong Depolarization at Node 5 to trigger spike
    V = state.V.at[5].set(-10.0) 
    state = ChappieState(V=V, gates=state.gates, t=0.0, step=0)
    
    print(f"Running {cfg.solver.n_steps} steps (5ms simulated)...")
    start = time.time()
    V_hist = [np.array(state.V)]
    gates_hist = [state.gates]
    
    def scan_fn(carry, _):
        new_state = simulate_step(carry, cfg)
        return new_state, (np.array(new_state.V), new_state.gates)
    
    final_state, (V_hist_jax, gates_hist_jax) = jax.lax.scan(scan_fn, state, jnp.arange(cfg.solver.n_steps))
    V_hist = [np.array(state.V)] + list(np.array(V_hist_jax))
    gates_hist = [state.gates] + list(gates_hist_jax)
    print(f"Done in {time.time()-start:.2f}s.")
    
    # Plot Node 5
    node_idx = 5
    t_axis = np.arange(cfg.solver.n_steps + 1) * cfg.solver.dt
    V_node = np.array([v[node_idx] for v in V_hist])
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t_axis, V_node, 'k-')
    axes[0].set_ylabel('V (mV)')
    axes[0].set_title('Test 3: HH Action Potential at Node 5')
    axes[0].grid(True, alpha=0.3)
    
    gate_names = sorted(state.gates.keys())
    for name in gate_names:
        g_vals = np.array([g[name][node_idx] for g in gates_hist])
        axes[1].plot(t_axis, g_vals, label=name)
    axes[1].set_ylabel('Gate Value')
    axes[1].set_xlabel('Time (ms)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.show()

def run_test_coupled_wave_hh():
    """Test 4: Coupled Wave + HH - Spike Propagation (Action Potential Traveling)."""
    print("\n" + "="*60)
    print("TEST 4: Coupled Wave+HH - Traveling Spike (c=0.5, D=0.01)")
    print("="*60)
    
    cfg = ModelConfig(
        spatial=SpatialConfig(n_nodes=1024, domain_length=20.0),
        physics=PhysicsConfig(
            D_base=0.01,    
            c_base=0.5,     
        ),
        solver=SolverConfig(dt=0.001, n_steps=3000) # DT SMALLER FOR HH STABILITY
    )
    
    state = init_state(jax.random.PRNGKey(3), cfg)
    
    # Trigger Spike at Left Boundary (x=0)
    N = cfg.spatial.n_nodes
    V = state.V.at[:10].set(30.0) 
    state = ChappieState(V=V, gates=state.gates, t=0.0, step=0)
    
    print(f"Running {cfg.solver.n_steps} steps...")
    start = time.time()
    V_hist = [np.array(state.V)]
    gates_hist = [state.gates]
    
    def scan_fn(carry, _):
        new_state = simulate_step(carry, cfg)
        return new_state, (np.array(new_state.V), new_state.gates)
    
    final_state, (V_hist_jax, gates_hist_jax) = jax.lax.scan(scan_fn, state, jnp.arange(cfg.solver.n_steps))
    V_hist = [np.array(state.V)] + list(np.array(V_hist_jax))
    gates_hist = [state.gates] + list(gates_hist_jax)
    print(f"Done in {time.time()-start:.2f}s.")
    
    plot_spacetime(V_hist, cfg.solver.dt, cfg.spatial.dx, "Test 4: Traveling Action Potential")
    plot_snapshots(V_hist, [0, 500, 1000, 2000, 2999], cfg.spatial.dx, "Traveling Spike Snapshots")
    create_animation(V_hist, cfg.solver.dt, cfg.spatial.dx, "test4_traveling_spike.mp4")
    
    plot_gates_history(gates_hist, sorted(state.gates.keys()), cfg.solver.dt, "Gates at x=10 during Spike Pass")

# ---------------------------------------------------------
# 3. Main Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    print("JAX Devices:", jax.devices())
    print("Default Backend:", jax.default_backend())
    
    # Enable 64-bit for numerical verification if needed (slower)
    # jax.config.update("jax_enable_x64", True)
    
    # Run Tests Sequentially
    run_test_wave_propagation()
    run_test_diffusion_smoothing()
    run_test_hh_excitability()
    run_test_coupled_wave_hh()
    
    print("\n" + "="*60)
    print("ALL PHYSICS TESTS COMPLETED.")
    print("Check plots for: Wave splitting, Diffusion smoothing, HH Spike, Traveling Wave.")
    print("="*60)