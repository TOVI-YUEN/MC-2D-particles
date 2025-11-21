# GPU-accelerated Metropolis Monte Carlo with temperature cycling and per-particle adaptive step sizes

import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
This script implements a GPU-accelerated Metropolis Monte Carlo sampling
scheme for a 2D charged system consisting of:

- A pair of fixed colloids (oppositely charged)
- Multiple mobile ions undergoing hard-sphere + softened Coulomb interaction
- Sequential single-particle Metropolis MC updates (no random particle selection)
- Per-particle adaptive step sizes (sigma_i) tuned to maintain target acceptance
- Temperature cycling: alternating between high-temperature (exploration)
  and low-temperature (sampling) phases
- GPU is used for all energy, distance, and random-number operations

CPU is used only for:
- Initial random placement of ions (once)
- Bookkeeping of sigma_i and acceptance counts
- Logging and visualization

This version contains full English documentation for GitHub release.
"""

# ================== Device Selection ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================== System Parameters ==================
n_ions = 20            # number of mobile ions
n_cols = 2             # number of fixed colloids
N = n_ions + n_cols    # total particle count

idx_col1 = n_ions      # index of colloid 1
idx_col2 = n_ions + 1  # index of colloid 2

L_box = 100.0          # periodic box size (square box)
R_col = 10.0           # colloid center separation

# Hard-sphere radii
r_ion = 1.0
r_col = 1.0
radii_np = np.zeros(N)
radii_np[:n_ions] = r_ion
radii_np[idx_col1:] = r_col
radii = torch.tensor(radii_np, device=device, dtype=torch.float32)

# Charges: half positive, half negative ions, two large-opposite colloids
q_ion = 1.0
Q_col = 60.0
charge_np = np.zeros(N)

half = n_ions // 2
charge_np[:half] = +q_ion
charge_np[half:n_ions] = -q_ion
charge_np[idx_col1] = +Q_col
charge_np[idx_col2] = -Q_col

charge = torch.tensor(charge_np, device=device, dtype=torch.float32)

# Softened Coulomb interaction parameters
k_c = 20.0
soft_eps = 1.0   # softening length in sqrt(r^2 + soft_eps^2)

# ================== Temperature Cycling Parameters ==================
beta_low = 1.0    # low-temperature phase
beta_high = 0.1   # high-temperature phase
n_cycles = 200    # number of high/low temperature cycles

n_sweeps_high = 50    # number of sweeps at high temperature per cycle
n_sweeps_low  = 100   # number of sweeps at low temperature per cycle

save_every_steps = 10   # record frame every X single-particle moves (low T only)

# ================== Adaptive Per-Ion Step Sizes ==================
adapt_window = 100       # number of attempts before each sigma_i adjustment
target_low = 0.6         # lower bound of target acceptance range
target_high = 0.8        # upper bound of target acceptance range
sigma_min = 0.1
sigma_max = 10.0
adapt_shrink = 0.8       # shrink factor when acceptance too low
adapt_grow   = 1.2       # grow factor when acceptance too high

# Each ion has its own adaptive sigma value
step_sigma_vec = np.full(n_ions, 0.5, dtype=float)
window_accept = np.zeros(n_ions, dtype=int)
window_total = np.zeros(n_ions, dtype=int)

# ================== Periodic Boundary Conditions ==================
def minimum_image(rij: torch.Tensor) -> torch.Tensor:
    """Apply minimum-image convention for 2D periodic boundaries."""
    return rij - torch.round(rij / L_box) * L_box

def wrap_positions(q: torch.Tensor) -> torch.Tensor:
    """Wrap positions into [-L/2, L/2)."""
    return (q + L_box / 2.0) % L_box - L_box / 2.0

# ================== Initial Placement on CPU ==================
def random_place_ions_cpu(n_ions, R_col, safe_factor=1.4, max_attempts=200000):
    """
    Randomly place ions inside the periodic box on CPU, avoiding overlaps
    with colloids and other ions. Used only once at startup.
    """
    ions = []

    col1 = np.array([-R_col / 2.0, 0.0])
    col2 = np.array([+R_col / 2.0, 0.0])

    def min_image_cpu(r):
        return r - np.round(r / L_box) * L_box

    attempts = 0
    while len(ions) < n_ions and attempts < max_attempts:
        attempts += 1

        cand = np.array([
            (np.random.rand() - 0.5) * L_box,
            (np.random.rand() - 0.5) * L_box,
        ])

        ok = True

        # Check overlap with colloids
        for col_pos in (col1, col2):
            rij = cand - col_pos
            rij = min_image_cpu(rij)
            r = np.linalg.norm(rij)
            r_min_col = safe_factor * (r_ion + r_col)
            if r < r_min_col:
                ok = False
                break

        # Check overlap with existing ions
        if ok:
            for p in ions:
                rij = cand - p
                rij = min_image_cpu(rij)
                r = np.linalg.norm(rij)
                r_min_ion = safe_factor * (2 * r_ion)
                if r < r_min_ion:
                    ok = False
                    break

        if ok:
            ions.append(cand)

    if len(ions) < n_ions:
        raise RuntimeError("Failed to place ions within maximum attempts.")

    return np.array(ions), col1, col2

# Generate initial coordinates on CPU
ions_xy_cpu, col1_pos_cpu, col2_pos_cpu = random_place_ions_cpu(n_ions, R_col)

q_np = np.zeros((N, 2), dtype=float)
q_np[:n_ions] = ions_xy_cpu
q_np[idx_col1] = col1_pos_cpu
q_np[idx_col2] = col2_pos_cpu

# Transfer coordinates to GPU\ nq = torch.tensor(q_np, device=device, dtype=torch.float32)
q = wrap_positions(q)

# ================== Fix Colloid Positions ==================
def apply_colloid_constraints(q: torch.Tensor) -> torch.Tensor:
    """Fix colloids to their predefined positions inside the periodic box."""
    q[idx_col1] = torch.tensor([-R_col / 2.0, 0.0], device=device)
    q[idx_col2] = torch.tensor([+R_col / 2.0, 0.0], device=device)
    return q

q = apply_colloid_constraints(q)

# ================== Coulomb Energy of One Particle ==================
def coulomb_energy_of_particle(q: torch.Tensor, i: int) -> torch.Tensor:
    """
    Compute Coulomb energy contribution of particle i
    interacting with all other particles using softened 1/r.

    E_i = sum_j k_c * q_i * q_j / sqrt(r_ij^2 + soft_eps^2)
    """
    qi = q[i].view(1, 2)
    rij = minimum_image(q - qi)
    r2 = torch.sum(rij * rij, dim=1) + 1e-12
    r_soft = torch.sqrt(r2 + soft_eps**2)

    qi_charge = charge[i]
    pair_E = k_c * qi_charge * charge / r_soft

    mask = torch.ones(N, dtype=torch.bool, device=device)
    mask[i] = False

    return torch.sum(pair_E[mask])

# ================== Hard-Sphere Overlap Check ==================
def has_overlap(q: torch.Tensor, i: int, q_proposed_i: torch.Tensor) -> bool:
    """Return True if the proposed position overlaps any other particle."""
    qi = q_proposed_i.view(1, 2)
    rij = minimum_image(q - qi)
    r2 = torch.sum(rij * rij, dim=1)
    r = torch.sqrt(r2 + 1e-12)

    rmin_vec = radii + radii[i]
    mask = torch.ones(N, dtype=torch.bool, device=device)
    mask[i] = False

    return torch.any(r[mask] < rmin_vec[mask]).item()

# ================== Main Temperature-Cycled MC Loop ==================
frames = []
energies = []
accept_count_low = 0
accept_count_high = 0

# Initial total energy\ nU_total = 0.5 * sum(coulomb_energy_of_particle(q, i) for i in range(N))
U_total = U_total.item()
print("Initial total Coulomb energy:", U_total)

single_move_counter = 0

for cycle in range(n_cycles):

    # ---------------- High Temperature Phase ----------------
    beta = beta_high
    for sweep in range(n_sweeps_high):
        for i in range(n_ions):
            single_move_counter += 1

            sigma_i = step_sigma_vec[i]
            E_old_i = coulomb_energy_of_particle(q, i)

            # Random trial move (GPU)
            disp = torch.randn(2, device=device) * sigma_i
            q_old_i = q[i].clone()
            q_new_i = q_old_i + disp
            q_new_i = ((q_new_i + L_box/2) % L_box) - L_box/2

            accepted = False

            if not has_overlap(q, i, q_new_i):
                q[i] = q_new_i
                q = apply_colloid_constraints(q)

                E_new_i = coulomb_energy_of_particle(q, i)
                dU = (E_new_i - E_old_i).item()
                acc_prob = 1.0 if dU <= 0 else math.exp(-beta * dU)

                # GPU RNG
                if torch.rand(1, device=device).item() < acc_prob:
                    U_total += dU
                    accept_count_high += 1
                    accepted = True
                else:
                    q[i] = q_old_i
                    q = apply_colloid_constraints(q)

            # Update adaptive statistics
            window_total[i] += 1
            if accepted:
                window_accept[i] += 1

            # Perform per-ion sigma adjustment
            if window_total[i] >= adapt_window:
                acc_rate_i = window_accept[i] / window_total[i]

                if acc_rate_i < target_low:
                    step_sigma_vec[i] = max(sigma_min, step_sigma_vec[i] * adapt_shrink)
                elif acc_rate_i > target_high:
                    step_sigma_vec[i] = min(sigma_max, step_sigma_vec[i] * adapt_grow)

                print(f"[HighT adapt ion {i}] cycle={cycle}, acc_rate={acc_rate_i:.3f}, sigma={step_sigma_vec[i]:.3f}")

                window_accept[i] = 0
                window_total[i] = 0

    # ---------------- Low Temperature Phase ----------------
    beta = beta_low
    for sweep in range(n_sweeps_low):
        for i in range(n_ions):
            single_move_counter += 1

            sigma_i = step_sigma_vec[i]
            E_old_i = coulomb_energy_of_particle(q, i)

            disp = torch.randn(2, device=device) * sigma_i
            q_old_i = q[i].clone()
            q_new_i = q_old_i + disp
            q_new_i = ((q_new_i + L_box/2) % L_box) - L_box/2

            accepted = False

            if not has_overlap(q, i, q_new_i):
                q[i] = q_new_i
                q = apply_colloid_constraints(q)

                E_new_i = coulomb_energy_of_particle(q, i)
                dU = (E_new_i - E_old_i).item()
                acc_prob = 1.0 if dU <= 0 else math.exp(-beta * dU)

                if torch.rand(1, device=device).item() < acc_prob:
                    U_total += dU
                    accept_count_low += 1
                    accepted = True
                else:
                    q[i] = q_old_i
                    q = apply_colloid_constraints(q)

            # Adaptive bookkeeping
            window_total[i] += 1
            if accepted:
                window_accept[i] += 1

            if window_total[i] >= adapt_window:
                acc_rate_i = window_accept[i] / window_total[i]

                if acc_rate_i < target_low:
                    step_sigma_vec[i] = max(sigma_min, step_sigma_vec[i] * adapt_shrink)
                elif acc_rate_i > target_high:
                    step_sigma_vec[i] = min(sigma_max, step_sigma_vec[i] * adapt_grow)

                print(f"[LowT adapt ion {i}] cycle={cycle}, acc_rate={acc_rate_i:.3f}, sigma={step_sigma_vec[i]:.3f}")

                window_total[i] = 0
                window_accept[i] = 0

            # Record only during low-T phase
            if single_move_counter % save_every_steps == 0:
                frames.append(q.detach().cpu().numpy())
                energies.append(U_total)

print("Simulation finished.")
print("Final total Coulomb energy:", energies[-1] if energies else None)
print("Final per-ion sigmas:", step_sigma_vec)

# Overall acceptance statistics
moves_high = n_cycles * n_sweeps_high * n_ions
moves_low  = n_cycles * n_sweeps_low  * n_ions

print("High-T acceptance rate:", accept_count_high / moves_high)
print("Low-T acceptance rate:",  accept_count_low  / moves_low )

# Convert recorded frames to array
frames = np.array(frames)
n_frames = frames.shape[0]

# ================== Animation ==================
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-L
