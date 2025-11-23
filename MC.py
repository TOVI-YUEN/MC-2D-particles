# GPU-accelerated Metropolis MC + HMC with Lennard-Jones soft-sphere repulsion
# (temperature cycling + per-particle adaptive step sizes)

import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
Two-stage sampling scheme for a 2D charged colloid–ion system:

1. Metropolis Monte Carlo (MC) pre-equilibration
   - Two fixed colloids (oppositely charged)
   - Multiple mobile ions (hard-sphere + softened Coulomb interaction)
   - Sequential single-particle Metropolis MC updates (no random particle selection)
   - Per-particle adaptive step sizes (sigma_i) tuned to maintain target acceptance
   - Temperature cycling: alternating between high-temperature (exploration)
     and low-temperature (sampling) phases
   - GPU is used for energy, distance, and random-number operations

2. Hamiltonian Monte Carlo (HMC) production sampling
   - Starts from the final configuration of the MC stage
   - All ions move simultaneously (full-vector HMC update)
   - Colloids remain fixed in space
   - Total potential = softened Coulomb + Lennard-Jones (12-6) soft-sphere repulsion
   - LJ parameters chosen such that sigma_ij ~ hard-sphere contact distance
   - Standard leapfrog integrator + Metropolis acceptance

CPU is used only for:
- Initial random placement of ions (once)
- Bookkeeping of sigma_i and acceptance counts in MC
- Logging and visualization

This version contains full English documentation suitable for GitHub release.
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

# Lennard-Jones parameters for soft-sphere repulsion (HMC only)
# U_LJ(r) = 4 * eps_LJ * [ (sigma / r)^12 - (sigma / r)^6 ]
# with sigma_ij chosen from radii[i] + radii[j].
eps_LJ = 1.0

# ================== Temperature Cycling Parameters (MC) ==================
beta_low = 1.0    # low-temperature phase (sampling)
beta_high = 0.1   # high-temperature phase (exploration)

n_cycles = 200    # number of high/low temperature cycles

n_sweeps_high = 50    # number of sweeps at high temperature per cycle
n_sweeps_low  = 100   # number of sweeps at low temperature per cycle

save_every_steps = 10   # record frame every X single-particle moves (low T only)

# ================== Adaptive Per-Ion Step Sizes (MC) ==================
adapt_window = 100       # number of attempts before each sigma_i adjustment
target_low = 0.6         # lower bound of target acceptance range
target_high = 0.8        # upper bound of target acceptance range
sigma_min = 0.1
sigma_max = 10.0
adapt_shrink = 0.8       # shrink factor when acceptance too low
adapt_grow   = 1.2       # grow factor when acceptance too high

# Each ion has its own adaptive sigma value (CPU arrays are sufficient)
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
    """Randomly place ions inside the periodic box on CPU.

    Overlaps with colloids and existing ions are avoided using a simple
    rejection scheme. This function is called only once at startup.
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

# Transfer coordinates to GPU
q = torch.tensor(q_np, device=device, dtype=torch.float32)
q = wrap_positions(q)


# ================== Fix Colloid Positions ==================
def apply_colloid_constraints(q: torch.Tensor) -> torch.Tensor:
    """Fix colloids to their predefined positions inside the periodic box."""
    q[idx_col1] = torch.tensor([-R_col / 2.0, 0.0], device=device)
    q[idx_col2] = torch.tensor([+R_col / 2.0, 0.0], device=device)
    return q


q = apply_colloid_constraints(q)


# ================== Coulomb Energy (MC & HMC) ==================
def total_coulomb_energy(q: torch.Tensor) -> torch.Tensor:
    """Compute the total softened Coulomb energy for all particle pairs."""
    U = 0.0
    for i in range(N):
        qi = q[i]
        for j in range(i + 1, N):
            rij = minimum_image(qi - q[j])
            r2 = torch.dot(rij, rij) + 1e-12
            r_soft = torch.sqrt(r2 + soft_eps**2)
            U = U + k_c * charge[i] * charge[j] / r_soft
    return U


def coulomb_energy_of_particle(q: torch.Tensor, i: int) -> torch.Tensor:
    """Compute Coulomb energy contribution of particle i with all others."""
    qi = q[i].view(1, 2)
    rij = minimum_image(q - qi)
    r2 = torch.sum(rij * rij, dim=1) + 1e-12
    r_soft = torch.sqrt(r2 + soft_eps**2)

    qi_charge = charge[i]
    pair_E = k_c * qi_charge * charge / r_soft

    mask = torch.ones(N, dtype=torch.bool, device=device)
    mask[i] = False

    return torch.sum(pair_E[mask])


# ================== Hard-Sphere Overlap Check (MC only) ==================
def has_overlap_for_particle(q: torch.Tensor, i: int, q_proposed_i: torch.Tensor) -> bool:
    """Return True if the proposed position of particle i overlaps any other."""
    qi = q_proposed_i.view(1, 2)
    rij = minimum_image(q - qi)
    r2 = torch.sum(rij * rij, dim=1)
    r = torch.sqrt(r2 + 1e-12)

    rmin_vec = radii + radii[i]
    mask = torch.ones(N, dtype=torch.bool, device=device)
    mask[i] = False

    return torch.any(r[mask] < rmin_vec[mask]).item()


# ================== MC Stage: Temperature-Cycled Single-Particle MC ==================
frames_mc = []
energies_mc = []
accept_count_low = 0
accept_count_high = 0

# Initial total energy (Coulomb only, plus hard-sphere rejection)
U_total = total_coulomb_energy(q).item()
print("Initial total Coulomb energy (MC stage):", U_total)

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
            q_new_i = ((q_new_i + L_box / 2.0) % L_box) - L_box / 2.0

            accepted = False

            # Hard-sphere check: reject if overlapping
            if not has_overlap_for_particle(q, i, q_new_i):
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
            q_new_i = ((q_new_i + L_box / 2.0) % L_box) - L_box / 2.0

            accepted = False

            if not has_overlap_for_particle(q, i, q_new_i):
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
                frames_mc.append(q.detach().cpu().numpy())
                energies_mc.append(U_total)

print("MC stage finished.")
print("Final total Coulomb energy (MC):", energies_mc[-1] if energies_mc else None)
print("Final per-ion sigmas (MC):", step_sigma_vec)

# Overall acceptance statistics for MC
moves_high = n_cycles * n_sweeps_high * n_ions
moves_low  = n_cycles * n_sweeps_low  * n_ions

print("High-T acceptance rate (MC):", accept_count_high / moves_high)
print("Low-T acceptance rate (MC):",  accept_count_low  / moves_low)


# ================== HMC Stage: Coulomb + Lennard-Jones Repulsion ==================
"""In the HMC stage, we use the final configuration from the MC stage
as the starting point. All ions are updated simultaneously using a
standard HMC scheme (leapfrog + Metropolis). Colloids remain fixed.

Total potential energy used in HMC:

  U(q) = U_coulomb(q) + U_LJ(q)

where the Lennard-Jones potential between particles i,j is:

  U_LJ(r_ij) = 4 eps_LJ [ (sigma_ij / r_ij)^12 - (sigma_ij / r_ij)^6 ],

with sigma_ij chosen as radii[i] + radii[j],
so that the LJ core roughly coincides with the hard-sphere contact.
"""

# Masses for HMC: finite mass for ions, very large for colloids (effectively fixed)
m_ion = 1.0
m_col = 1e6
m_vec = torch.ones(N, device=device)
m_vec[:n_ions] = m_ion
m_vec[idx_col1:] = m_col


def total_energy(q: torch.Tensor) -> torch.Tensor:
    """Total potential energy = softened Coulomb + Lennard-Jones (12-6)."""
    U = 0.0
    for i in range(N):
        qi = q[i]
        for j in range(i + 1, N):
            rij = minimum_image(qi - q[j])
            r2 = torch.dot(rij, rij) + 1e-12
            r = torch.sqrt(r2)

            # Coulomb part
            r_soft = torch.sqrt(r2 + soft_eps**2)
            U = U + k_c * charge[i] * charge[j] / r_soft

            # Lennard-Jones part
            sigma_ij = radii[i] + radii[j]
            inv_r2 = (sigma_ij * sigma_ij) / r2        # (sigma/r)^2
            sr6 = inv_r2**3                           # (sigma/r)^6
            sr12 = sr6 * sr6                          # (sigma/r)^12
            U = U + 4.0 * eps_LJ * (sr12 - sr6)

    return U


def grad_total_energy(q: torch.Tensor) -> torch.Tensor:
    """Gradient of total energy U(q) = U_coulomb + U_LJ.

    For each pair (i, j):

      U_coul = k_c * q_i q_j / sqrt(r^2 + soft_eps^2)
        => dU_coul/dr as in softened Coulomb

      U_LJ   = 4 eps_LJ [ (sigma/r)^12 - (sigma/r)^6 ]
        => dU_LJ/dr = 24 eps_LJ / r * ( (sigma/r)^6 - 2 (sigma/r)^12 )

    The gradient contribution on particle i is:
      grad_i U = dU/dr * (r_ij / r)
    and on particle j is the opposite.
    """
    grad = torch.zeros_like(q)

    for i in range(N):
        for j in range(i + 1, N):
            rij = minimum_image(q[i] - q[j])
            r2 = torch.dot(rij, rij) + 1e-12
            r = torch.sqrt(r2)

            # ---------- Coulomb part ----------
            r_soft2 = r2 + soft_eps**2
            r_soft = torch.sqrt(r_soft2)
            dU_dr_c = -k_c * charge[i] * charge[j] / r_soft2 * (r / r_soft)
            force_ij_coul = dU_dr_c * (rij / r)

            # ---------- Lennard-Jones part ----------
            sigma_ij = radii[i] + radii[j]
            inv_r2 = (sigma_ij * sigma_ij) / r2   # (sigma/r)^2
            sr6 = inv_r2**3                       # (sigma/r)^6
            sr12 = sr6 * sr6                      # (sigma/r)^12

            # dU_LJ/dr = 24 eps_LJ / r * ( (sigma/r)^6 - 2 (sigma/r)^12 )
            dU_dr_LJ = (24.0 * eps_LJ / r) * (sr6 - 2.0 * sr12)
            force_ij_LJ = dU_dr_LJ * (rij / r)

            # Total pairwise contribution
            force_ij = force_ij_coul + force_ij_LJ

            grad[i] += force_ij
            grad[j] -= force_ij

    # Colloids are fixed: zero out their gradients to avoid drift
    grad[idx_col1] = 0.0
    grad[idx_col2] = 0.0

    return grad


def kinetic_energy(p: torch.Tensor) -> torch.Tensor:
    """Kinetic energy: sum_i p_i^2 / (2 m_i)."""
    return 0.5 * torch.sum(torch.sum(p * p, dim=1) / m_vec)


def leapfrog_step(q: torch.Tensor, p: torch.Tensor, eps: float, L: int):
    """Perform L leapfrog steps for the HMC trajectory."""
    q = q.clone()
    p = p.clone()

    # Initial half-step in p
    gradU = grad_total_energy(q)
    p = p - 0.5 * eps * gradU

    for step in range(L):
        # Full step in q
        q = q + eps * p / m_vec.view(-1, 1)
        q = wrap_positions(q)
        q = apply_colloid_constraints(q)

        # Full step in p (except after the last step)
        if step != L - 1:
            gradU = grad_total_energy(q)
            p = p - eps * gradU

    # Final half-step in p
    gradU = grad_total_energy(q)
    p = p - 0.5 * eps * gradU

    # Momentum flip for reversibility
    p = -p

    # Enforce colloid constraints on momenta
    p[idx_col1] = 0.0
    p[idx_col2] = 0.0

    return q, p


# ---------- HMC Parameters ----------
n_hmc_trajectories = 400
hmc_eps = 0.01
hmc_L = 20

frames_hmc = []
energies_hmc = []

# Use the final MC configuration as the starting point for HMC
q_hmc = q.clone()

for t in range(n_hmc_trajectories):
    # Sample initial momenta from Gaussian: p ~ N(0, sqrt(m / beta_low))
    std = torch.sqrt(m_vec / beta_low)
    p0 = torch.randn_like(q_hmc) * std.view(-1, 1)
    # Fix colloids: zero their momenta
    p0[idx_col1] = 0.0
    p0[idx_col2] = 0.0

    # Compute initial Hamiltonian
    U_old = total_energy(q_hmc)
    K_old = kinetic_energy(p0)
    H_old = U_old + K_old

    # Propose new state via leapfrog
    q_prop, p_prop = leapfrog_step(q_hmc, p0, hmc_eps, hmc_L)

    U_new = total_energy(q_prop)
    K_new = kinetic_energy(p_prop)
    H_new = U_new + K_new

    dH = (H_new - H_old).item()
    acc_prob = 1.0 if dH <= 0 else math.exp(-dH)

    if torch.rand(1, device=device).item() < acc_prob:
        q_hmc = q_prop
        U_use = U_new
    else:
        U_use = U_old

    frames_hmc.append(q_hmc.detach().cpu().numpy())
    energies_hmc.append(U_use.item())

print("HMC stage finished.")
print("Final total potential energy (HMC):", energies_hmc[-1] if energies_hmc else None)


# ================== Animation (HMC Stage) ==================
frames = np.array(frames_hmc)
n_frames = frames.shape[0]

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-L_box / 2, L_box / 2)
ax.set_ylim(-L_box / 2, L_box / 2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("HMC sampling (Coulomb + Lennard-Jones)")

ion_scat = ax.scatter([], [], s=20, c='tab:blue', label='ions')
col_scat = ax.scatter([], [], s=300, c='tab:red', label='colloids')
ax.legend(loc='upper right')


def init():
    ion_scat.set_offsets(np.empty((0, 2)))
    col_scat.set_offsets(np.empty((0, 2)))
    return ion_scat, col_scat


def update(frame_idx):
    pos = frames[frame_idx]
    ions = pos[:n_ions]
    cols = pos[n_ions:]
    ion_scat.set_offsets(ions)
    col_scat.set_offsets(cols)
    return ion_scat, col_scat


ani = animation.FuncAnimation(
    fig,
    update,
    frames=n_frames,
    init_func=init,
    interval=30,
    blit=True,
)

plt.tight_layout()
plt.show()
