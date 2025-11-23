# GPU-accelerated MC + HMC Simulation of Charged Colloids and Ions

This project implements a high‑performance **GPU‑accelerated sampling framework** for a 2D charged colloid–ion system. The workflow combines:

* **Temperature-cycled Metropolis Monte Carlo (MC)** for pre‑equilibration
* **Full‑vector Hamiltonian Monte Carlo (HMC)** for efficient global sampling
* **Lennard–Jones soft-sphere repulsion** (HMC)
* **Softened Coulomb interaction** (MC & HMC)
* **Per‑particle adaptive step‑sizes** (MC)
* **GPU‑accelerated energy and gradient evaluation** (PyTorch CUDA)

Colloids are fixed in space; ions move under hard‑sphere constraints (MC) and smooth LJ repulsion (HMC). The simulation supports periodic boundaries and produces an animated trajectory of the HMC stage.

---

## Features

### ✅ Monte Carlo (MC) Stage

* Sequential single-particle Metropolis updates
* Hard-sphere collision rejection
* Softened Coulomb interaction
* Temperature cycling: **high-T exploration** → **low-T sampling**
* Per-particle **adaptive sigma** maintaining target acceptance (0.6–0.8)
* GPU evaluation of all distances and energies

### ✅ Hamiltonian Monte Carlo (HMC) Stage

* All ions move simultaneously
* Leapfrog integrator + Metropolis acceptance
* Potential energy:

  * **Softened Coulomb**
  * **LJ 12-6 soft-sphere repulsion**
* Colloids remain fixed by constraints
* Produces trajectory animation of HMC evolution

---



## License

Apache-2.0

---

## Author

Tovi Yuen
