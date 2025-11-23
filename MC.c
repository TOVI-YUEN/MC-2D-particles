/*
 * mc_cpu_mc_only.c
 *
 * CPU-based Metropolis Monte Carlo with temperature cycling
 * for a 2D charged colloid–ion system.
 *
 * - Two fixed colloids (opposite charges)
 * - Multiple mobile ions (hard-sphere + softened Coulomb)
 * - Sequential single-particle Metropolis MC
 * - Per-particle adaptive step sizes (sigma_i)
 * - High-T / low-T cycling (simulated annealing + sampling)
 * - OpenMP used to parallelize per-particle energy evaluation
 *
 * Output:
 *   mc_final.bin  (binary float32, shape [N, 2], wrapped to PBC)
 *
 * This file can be read by a Python + GPU HMC script as initial configuration.
 *
 * Compile (GCC example):
 *   gcc mc_cpu_mc_only.c -O3 -march=native -ffast-math -fopenmp -o mc_cpu_mc_only
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* ================== System Parameters ================== */

static const int   N_IONS = 800;
static const int   N_COLS = 2;
static const float L_BOX  = 120.0f;
static const float R_COL  = 15.0f;

static const float R_ION  = 1.0f;
static const float R_COLR = 10.0f;

static const float Q_ION  = 1.0f;
static const float Q_COL  = 60.0f;

static const float K_C    = 20.0f;
static const float SOFT_EPS = 1.0f;

/* MC parameters (temperature cycling) */
static const double BETA_LOW  = 1.0;   /* low-T phase */
static const double BETA_HIGH = 0.1;   /* high-T phase */

static const int N_CYCLES      = 200;
static const int N_SWEEPS_HIGH = 50;
static const int N_SWEEPS_LOW  = 100;

/* Adaptive sigma parameters */
static const int    ADAPT_WINDOW = 100;
static const double TARGET_LOW   = 0.6;
static const double TARGET_HIGH  = 0.8;
static const double SIGMA_MIN    = 0.1;
static const double SIGMA_MAX    = 10.0;
static const double ADAPT_SHRINK = 0.8;
static const double ADAPT_GROW   = 1.2;

/* Maximum total particles */
#define N_MAX 2000

/* ================== Utility: Random ================== */

static double urand01(void) {
    return (double)rand() / (double)RAND_MAX;
}

/* ================== Periodic Boundary Conditions ================== */

static inline float min_image(float x) {
    return x - roundf(x / L_BOX) * L_BOX;
}

static void wrap_position(float q[2]) {
    for (int d = 0; d < 2; ++d) {
        float x = q[d] + 0.5f * L_BOX;
        x = fmodf(x, L_BOX);
        if (x < 0.0f) x += L_BOX;
        q[d] = x - 0.5f * L_BOX;
    }
}

/* ================== Random Placement (CPU) ================== */

static void random_place_ions(
    int n_ions,
    float q[][2],
    float radii[],
    int idx_col1,
    int idx_col2,
    float safe_factor,
    int max_attempts
) {
    float col1[2] = {-R_COL * 0.5f, 0.0f};
    float col2[2] = {+R_COL * 0.5f, 0.0f};

    int placed = 0;
    int attempts = 0;

    while (placed < n_ions && attempts < max_attempts) {
        attempts++;

        float cand[2];
        cand[0] = (float)(urand01() - 0.5) * L_BOX;
        cand[1] = (float)(urand01() - 0.5) * L_BOX;

        int ok = 1;

        /* check vs colloids */
        {
            float rij[2];
            float r, rmin;

            rij[0] = cand[0] - col1[0];
            rij[1] = cand[1] - col1[1];
            rij[0] = min_image(rij[0]);
            rij[1] = min_image(rij[1]);
            r = sqrtf(rij[0]*rij[0] + rij[1]*rij[1]);
            rmin = safe_factor * (R_ION + R_COLR);
            if (r < rmin) ok = 0;

            rij[0] = cand[0] - col2[0];
            rij[1] = cand[1] - col2[1];
            rij[0] = min_image(rij[0]);
            rij[1] = min_image(rij[1]);
            r = sqrtf(rij[0]*rij[0] + rij[1]*rij[1]);
            rmin = safe_factor * (R_ION + R_COLR);
            if (r < rmin) ok = 0;
        }

        /* check vs placed ions */
        for (int k = 0; k < placed && ok; ++k) {
            float rij[2];
            rij[0] = cand[0] - q[k][0];
            rij[1] = cand[1] - q[k][1];
            rij[0] = min_image(rij[0]);
            rij[1] = min_image(rij[1]);
            float r = sqrtf(rij[0]*rij[0] + rij[1]*rij[1]);
            float rmin = safe_factor * (2.0f * R_ION);
            if (r < rmin) ok = 0;
        }

        if (ok) {
            q[placed][0] = cand[0];
            q[placed][1] = cand[1];
            placed++;
        }
    }

    if (placed < n_ions) {
        fprintf(stderr, "Error: Failed to place ions (placed=%d)\n", placed);
        exit(EXIT_FAILURE);
    }

    /* Set colloid positions */
    q[idx_col1][0] = col1[0];
    q[idx_col1][1] = col1[1];
    q[idx_col2][0] = col2[0];
    q[idx_col2][1] = col2[1];
}

/* ================== Coulomb Energy ================== */

/* Total Coulomb energy (for initialization / diagnostics) */
static double total_coulomb_energy(
    int N,
    float q[][2],
    const float charge[],
    float soft_eps
) {
    double U = 0.0;
    for (int i = 0; i < N; ++i) {
        float qi0 = q[i][0];
        float qi1 = q[i][1];
        for (int j = i + 1; j < N; ++j) {
            float rij0 = qi0 - q[j][0];
            float rij1 = qi1 - q[j][1];
            rij0 = min_image(rij0);
            rij1 = min_image(rij1);
            float r2 = rij0*rij0 + rij1*rij1 + 1e-12f;
            float r_soft = sqrtf(r2 + soft_eps * soft_eps);
            U += (double)K_C * (double)charge[i] * (double)charge[j] / (double)r_soft;
        }
    }
    return U;
}

/*
 * Coulomb energy contribution of particle i:
 *
 *   E_i = sum_{j != i} K_C * q_i q_j / sqrt(r_ij^2 + eps^2)
 *
 * This is used for local energy difference upon moving ion i.
 * We parallelize over j with OpenMP.
 */
static double coulomb_energy_of_particle(
    int N,
    float q[][2],
    const float charge[],
    int i,
    float soft_eps
) {
    double Ei = 0.0;
    float qi0 = q[i][0];
    float qi1 = q[i][1];

    #pragma omp parallel for reduction(+:Ei)
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;

        float rij0 = qi0 - q[j][0];
        float rij1 = qi1 - q[j][1];
        rij0 = min_image(rij0);
        rij1 = min_image(rij1);
        float r2 = rij0*rij0 + rij1*rij1 + 1e-12f;
        float r_soft = sqrtf(r2 + soft_eps * soft_eps);

        double pairE = (double)K_C * (double)charge[i] * (double)charge[j] / (double)r_soft;
        Ei += pairE;
    }

    return Ei;
}

/* ================== Hard-Sphere Overlap Check ================== */

static int has_overlap_for_particle(
    int N,
    float q[][2],
    const float radii[],
    int i,
    const float q_new_i[2]
) {
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;

        float rij0 = q_new_i[0] - q[j][0];
        float rij1 = q_new_i[1] - q[j][1];
        rij0 = min_image(rij0);
        rij1 = min_image(rij1);
        float r2 = rij0*rij0 + rij1*rij1;
        float r = sqrtf(r2 + 1e-12f);

        float rmin = radii[i] + radii[j];
        if (r < rmin) {
            return 1; /* overlap */
        }
    }
    return 0;
}

/* ================== Main: MC with Temperature Cycling ================== */

int main(void) {
    srand((unsigned int)time(NULL));

    const int N = N_IONS + N_COLS;
    if (N > N_MAX) {
        fprintf(stderr, "Error: N exceeds N_MAX\n");
        return EXIT_FAILURE;
    }

    /* Coordinates, radii, charges */
    float q[N_MAX][2];
    float radii[N_MAX];
    float charge[N_MAX];

    /* Initialize radii */
    for (int i = 0; i < N; ++i) radii[i] = 0.0f;
    for (int i = 0; i < N_IONS; ++i) radii[i] = R_ION;
    for (int i = N_IONS; i < N; ++i) radii[i] = R_COLR;

    /* Initialize charges */
    for (int i = 0; i < N; ++i) charge[i] = 0.0f;
    int half = N_IONS / 2;
    for (int i = 0; i < half; ++i)      charge[i]          = +Q_ION;
    for (int i = half; i < N_IONS; ++i) charge[i]          = -Q_ION;
    charge[N_IONS]     = +Q_COL;  /* colloid 1 */
    charge[N_IONS + 1] = -Q_COL;  /* colloid 2 */

    int idx_col1 = N_IONS;
    int idx_col2 = N_IONS + 1;

    /* Random initial positions for ions, fixed colloids */
    random_place_ions(N_IONS, q, radii, idx_col1, idx_col2, 1.4f, 200000);

    /* Wrap all positions into box (just to be safe) */
    for (int i = 0; i < N; ++i) {
        wrap_position(q[i]);
    }

    /* Per-ion adaptive step sizes and windows */
    double sigma[N_IONS];
    int    window_accept[N_IONS];
    int    window_total[N_IONS];

    for (int i = 0; i < N_IONS; ++i) {
        sigma[i] = 0.5;
        window_accept[i] = 0;
        window_total[i]  = 0;
    }

    /* Initial total Coulomb energy */
    double U_total = total_coulomb_energy(N, q, charge, SOFT_EPS);
    printf("Initial total Coulomb energy (MC stage): %.6f\n", U_total);

    long long accept_count_high = 0;
    long long accept_count_low  = 0;

    /* Main temperature-cycled MC loop */
    for (int cycle = 0; cycle < N_CYCLES; ++cycle) {

        /* ---------- High Temperature Phase ---------- */
        double beta = BETA_HIGH;
        for (int sweep = 0; sweep < N_SWEEPS_HIGH; ++sweep) {
            for (int i = 0; i < N_IONS; ++i) {

                double sigma_i = sigma[i];

                /* Old energy contribution (Coulomb only) */
                double E_old_i = coulomb_energy_of_particle(N, q, charge, i, SOFT_EPS);

                /* Propose new position */
                float q_old_i[2] = { q[i][0], q[i][1] };
                float q_new_i[2];
                q_new_i[0] = q_old_i[0] + (float)(sigma_i * ((urand01() * 2.0) - 1.0));
                q_new_i[1] = q_old_i[1] + (float)(sigma_i * ((urand01() * 2.0) - 1.0));

                wrap_position(q_new_i);

                int accepted = 0;

                if (!has_overlap_for_particle(N, q, radii, i, q_new_i)) {
                    /* Temporarily update position */
                    q[i][0] = q_new_i[0];
                    q[i][1] = q_new_i[1];

                    /* Keep colloids fixed */
                    q[idx_col1][0] = -R_COL * 0.5f;
                    q[idx_col1][1] = 0.0f;
                    q[idx_col2][0] = +R_COL * 0.5f;
                    q[idx_col2][1] = 0.0f;

                    /* New local energy */
                    double E_new_i = coulomb_energy_of_particle(N, q, charge, i, SOFT_EPS);

                    /* True ΔU for moving i: 0.5 * (E_new - E_old), because total U = 0.5 * Σ_i E_i */
                    double dU = 0.5 * (E_new_i - E_old_i);

                    double acc_prob = 1.0;
                    if (dU > 0.0) {
                        acc_prob = exp(-beta * dU);
                    }

                    if (urand01() < acc_prob) {
                        U_total += dU;
                        accept_count_high++;
                        accepted = 1;
                    } else {
                        /* revert */
                        q[i][0] = q_old_i[0];
                        q[i][1] = q_old_i[1];
                        q[idx_col1][0] = -R_COL * 0.5f;
                        q[idx_col1][1] = 0.0f;
                        q[idx_col2][0] = +R_COL * 0.5f;
                        q[idx_col2][1] = 0.0f;
                    }
                }

                /* Adaptive statistics */
                window_total[i] += 1;
                if (accepted) window_accept[i] += 1;

                if (window_total[i] >= ADAPT_WINDOW) {
                    double acc_rate_i = (double)window_accept[i] / (double)window_total[i];
                    if (acc_rate_i < TARGET_LOW) {
                        sigma[i] *= ADAPT_SHRINK;
                        if (sigma[i] < SIGMA_MIN) sigma[i] = SIGMA_MIN;
                    } else if (acc_rate_i > TARGET_HIGH) {
                        sigma[i] *= ADAPT_GROW;
                        if (sigma[i] > SIGMA_MAX) sigma[i] = SIGMA_MAX;
                    }
          //          printf("[HighT adapt ion %d] cycle=%d, acc_rate=%.3f, sigma=%.3f\n",
           //                i, cycle, acc_rate_i, sigma[i]);
                    window_accept[i] = 0;
                    window_total[i]  = 0;
                }
            }
        }

        /* ---------- Low Temperature Phase ---------- */
        beta = BETA_LOW;
        for (int sweep = 0; sweep < N_SWEEPS_LOW; ++sweep) {
            for (int i = 0; i < N_IONS; ++i) {

                double sigma_i = sigma[i];

                double E_old_i = coulomb_energy_of_particle(N, q, charge, i, SOFT_EPS);

                float q_old_i[2] = { q[i][0], q[i][1] };
                float q_new_i[2];
                q_new_i[0] = q_old_i[0] + (float)(sigma_i * ((urand01() * 2.0) - 1.0));
                q_new_i[1] = q_old_i[1] + (float)(sigma_i * ((urand01() * 2.0) - 1.0));

                wrap_position(q_new_i);

                int accepted = 0;

                if (!has_overlap_for_particle(N, q, radii, i, q_new_i)) {
                    q[i][0] = q_new_i[0];
                    q[i][1] = q_new_i[1];

                    q[idx_col1][0] = -R_COL * 0.5f;
                    q[idx_col1][1] = 0.0f;
                    q[idx_col2][0] = +R_COL * 0.5f;
                    q[idx_col2][1] = 0.0f;

                    double E_new_i = coulomb_energy_of_particle(N, q, charge, i, SOFT_EPS);
                    double dU = 0.5 * (E_new_i - E_old_i);

                    double acc_prob = 1.0;
                    if (dU > 0.0) {
                        acc_prob = exp(-beta * dU);
                    }

                    if (urand01() < acc_prob) {
                        U_total += dU;
                        accept_count_low++;
                        accepted = 1;
                    } else {
                        q[i][0] = q_old_i[0];
                        q[i][1] = q_old_i[1];
                        q[idx_col1][0] = -R_COL * 0.5f;
                        q[idx_col1][1] = 0.0f;
                        q[idx_col2][0] = +R_COL * 0.5f;
                        q[idx_col2][1] = 0.0f;
                    }
                }

                window_total[i] += 1;
                if (accepted) window_accept[i] += 1;

                if (window_total[i] >= ADAPT_WINDOW) {
                    double acc_rate_i = (double)window_accept[i] / (double)window_total[i];
                    if (acc_rate_i < TARGET_LOW) {
                        sigma[i] *= ADAPT_SHRINK;
                        if (sigma[i] < SIGMA_MIN) sigma[i] = SIGMA_MIN;
                    } else if (acc_rate_i > TARGET_HIGH) {
                        sigma[i] *= ADAPT_GROW;
                        if (sigma[i] > SIGMA_MAX) sigma[i] = SIGMA_MAX;
                    }
            //        printf("[LowT adapt ion %d] cycle=%d, acc_rate=%.3f, sigma=%.3f\n",
             //              i, cycle, acc_rate_i, sigma[i]);
                    window_accept[i] = 0;
                    window_total[i]  = 0;
                }
            }
        }
    }

    printf("MC stage finished.\n");
    printf("Final total Coulomb energy (MC): %.6f\n", U_total);

    long long total_moves_high = (long long)N_CYCLES * N_SWEEPS_HIGH * N_IONS;
    long long total_moves_low  = (long long)N_CYCLES * N_SWEEPS_LOW  * N_IONS;

    printf("High-T acceptance rate (MC): %.6f\n",
           (double)accept_count_high / (double)total_moves_high);
    printf("Low-T  acceptance rate (MC): %.6f\n",
           (double)accept_count_low  / (double)total_moves_low);

    /* Wrap all positions again for safety */
    for (int i = 0; i < N; ++i) {
        wrap_position(q[i]);
    }

    /* Write final configuration to binary file (float32, shape [N, 2]) */
    const char *fname = "mc_final.bin";
    FILE *fp = fopen(fname, "wb");
    if (!fp) {
        perror("fopen mc_final.bin");
        return EXIT_FAILURE;
    }
    size_t written = fwrite(q, sizeof(float), (size_t)(N * 2), fp);
    fclose(fp);

    if (written != (size_t)(N * 2)) {
        fprintf(stderr, "Error: wrote %zu floats, expected %d\n",
                written, N * 2);
        return EXIT_FAILURE;
    }

    printf("Final configuration written to %s\n", fname);
    return EXIT_SUCCESS;
}
