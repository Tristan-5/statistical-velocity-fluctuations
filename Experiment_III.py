import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

nu_c = 5.0          
sigma_v = 1.0       
k = 2 * np.pi       

T = 20.0            
dt = 5e-4           
M = 1000000 
n_steps = int(T / dt)

nu_values = [0.1, 0.2, 0.5, 1.0]
var_empirical = []
var_theory = [(nu_c * sigma_v**2) / (2 * (nu * k**2)) for nu in nu_values]

print(f"{'Viscosity':>9} | {'Empirical Var':>13} | {'Theory Var':>10} | {'Rel Error':>10}")

start_time = time.time()

for i, nu in enumerate(nu_values):
    u = np.zeros(M)
    gamma = nu * k**2
    decay_constant = np.exp(-gamma * dt)
    lam = nu_c * dt
    a = np.sqrt(3) * sigma_v

    for _ in range(n_steps):
        u *= decay_constant
        n_collisions = np.random.poisson(lam, size=M)
        max_c = np.max(n_collisions)
        all_jumps = np.random.uniform(-a, a, size=(max_c, M))
        mask = np.arange(max_c)[:, None] < n_collisions
        u += np.sum(all_jumps * mask, axis=0)

    v_emp = np.var(u)
    var_empirical.append(v_emp)
    rel_err = abs(v_emp - var_theory[i]) / var_theory[i]
    print(f"{nu:9.1f} | {v_emp:13.6f} | {var_theory[i]:10.6f} | {rel_err:10.4f}")

total_duration = time.time() - start_time
slope = np.polyfit(np.log(nu_values), np.log(var_empirical), 1)[0]

print(f"Log-Log slope (Paper Target -1.0): {slope:.4f}")

plt.figure(figsize=(10, 6))
plt.loglog(nu_values, var_empirical, 's', markersize=8, label=f"Poisson Jump Ensemble (M={M})")
plt.loglog(nu_values, var_theory, 'r--', alpha=0.7, label="Theoretical Inverse-Viscosity Scaling")
plt.xlabel("Viscosity (ν)")
plt.ylabel("Stationary Variance ⟨u²⟩")
plt.title(f"Validation of Discrete Collisional Model\nSlope = {slope:.4f} (Theoretical -1.0)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("Experiment_III.png", dpi=600, bbox_inches="tight")

plt.show()
