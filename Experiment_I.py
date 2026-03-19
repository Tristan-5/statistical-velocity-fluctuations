import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Parameters
nu_c = 5.0          # Collision rate (s^-1)
sigma_v = 1.0       # Target std dev of velocity increments (m/s)
T = 10.0            # Total time (s)
n_snapshots = 150   # Number of temporal snapshots
M = 1000000         # Number of trajectories (10^6)

t = np.linspace(0.0, T, n_snapshots)
a = np.sqrt(3) * sigma_v
var_empirical = np.zeros(n_snapshots)

for i in range(1, n_snapshots):
    current_t = t[i]
    counts = np.random.poisson(nu_c * current_t, size=M)
    total_kicks = counts.sum()
    
    if total_kicks > 0:
        kicks = np.random.uniform(-a, a, size=total_kicks)
        trajectory_ids = np.repeat(np.arange(M), counts)
        V_independent = np.bincount(trajectory_ids, weights=kicks, minlength=M)
        var_empirical[i] = np.var(V_independent)

var_theory_line = nu_c * (sigma_v**2) * t
theory_slope = nu_c * (sigma_v**2)

slope, intercept, r_val, p_val, std_err = stats.linregress(t[1:], var_empirical[1:])

rel_error = abs(slope - theory_slope) / theory_slope

ci_lower, ci_upper = slope - 1.96 * std_err, slope + 1.96 * std_err

print(f"Theoretical slope: {theory_slope:.4f}")
print(f"Fitted Empirical slope: {slope:.4f}")
print(f"Relative Error: {rel_error:.2e}")
print(f"95% CI for slope: [{ci_lower:.4f}, {ci_upper:.4f}]")

plt.figure(figsize=(10, 6))
plt.plot(t, var_empirical, 'b-', label="Monte Carlo (Independent Ensembles)", alpha=0.8)
plt.plot(t, var_theory_line, 'r--', label="Theory (ν_c σ_v² t)", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Velocity Variance (m²/s²)")
plt.title("Experiment I: Variance Growth Validation")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("Experiment_I.png", dpi=600, bbox_inches="tight")

plt.show()
