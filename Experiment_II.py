import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42) 

# Parameters
nu_c = 5.0          # Collision rate (s^-1)
sigma_v = 1.0       # Std dev of velocity increments
M = 1000000         # 10^6 trajectories
B = 1000            # Bootstrap iterations
test_times = [0.1, 5.1, 10.1, 20.0]

a = np.sqrt(3) * sigma_v
results = {}

for t_val in test_times:
    N_total = rng.poisson(nu_c * t_val, size=M)
    total_kicks = N_total.sum()
    
    v_independent = np.zeros(M)
    if total_kicks > 0:
        kicks = rng.uniform(-a, a, size=total_kicks)
        traj_ids = np.repeat(np.arange(M), N_total)
        v_independent = np.bincount(traj_ids, weights=kicks, minlength=M)
    
    theoretical_std = np.sqrt(nu_c * sigma_v**2 * t_val)
    Z = v_independent / theoretical_std
    
    results[t_val] = {
        'Z': Z,
        'skew': stats.skew(v_independent),
        'kurt': stats.kurtosis(v_independent, fisher=False)
    }
    print(f"Completed t={t_val:4.1f}: Skew={results[t_val]['skew']:8.4f}, Kurt={results[t_val]['kurt']:8.4f}")

plt.figure(figsize=(14, 10))
x = np.linspace(-4, 4, 400)
normal_pdf = stats.norm.pdf(x)

for i, t_val in enumerate(test_times):
    data = results[t_val]
    plt.subplot(2, 2, i+1)
    plt.hist(data['Z'], bins=200, density=True, alpha=0.6, color='blue', label='Empirical')
    plt.plot(x, normal_pdf, 'r--', linewidth=2, label='Normal PDF')
    plt.title(f"t = {t_val}\nSkew={data['skew']:.4f}, Kurt={data['kurt']:.4f}")
    plt.xlim(-5, 5)
    plt.legend(prop={'size': 8})
    plt.grid(alpha=0.2)

plt.suptitle("Experiment II: Gaussian Convergence Validation", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig("Experiment_II.png", dpi=600, bbox_inches="tight")

plt.show()

Z_final = results[20.0]['Z']

bootstrap_skew = stats.bootstrap((Z_final,), 
                                 lambda x: stats.skew(x), 
                                 n_resamples=B, 
                                 confidence_level=0.95,
                                 method='percentile')

ci_skew_low, ci_skew_high = bootstrap_skew.confidence_interval

bootstrap_kurt = stats.bootstrap((Z_final,), 
                                 lambda x: stats.kurtosis(x, fisher=False), 
                                 n_resamples=B, 
                                 confidence_level=0.95,
                                 method='percentile')

ci_kurt_low, ci_kurt_high = bootstrap_kurt.confidence_interval

print(f"Empirical Skewness: {results[20.0]['skew']:.4f}")
print(f"95% Bootstrap CI:  [{ci_skew_low:.4f}, {ci_skew_high:.4f}]")
print(f"Target Skewness:    0.0000")
print(f"Empirical Kurtosis: {results[20.0]['kurt']:.4f}")
print(f"95% Bootstrap CI:  [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Target Kurtosis:    3.0000")
