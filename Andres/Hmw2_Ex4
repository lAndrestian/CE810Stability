import numpy as np
import matplotlib.pyplot as plt

EA = 50000000 # [N]
z = 25 # [mm]
L = 2500 # [mm]
Ks = 1.35 # [N/mm]
DeltaW = -7 # [N]
STOP_FORCE = -91 # [N]

def compute_kt(w, N):
    return ((EA / L) * ((z + w) / L)**2) + (N / L) + Ks

def compute_deflection(w_prev, kt_prev):
    return w_prev + DeltaW / kt_prev

def compute_normal_force(w):
    return EA * ((z / L) * (w / L) + 0.5 * (w / L)**2)

def compute_real_force(w, N):
    #return (EA / L**3) * (z**2 * w + 1.5 * z * w**2 + 0.5 * w**3) + Ks * w
    return N * ((z + w) / L) + Ks * w

def compute_calculated_force(Wreal_prev):
    return Wreal_prev + DeltaW

def relative_error(W, Wreal):
    return abs(W - Wreal) / abs(Wreal) if Wreal != 0 else 0.0

def analytical_solution(w):
    return (((EA / L**3) * (z**2 * w + 1.5 * z * w**2 + 0.5 * w**3)) + Ks * w)

def incremental_solution(max_iterations=1000, tol=1e-4, max_newton=25):
    w, N, W, Wreal = 0.0, 0.0, 0.0, 0.0
    kt = compute_kt(w, N)
    err = relative_error(W, Wreal)

    history = {"kt": [kt], "w": [w], "-w": [-w], "N": [N], "W": [W], "-W": [-W], "Wreal": [Wreal], "error": [err]}
    print(f"Step 0: kt={kt:.3f}, w={w:.6f}, -w={-w:.6f}, N={N:.6f}, W={W:.6f}, -W={-W:.6f}, Wreal={Wreal:.6f}, Error={err:.3%}")

    for step in range(1, max_iterations + 1):
        W = compute_calculated_force(W)
        w_it = compute_deflection(history["w"][-1], history["kt"][-1])
        converged = False
        for it in range(1, max_newton + 1):
            N_it = compute_normal_force(w_it)
            kt_it = compute_kt(w_it, N_it)
            Wreal_it = compute_real_force(w_it, N_it)  
            g = W - Wreal_it  

            print(f"Step {step}: Num_it={it:02d}, |g|={g:.3e}, kt={kt_it:.3f}, w={w_it:.6f}, -w={-w_it:.6f}, N={N_it:.6f}, W={W:.6f}, -W={-W:.6f}, Wreal={Wreal_it:.6f}, Error={err:.5%}")
            if abs(g) <= tol:
                converged = True
                break

            delta_w = g / kt_it
            w_it += delta_w

            

        w = w_it
        N = compute_normal_force(w)
        kt = compute_kt(w, N)
        Wreal = compute_real_force(w, N)
        err = relative_error(W, Wreal)

        history["kt"].append(kt)
        history["w"].append(w);   history["-w"].append(-w)
        history["N"].append(N)
        history["W"].append(W);   history["-W"].append(-W)
        history["Wreal"].append(Wreal)
        history["error"].append(err)

        

        #if converged:
        #    print(f"Step {step}: Num_it={it:02d}, |g|={abs(W-Wreal):.3e}, kt={kt:.3f}, w={w:.6f}, -w={-w:.6f}, N={N:.6f}, W={W:.6f}, -W={-W:.6f}, Wreal={Wreal:.6f}, Error={err:.5%}")
        #else:
        #    print(f"Step {step}: NO CONVERGE (max it={max_newton}), |g|={abs(W-Wreal):.3e}, kt={kt:.3f}, w={w:.6f}, -w={-w:.6f}, N={N:.6f}, W={W:.6f}, -W={-W:.6f}, Wreal={Wreal:.6f}, Error={err:.3%}")

        if W <= STOP_FORCE:
            print(f"\nStopping at step {step} because W = {W:.3f} <= {STOP_FORCE}")
            break

    return history


def plot_results(history):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    ax.plot(
        history["-w"], history["-W"],
        marker="o", markersize=5, linewidth=2,
        color='tab:blue', label="Incremental Solution"
    )

    w_vals = np.linspace(min(history["w"]), max(history["w"]), 1000)
    W_exact = analytical_solution(w_vals)
    ax.plot(
        -w_vals, -W_exact,
        linestyle='-', linewidth=2.0,
        color='tab:red', label="Analytical Solution"
    )

    ax.set_xlabel("Deflection  $-w$  [mm]", fontsize=13)
    ax.set_ylabel("Force  $-W$  [N]", fontsize=13)
    ax.minorticks_on()
    ax.grid(True, which='both', linestyle=':', linewidth=0.8)
    ax.legend(frameon=False, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    plt.show()

history = incremental_solution()
plot_results(history)
