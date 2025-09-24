import numpy as np
import matplotlib.pyplot as plt

#Data of the Problem
EA = 50000000 #[N]
z  = 25 #[mm]
l  = 2500 #[mm]
Ks = 2.35 #[N/mm]

dUe = 70 #[N] 
dWe = -7 #[N]
STOP_We = -259 #[N]

TOL = 1e-4
MAX_IT = 100


def Strain(u, w):
    return -u/l + (z/l)*(w/l) + 0.5*(w/l)**2

def N_Axial(u, w):
    return EA * Strain(u, w)

def Unbalanced_Vector(u, w, Ue, We):
    N = N_Axial(u, w) 
    q1 = - Ue - N
    q2 = - We + N*((z + w)/l) + Ks*w
    return np.array([q1, q2], dtype=float)

def Tangent_Stiffness(u, w):
    K11 = EA*(1.0/l)
    K12 = -EA*((z + w)/l**2) 
    K21 = -((z + w)/l ) * K11                  
    K22 = -((z + w)/l ) * K12 + (1.0/l)*N_Axial(u,w) + Ks

    return np.array([[K11, K12],
                     [K21, K22]], dtype=float)

def Incremental_Iterative_Solution(u0=0, w0=0, Ue0=0, We0=0, max_steps=1000, show_iter=True):

    u, w = float(u0), float(w0)
    Ue, We = float(Ue0), float(We0)
  
    hist = {"step": [], "u": [], "w": [], "N": [],"Ue": [], "We": [], "-w": [], "-We": []}
    step = 0

    if show_iter:
        print(f"Step: {step:2d}: Ue={Ue:5.0f}, We={We:5.0f}, u={u:10.11f}, w={w:10.10f}, N={N_Axial(u,w):5.2f}")

    for step in range(1, max_steps+1):
        #Load Increment
        dq_ext = np.array([dUe, dWe], dtype=float)
        Ue += dUe
        We += dWe

        #Predictor: ΔP = (Δu, Δw) = K^{-1} * (ΔUe,ΔWe) 
        K_prev = Tangent_Stiffness(u, w)
        try:
            dP = np.linalg.solve(K_prev, dq_ext)
        except np.linalg.LinAlgError:
            dP = np.zeros(2)

        #Predictor Increment
        u_it = u + dP[0]
        w_it = w + dP[1]

        #Newton-Raphson:
        converged = False
        for it in range(1, MAX_IT+1):
        
            q = Unbalanced_Vector(u_it, w_it, Ue, We)
            if np.linalg.norm(q, ord=2) <= TOL:
                converged = True
                break

            Kt = Tangent_Stiffness(u_it, w_it)
            try:
                dq = -np.linalg.solve(Kt, q)
            except np.linalg.LinAlgError:
                dq = -np.linalg.solve(Kt + 1e-12*np.eye(2), q)

            u_it += dq[0]
            w_it += dq[1]

        u, w = u_it, w_it
        N = N_Axial(u, w)

        if show_iter:
            print(f"Step: {step:2d}: Ue={Ue:5.0f}, We={We:5.0f}, u={u:10.10f}, w={w:10.10f}, N={N_Axial(u,w):5.2f}, Iterations={it:2d}")

        hist["step"].append(step)
        hist["u"].append(u)  
        hist["w"].append(w)  
        hist["N"].append(N)
        hist["Ue"].append(Ue)
        hist["We"].append(We)
        hist["-w"].append(-w)
        hist["-We"].append(-We)

        if We <= STOP_We:
            break

    return hist

# Plots
def plot_results(history):
    fig, ax = plt.subplots(1, 2, figsize=(12,4.5), dpi=150)

    # Ue vs u
    ax[0].set_title("Load – Deflection Curve: Ue vs u", weight='bold', fontsize=12)  
    ax[0].plot(history["u"], history["Ue"], marker='s', lw=1, markerfacecolor='gray',  markeredgecolor='blue')
    ax[0].set_xlabel("u [mm]"); ax[1].set_ylabel("Ue [N]")
    ax[0].grid(True, ls='--')

    # (-We) vs (-w)
    ax[1].set_title("Load – Deflection Curve: −We vs −w", weight='bold', fontsize=12)  
    ax[1].plot(history["-w"], history["-We"], marker='o', lw=1, markerfacecolor='gray',  markeredgecolor='green')
    ax[1].set_xlabel("-w [mm]"); ax[0].set_ylabel("-We [N]")
    ax[1].grid(True, ls='--')

    fig.tight_layout()
    plt.show()

# Play
history = Incremental_Iterative_Solution(show_iter=True)
plot_results(history)
