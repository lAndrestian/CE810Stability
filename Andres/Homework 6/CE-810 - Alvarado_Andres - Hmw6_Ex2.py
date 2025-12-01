import numpy as np

DOF_PER_NODE = 6  # [ux, uy, uz, rx, ry, rz] per node

# 1. MATERIALS: 
MAT = np.array([
    [29000.0, 0.30], #[E (ksi), nu]
])

# 2. SECTION: 
SEC = np.array([
    [20.0, 6.67, 166.67, 26.67], #[A (in^2), Iy (in^4), Iz (in^4), J (in^4)]
])

# 3. NODES AND ELEMENTS WITH 1D MESH IN X DIRECTION:
def create_beam_mesh_1D(x0: float, L: float, num_elems: int, mat_id: int, sec_id: int):
    num_nodes = num_elems + 1
    x_coords = np.linspace(x0, x0 + L, num_nodes)
    # Node coordinates
    nodes = np.zeros((num_nodes, 3))
    nodes[:, 0] = x_coords
    # Elements: [node_i, node_j, material_id, section_id]
    elems = np.zeros((num_elems, 4), dtype=int)
    for e in range(num_elems):
        elems[e, 0] = e
        elems[e, 1] = e + 1
        elems[e, 2] = mat_id
        elems[e, 3] = sec_id
    return nodes, elems

# 4. FUNCTIONS:
def node_dof_indices(node_id: int) -> np.ndarray:
    start = DOF_PER_NODE * node_id
    return np.arange(start, start + DOF_PER_NODE)
def element_length_and_direction(nodes, node_i: int, node_j: int):
    xi, yi, zi = nodes[node_i]
    xj, yj, zj = nodes[node_j]
    dx, dy, dz = xj - xi, yj - yi, zj - zi
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    lx, ly, lz = dx / L, dy / L, dz / L
    return L, (lx, ly, lz)

# 5. TRANSFORMATION MATRIX:
def build_T_matrix(lx: float, ly: float, lz: float) -> np.ndarray:
    return np.eye(12) #Identity for now

# 6. SUPPORTS (1: Fixed, 0: Free):
def create_support_matrix(num_nodes: int, supports_dict: dict) -> np.ndarray:
    sup = np.zeros((num_nodes, DOF_PER_NODE), dtype=int)
    for node_id, dofs in supports_dict.items():
        sup[node_id, :] = np.array(dofs, dtype=int)
    return sup
def get_fixed_dofs(sup: np.ndarray) -> np.ndarray:
    fixed = []
    num_nodes = sup.shape[0]
    for node_id in range(num_nodes):
        for local_dof in range(DOF_PER_NODE):
            if sup[node_id, local_dof] == 1:
                g = node_id * DOF_PER_NODE + local_dof
                fixed.append(g)
    return np.array(sorted(set(fixed)), dtype=int)

# 7. ELEMENT MATERIAL STIFFNESS MATRIX (LOCAL) k:
def beam_local_stiffness(E, nu, A, Iy, Iz, J, L):
    k = np.zeros((12, 12), dtype=float)
    EA_L   = E * A / L
    EIz_L3 = E * 12 * Iz / L**3
    EIy_L3 = E * 12 * Iy / L**3
    EIz_L2 = E * 6  * Iz / L**2
    EIy_L2 = E * 6  * Iy / L**2
    EIy_L  = E * Iy / L
    EIz_L  = E * Iz / L
    GJ_L   = E * J / (2 * (1 + nu) * L)   

    k[0, 0] =  EA_L
    k[0, 6] = -EA_L
    k[1, 1]  =  EIz_L3
    k[1, 5]  =  EIz_L2
    k[1, 7]  = -EIz_L3
    k[1, 11] =  EIz_L2
    k[2, 2]  =  EIy_L3
    k[2, 4]  = -EIy_L2
    k[2, 8]  = -EIy_L3
    k[2, 10] = -EIy_L2
    k[3, 3]  =  GJ_L
    k[3, 9]  = -GJ_L
    k[4, 2]  = -EIy_L2
    k[4, 4]  =  4 * EIy_L
    k[4, 8]  =  EIy_L2
    k[4, 10] =  2 * EIy_L
    k[5, 1]  =  EIz_L2
    k[5, 5]  =  4 * EIz_L
    k[5, 7]  = -EIz_L2
    k[5, 11] =  2 * EIz_L
    k[6, 0] = -EA_L
    k[6, 6] =  EA_L
    k[7, 1]  = -EIz_L3
    k[7, 5]  = -EIz_L2
    k[7, 7]  =  EIz_L3
    k[7, 11] = -EIz_L2
    k[8, 2]  = -EIy_L3
    k[8, 4]  =  EIy_L2
    k[8, 8]  =  EIy_L3
    k[8, 10] =  EIy_L2
    k[9, 3]  = -GJ_L
    k[9, 9]  =  GJ_L
    k[10, 2]  = -EIy_L2
    k[10, 4]  =  2 * EIy_L
    k[10, 8]  =  EIy_L2
    k[10, 10] =  4 * EIy_L
    k[11, 1]  =  EIz_L2
    k[11, 5]  =  2 * EIz_L
    k[11, 7]  = -EIz_L2
    k[11, 11] = 4 * EIz_L
    return k

# 8. ELEMENT GEOMETRIC STIFFNESS MATRIX (LOCAL) kg:
def beam_local_geometric_stiffness_multiload(Fx2, Mx2, My1, My2, Mz1, Mz2, L, A, Ip):
    kg = np.zeros((12, 12), dtype=float)
 
    kg[0, 0] = Fx2 / L
    kg[0, 6] = -Fx2 / L
    kg[1, 1] = 6 * Fx2 / (5 * L)         
    kg[1, 3] = My1 / L
    kg[1, 4] = Mx2 / L
    kg[1, 5] = Fx2 / 10
    kg[1, 7] = -6 * Fx2 / (5 * L)
    kg[1, 9] = My2 / L
    kg[1, 10] = - Mx2 / L
    kg[1, 11] = Fx2 / 10
    kg[2, 2] = 6 * Fx2 / (5 * L)
    kg[2, 3] = Mz1 / L
    kg[2, 4] = -Fx2 / 10
    kg[2, 5] = Mx2 / L
    kg[2, 8] = -6 * Fx2 / (5 * L)
    kg[2, 9] = Mz2 / L
    kg[2, 10] = -Fx2 / 10
    kg[2, 11] = -Mx2 / L
    kg[3, 3] = Fx2 * Ip / (A * L)
    kg[3, 4] = -(2 * Mz1 - Mz2) / 6
    kg[3, 5] = (2 * My1 - My2) / 6
    kg[3, 7] = -My1 / L
    kg[3, 8] = -Mz1 / L
    kg[3, 9] = -Fx2 * Ip / (A * L)
    kg[3,10] = -(Mz1 + Mz2) / 6
    kg[3,11] = (My1 + My2) / 6
    kg[4,4] = 2 * Fx2 * L / 15
    kg[4,7] = -Mx2 / L
    kg[4,8] = Fx2 / 10
    kg[4,9] = -(Mz1 + Mz2) / 6
    kg[4,10] = -Fx2 * L / 30
    kg[4,11] = Mx2 / 2
    kg[5,5] = 2 * Fx2 * L / 15
    kg[5,7] = -Fx2 / 10
    kg[5,8] = -Mx2 / L
    kg[5,9] = (My1 + My2) / 6
    kg[5,10] = -Mx2/2
    kg[5,11] = - Fx2 * L / 30
    kg[6,6] = Fx2 / L
    kg[7,7] = 6 * Fx2 / (5 * L)
    kg[7,9] = -My2/L
    kg[7,10] = Mx2/L
    kg[7,11] = -Fx2/10
    kg[8,8] = 6 * Fx2 / (5 * L)
    kg[8,9] = -Mz2 / L
    kg[8,10] = Fx2 / 10   
    kg[8,11] = Mx2 / L
    kg[9,9] = Fx2 * Ip / (A * L)
    kg[9,10] = (Mz1 - 2 * Mz2) / 6
    kg[9,11] = -(My1 - 2 * My2) / 6
    kg[10,10] = 2 * Fx2 * L / 15
    kg[11,11] = 2 * Fx2 * L / 15
    for i in range(12):
        for j in range(i+1, 12):
            kg[j, i] = kg[i, j]
    return kg

# 9. ASSEMBLY OF THE GLOBAL MATERIAL STIFFNESS MATRIX K AND GEOMETRIC STIFFNESS MATRIX Kg
def assemble_global_matrices(MAT, SEC, NODES, ELEMS, PRELOADS):
    num_nodes = NODES.shape[0]
    ndof = DOF_PER_NODE * num_nodes
    num_elems = ELEMS.shape[0]
    K  = np.zeros((ndof, ndof))
    Kg = np.zeros((ndof, ndof))
    for e in range(num_elems):
        node_i, node_j, mat_id, sec_id = ELEMS[e, :]
        E, nu = MAT[mat_id, :]
        A, Iy, Iz, J = SEC[sec_id, :]
        L, (lx, ly, lz) = element_length_and_direction(NODES, node_i, node_j)
        k_local = beam_local_stiffness(E, nu, A, Iy, Iz, J, L)
        Fx2, Mx2, My1, My2, Mz1, Mz2 = PRELOADS[e, :]
        Ip = Iy + Iz
        kg_local = beam_local_geometric_stiffness_multiload(Fx2, Mx2, My1, My2, Mz1, Mz2, L, A, Ip)
        T = build_T_matrix(lx, ly, lz)
        ke  = T.T @ k_local  @ T
        kge = T.T @ kg_local @ T
        dofs_i = node_dof_indices(node_i)
        dofs_j = node_dof_indices(node_j)
        elem_dofs = np.concatenate([dofs_i, dofs_j])
        for a in range(12):
            A_dof = elem_dofs[a]
            for b in range(12):
                B_dof = elem_dofs[b]
                K[A_dof,  B_dof] += ke[a, b]
                Kg[A_dof, B_dof] += kge[a, b]
    return K, Kg

# 10. ASSEMBLY OF THE REDUCED GLOBAL MATERIAL STIFFNESS MATRIX (Kff) AND THE REDUCED GLOBAL GEOMETRIC STIFFNESS MATRIX (Kgff)
def apply_boundary_conditions(K, Kg, F, fixed_dofs):
    ndof = K.shape[0]
    all_dofs = np.arange(ndof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    Kff  = K[np.ix_(free_dofs, free_dofs)]
    Kgff = Kg[np.ix_(free_dofs, free_dofs)]
    Ff   = F[free_dofs]

    return Kff, Kgff, Ff, free_dofs

# 11. SOLVE FOR EIGENVALUES AND EIGENVECTORS
def solve_buckling(Kff, Kgff):
    if Kff.size == 0:
        return np.array([]), np.array([[]])

    from numpy.linalg import eig, solve
    A = solve(Kff, Kgff)   # equivalent to A = Kff^{-1} @ Kgff
    mu, vecs = eig(A)      # mu = 1/lambda
    mu = np.real(mu)
    vecs = np.real(vecs)
    # Avoid division by zero (or near-zero)
    eps = 1e-12
    valid = np.abs(mu) > eps
    lambdas = np.zeros_like(mu)
    lambdas[valid] = 1.0 / mu[valid]
    return lambdas, vecs





#SOLUTION FOR PROBLEM 2:

if __name__ == "__main__":

    # NODES AND ELEMENTS WITH 1D MESH IN X DIRECTION:
    L_total = 60 #Longitude [in]
    num_elems = 256 #Number of Elements
    mat_id = 0 #Id of Material
    sec_id = 0 #If of Section

    NODES, ELEMS = create_beam_mesh_1D(0.0, L_total, num_elems, mat_id, sec_id) #Create Mesh with num_elems as Number of Elements
    num_nodes = NODES.shape[0]
    ndof = DOF_PER_NODE * num_nodes

    supports = {
        0: [1, 1, 1, 1, 1, 1], #1: Fixed, 0: Free.
    }
    SUP = create_support_matrix(num_nodes, supports)
    fixed_dofs = get_fixed_dofs(SUP)

    # EXTERNAL LOADS PER NODE:
    LOADS = np.zeros((num_nodes, 6))  # [Fx, Fy, Fz, Mx, My, Mz] per node
    F = LOADS.reshape(-1)

    # PRELOADS PER ELEMENT:
    PRELOADS = np.zeros((num_elems, 6))
    P_ref = 1  # Reference Force (kip)

    for e in range(num_elems):

        x_i = NODES[e, 0]
        x_j = NODES[e + 1, 0]

        Fx2 = 0
        Mx2 = 0
        My1 = 0
        My2 = 0
        Mz1 = P_ref * (L_total - x_i)
        Mz2 =  -P_ref * (L_total - x_j)
        PRELOADS[e, :] = [Fx2, Mx2, My1, My2, Mz1, Mz2]

    # ASSEMBLY OF THE GLOBAL MATERIAL STIFFNESS MATRIX K AND GEOMETRIC STIFFNESS MATRIX Kg
    K, Kg = assemble_global_matrices(MAT, SEC, NODES, ELEMS, PRELOADS)

    # ASSEMBLY OF THE REDUCED GLOBAL MATERIAL STIFFNESS MATRIX (Kff) AND THE REDUCED GLOBAL GEOMETRIC STIFFNESS MATRIX (Kgff)
    Kff, Kgff, Ff, free_dofs = apply_boundary_conditions(K, Kg, F, fixed_dofs)
    print("Total DOFs:", ndof)
    print("Fixed DOFs:", (fixed_dofs + 1).tolist())

    # SOLVE FOR EIGENVALUES AND EIGENVECTORS
    eigvals, eigvecs = solve_buckling(Kff, Kgff)
    if eigvals.size > 0:
        positive = eigvals[eigvals > 0]
        if positive.size > 0:
            lam_cr = np.min(positive)
            P_cr = lam_cr * P_ref
            print("P_cr =", P_cr, "kip")
        else:
            print("No positive eigenvalues.")
    else:
        print("No free DOFs (check supports).")
