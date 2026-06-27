import numpy as np

# ============================================================
# ROTATION MATRICES (UNCHANGED MATH)
# ============================================================
def rotation_matrix_euler(phi, theta, psi):
    Rz1 = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi),  np.cos(phi), 0],
                    [0,            0,           1]])

    Ry = np.array([[np.cos(theta),  0, np.sin(theta)],
                   [0,              1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])

    Rz2 = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi),  np.cos(psi), 0],
                    [0,            0,           1]])

    return Rz2 @ Ry @ Rz1


def rotate_tensor_4d_euler(T, phi, theta, psi):
    R = rotation_matrix_euler(phi, theta, psi)
    return np.einsum('ia,jb,kc,ld,abcd->ijkl', R, R, R, R, T)


# ============================================================
# VOIGT <-> 4TH ORDER TENSOR (UNCHANGED MATH)
# ============================================================
voigt_map = {
    (0,0):0,(1,1):1,(2,2):2,
    (1,2):3,(2,1):3,
    (0,2):4,(2,0):4,
    (0,1):5,(1,0):5
}

def matrix_to_four_range(matrix):
    T = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    T[i,j,k,l] = matrix[voigt_map[(i,j)], voigt_map[(k,l)]]
    return T


def four_range_to_matrix(tensor):
    M = np.zeros((6,6))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    m = voigt_map[(i,j)]
                    n = voigt_map[(k,l)]
                    M[m,n] = tensor[i,j,k,l]
                    M[n,m] = tensor[i,j,k,l]
    return M


# ============================================================
# MATERIAL MATRIX (UNCHANGED)
# ============================================================
Cmat = np.array([
 [4.975E02,1.63E02,1.16E02,2.2E01,0,0],
 [1.63E02,4.975E02,1.17E02,-2.2E01,0,0],
 [1.16E02,1.16E02,5.01E02,0,0,0],
 [2.2E01,-2.2E01,0,1.47E02,0,0],
 [0,0,0,0,1.47E02,0],
 [0,0,0,0,0,1.67E02]
])



Cten = matrix_to_four_range(Cmat)


# ============================================================
# LOAD GRAIN ORIENTATIONS
# ============================================================
datapoly = np.loadtxt("datapoly.txt", dtype=np.float64)

phis   = np.radians(datapoly[:,1])
thetas = np.radians(datapoly[:,2])
psis   = np.radians(datapoly[:,3])


# ============================================================
# ROTATE ALL TENSORS (FASTER LOOP)
# ============================================================
C_rotated_all = np.array([
    rotate_tensor_4d_euler(Cten, p, t, s)
    for p, t, s in zip(phis, thetas, psis)
])

print("Rotated tensor array shape:", C_rotated_all.shape)


# ============================================================
# CONVERT TO 6x6 MATRICES
# ============================================================
matrix_rot_all = np.array([
    four_range_to_matrix(T) for T in C_rotated_all
])

print("Rotated Voigt matrix shape:", matrix_rot_all.shape)


# ============================================================
# WRITE OUTPUT FILE AS FIXED-WIDTH TXT (FORTRAN STYLE)
# ============================================================
filename = "output.txt"

with open(filename, "w") as file:
    for prop_matrix in matrix_rot_all:

        C11, C12, C13, C14, C15, C16 = prop_matrix[0]
        C22, C23, C24, C25, C26      = prop_matrix[1,1:]
        C33, C34, C35, C36           = prop_matrix[2,2:]
        C44, C45, C46                = prop_matrix[3,3:]
        C55, C56                     = prop_matrix[4,4:]
        C66                          = prop_matrix[5,5]

        float_values = [
            C11,C12,C13,C14,C15,C16,
            C22,C23,C24,C25,C26,
            C33,C34,C35,C36,
            C44,C45,C46,
            C55,C56,
            C66
        ]

        # ✅ EXACTLY like your example: one leading space per number
        line = "".join(f"{x:14.7E}" for x in float_values)
        file.write(line + "\n")

print("TXT output written to:", filename)

