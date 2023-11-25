import signal
import numpy as np
import scipy.interpolate as interp
import numpy.polynomial.legendre as leg

def save_potential(path: str, filename: str, r_start: float, r_end: float, r_points_no: int, polar_points_no: int, kx, ky, is_gamma: bool):
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    r_step = (r_end - r_start) / (r_points_no - 1)
    r_grid = np.array([r_start + i * r_step for i in range(r_points_no)])

    x, _ = leg.leggauss(polar_points_no)
    theta_grid = np.flip(np.arccos(x))

    potential_to_npy(path, filename, r_grid, theta_grid, lambda x: np.sqrt(x), lambda x: x**2, kx=kx, ky=ky) if is_gamma else potential_to_npy(path, filename, r_grid, theta_grid, kx=kx, ky=ky)

def potential_to_npy(path, filename, r_grid, theta_grid, transformation = None, inverse_transformation = None, kx=3, ky=3):
    thetas, rs, V_values = load_from_file(path, filename)

    rs_full = rs

    for i in range(len(thetas)):
        rs_extended_0, V_extended_0 = extrapolate_to_zero(rs, V_values[i])
        rs_extended_infty, V_extended_infty = extrapolate_to_infinity(rs, V_values[i], r_grid[-1])
        rs_full = rs_extended_0 + rs + rs_extended_infty
        V_values[i] = V_extended_0 + V_values[i] + V_extended_infty

    V_values_np = np.array(V_values).T

    interpolation = interp.RectBivariateSpline(rs_full, thetas, V_values_np if transformation == None else transformation(V_values_np), kx=kx, ky=ky, s=0)

    V_grid = interpolation(r_grid, theta_grid)

    if transformation != None and inverse_transformation == None:
        raise Exception("inverse_transformation must be specified if transformation is specified")
    
    if inverse_transformation != None:
        V_grid = inverse_transformation(V_grid)
    
    np.save(path + filename.split(".")[0] + ".npy", V_grid)
    # save start, end and number of points in grids
    np.save(path + filename.split(".")[0] + "_grid.npy", [r_grid[0], r_grid[-1], r_grid.shape[0], theta_grid[0], theta_grid[-1], theta_grid.shape[0]])

def load_from_file(path, filename):
    thetas = []
    rs = []
    V_values = []
    
    i = 0
    V_theta = []

    with open(path + filename) as file:
        contents = file.readlines()
        for line in contents:
            if line.startswith("# theta ="):
                i += 1
                thetas.append(float(line.split("=")[1]) * np.pi / 180)

                if i > 1:
                    V_values.append(V_theta)

                V_theta = []
                continue
            if line.startswith("#"):
                continue

            lineSplitted = line.split()
            if i == 1:
                rs.append(float(lineSplitted[0]))

            V_theta.append(float(lineSplitted[1]))
        V_values.append(V_theta)
    
    return thetas, rs, V_values

def extrapolate_to_zero(rs, V_values, max_val=1000):
    dr = rs[1] - rs[0]
    dVRatio = V_values[0] / V_values[1]
    dVRatio *= 1.005

    rs_extended = [rs[0] - dr]
    V_extended = [V_values[0] * dVRatio]

    dumping = 0
    while rs_extended[-1] > 0:
        if V_extended[-1] >= max_val:
            V_extended[-1] = max_val + 1000 * np.sqrt(dumping)
            dVRatio = 1
            dumping += 1
        else:
            dVRatio *= 1.005
        rs_extended.append(rs_extended[-1] - dr)
        V_extended.append(V_extended[-1] * dVRatio)

    rs_extended.reverse()
    V_extended.reverse()

    return rs_extended, V_extended

def extrapolate_to_infinity(rs, V_values, max_distance = 60):
    dr = rs[-1] - rs[-2]
    r_extended = [rs[-1] + dr]
    V_extended = [0.0]
    while(r_extended[-1] < max_distance):
        r_extended.append(r_extended[-1] + dr)
        V_extended.append(0.0)

    return r_extended, V_extended

if __name__ == "__main__":
    save_potential("C:/Users/marcr/Documents/vs_code/Ne_Ocs/", "potentials/potential.dat", 0.0, 60.0, 1024, 160, kx=5, ky=5, is_gamma=False)