#!/usr/bin/env python
"""
Generates a GIZMO/Gadget/AREPO HDF5 initial conditions and params files for the standard GMC initial condition: a
uniform-density cloud (realized as particle glass configuration) in an ambient medium 1/1000 the density. Units system
is defined by Msun - km/s - pc - gauss; if you are using another units system for ISM simulations revisit your life
choices. Initial temperature is taken to be 100K (1e4K) in the cloud (ambient medium), both consisting of neutral H.

Usage:
./gizmo_IC.py [options]

Options:
    -h --help         Show this screen
    --M=<msun>        Mass of the cloud in solar [default: 2e4]
    --R=<pc>          Radius of the cloud in pc [default: 10]
    --dm=<msun>       Target mass resolution in solar [default: 1e-3]
    --N=<num>         Number of equal-mass cells/particles in the cloud - overrides dm
    --alpha=<val>     Virial parameters (2x ratio of turbulent to gravitational energy) [default: 2]
    --B_scale=<f>     Magnetic field strength relative to Crutcher 2010 fit for the cloud density [default: 0.333]
    --box_scale=<v>   Scaling factor of box side-length in units of the cloud radius [default: 10.0]
    --ISRF=<f>        Scaling factor for the ISRF (affects params file options for GIZMO) [default: 1.0]
    --Z=<f>           Scaling factor for the metallicity (affects params file options for GIZMO) [default: 1.0]
    --z=<f>           Redshift assumed for the ambient radiation background field [default: 0.0]
    --amb_dens=<f>    Scaling factor for ambient density relative to cloud density [default: 1e-3]
    --dens_power=<p>  Power such that density goes as r^p [default: 0.0]
"""

from docopt import docopt
import h5py
from astropy import units as u, constants as c
import numpy as np
import os
from meshoid import Meshoid
from scipy.ndimage import gaussian_filter


def get_glass_coords(N_gas: int, verbose=False) -> np.ndarray:
    printv = print if verbose else lambda *a, **k: None
    glass_path = os.path.expanduser("~") + "/glass_orig.npy"
    if not os.path.exists(glass_path):
        import urllib.request

        printv("Downloading glass file...")
        urllib.request.urlretrieve("http://www.tapir.caltech.edu/~mgrudich/glass_orig.npy", glass_path)

    x = np.load(glass_path)
    while len(x) * np.pi * 4 / 3 / 8 < N_gas:
        printv(
            "Need %d particles, have %d. Tessellating 8 copies of the glass file to get required particle number"
            % (N_gas * 8 / (4 * np.pi / 3), len(x))
        )
        x = np.concatenate(
            [
                x / 2 + i * np.array([0.5, 0, 0]) + j * np.array([0, 0.5, 0]) + k * np.array([0, 0, 0.5])
                for i in range(2)
                for j in range(2)
                for k in range(2)
            ]
        )
    printv("Glass loaded!")
    return x


def cloud_coordinates(
    num_cloud_cells: int, box_size: float, cloud_radius: float, density_exponent: float = 0.0, verbose=False
) -> np.ndarray:
    """Generates positions of gas cells in the spherical cloud, taken from a uniform-density glass-like particle
    arrangement.

    Parameters
    ----------
    num_cloud_cells: int
        Number of gas cells in the spherical cloud
    cloud_radius: float
        Radius of the cloud in code units
    density_exponent: float, optional
        p such that density goes as r^p, if a radial power-law density profile is desired (default 0)
    verbose: boolean, optional
        Write what this function is doing to stdout (default False)

    Returns
    -------
    x: np.ndarray
        Shape (num_cloud_cells, 3) array of gas cell positions.
    """
    printv = print if verbose else lambda *a, **k: None

    x = get_glass_coords(num_cloud_cells, verbose)
    x = 2 * (x - 0.5)

    def radius(x):
        """Returns radius of shape (N,d) coordinates x from the origin."""
        printv("Computing radii...")
        return np.sqrt(np.sum(x * x, axis=1))

    r = radius(x)
    printv("Done! Sorting coordinates...")
    x = x[r.argsort()][:num_cloud_cells]
    printv("Done! Rescaling...")
    x *= (float(len(x)) / num_cloud_cells * 4 * np.pi / 3 / 8) ** (1.0 / 3)  # * cloud_radius
    printv("Done! Recomupting radii...")
    r = radius(x)
    printv("Doing density profile...")
    rnew = (r / r.max()) ** (3.0 / (3 + density_exponent)) * cloud_radius
    x = x * (rnew / r)[:, None]
    x[rnew == 0.0] = np.zeros((3,))
    r = radius(x)
    return np.take(x, r.argsort(), axis=0) + 0.5 * box_size


def ambient_coordinates(num_ambient_cells: int, box_size: float, cloud_radius: float, verbose=False) -> np.ndarray:
    """
    Returns the coordinates of the box-filling ambient medium gas cells

    Parameters
    ----------
    num_cloud_cells: int
        Number of gas cells in the ambient medium
    cloud_radius: float
        Radius of the cloud in code units
    verbose: boolean, optional
        Print what the routine is doing to stdout (default False)

    Returns
    -------
    x: np.ndarray
        Shape (num_cloud_cells, 3) array of gas cell positions.
    """
    # start with a cube with enough cells to fill the whole box
    volume_fac = box_size**3 / (box_size**3 - 4 * np.pi / 3 * cloud_radius**3)
    num_initial_cube = int(volume_fac * num_ambient_cells + 1)
    x = get_glass_coords(num_initial_cube, verbose)  # has at least num_initial cube in it
    x = x[np.max(x, axis=1).argsort()][:num_initial_cube]  # take the cube containing the desired number
    x /= np.max(x) * (1 + np.finfo(x.dtype).eps)  # normalize to farthest corner
    x = x * box_size  # scale coordinates to box size

    r_center = np.sqrt(np.sum((x - 0.5 * box_size) ** 2, axis=1))  # calculate distance from box center
    x = x[r_center.argsort()][(num_initial_cube - num_ambient_cells) :]  # excise central sphere by cutting small radii
    return x


def load_seed_field(path):
    """Returns a tuple of 2 dicts containing the seed field coordinates and velocities, respectively."""
    coords, velocities = {}, {}
    with h5py.File(path, "r") as F:
        for dir in "xyz":
            coords[dir] = np.array(F[dir])
            velocities[dir] = np.array(F["v" + dir])

    return coords, velocities


def interpolate_velocity_to_cloud(
    x_cloud: np.ndarray, box_size: float, cloud_radius: float, order: int = 1, smooth_to_resolution=True
) -> np.ndarray:
    coords, vel = load_seed_field("../../velocity_field_seed0.h5")
    for dir in "xyz":
        coords[dir] = coords[dir] * 2 * cloud_radius
        coords[dir] += 0.5 * box_size - cloud_radius

    # if this is a low-res simulation, should smooth to that resolution
    if smooth_to_resolution:
        res_effective = 2 * (3 * x_cloud.shape[0] / (4 * np.pi)) ** (1.0 / 3)
        if res_effective < vel["x"].shape[0]:
            sigma = vel["x"].shape[0] / res_effective
            for dir in "xyz":
                vel[dir] = gaussian_filter(vel[dir], sigma, mode="wrap")

    coords_flat = np.array([coords[dir].flatten() for dir in "xyz"]).T
    vel_flat = np.array([vel[dir].flatten() for dir in "xyz"]).T
    v_cloud = Meshoid(coords_flat, boxsize=box_size.value).Reconstruct(vel_flat, x_cloud.value, order=order)
    return v_cloud


def crutcher_Bfield(density, X_H=0.76):
    B0, n0, alpha = 1e-5 * u.gauss, 300 * u.cm**-3, 0.65
    n = (density * X_H / c.m_p).to(u.cm**-3)
    return max(B0, B0 * (n / n0) ** alpha)


def grav_energy_sphere(mass: u.Quantity, radius: u.Quantity, p: int = 0):
    """Gravitational binding energy of a sphere of given mass, radius, and density profile index p

    e.g. -3/5 GM^2/R  for p=0 (uniform sphere)
    """
    return -(3 + p) / (2 * p + 5) * c.G * mass * mass / radius


def normalize_velocity(v, masses, cloud_mass, cloud_radius, alpha, p):
    kinetic_energy_target = np.abs(grav_energy_sphere(cloud_mass, cloud_radius, p)) * (alpha / 2)
    v[:] = v * np.sqrt(kinetic_energy_target / np.sum(0.5 * masses[:, None] * v * v))


def divv_errnorm(x, v, box_size):
    M = Meshoid(x, boxsize=box_size.value)
    div = M.Div(v).std()
    curl = np.sum(M.Curl(v) ** 2, axis=1).mean() ** 0.5
    return div / curl


def magnetic_energy(B: u.Quantity, cloud_radius: u.Quantity):
    return B**2 / (2 * c.mu0) * (4 * np.pi / 3 * cloud_radius**3)


def turbulent_energy(masses: u.Quantity, v: u.Quantity):
    return 0.5 * np.sum(masses[:, None] * v**2)


def make_IC_and_paramsfile(args):
    """Master routine that parses options and generates the IC and parameter file."""
    for k in args.keys():
        if k == "--help":
            continue
        if k is not None and args[k] is not None:
            args[k] = float(args[k])

    unit_mass = u.Msun
    unit_length = u.pc
    unit_dens = unit_mass / unit_length**3
    unit_speed = u.km / u.s
    unit_time = unit_length / unit_speed
    unit_magnetic_field = u.gauss

    cloud_mass = args["--M"] * unit_mass
    cloud_radius = args["--R"] * unit_length
    cloud_volume = 4 * np.pi / 3 * cloud_radius**3
    cloud_density = cloud_mass / cloud_volume
    alpha = args["--alpha"]
    ambient_density = cloud_density * args["--amb_dens"]
    box_size = args["--box_scale"] * cloud_radius
    ambient_mass = (box_size**3 - cloud_volume) * ambient_density
    dens_power = args["--dens_power"]
    frac_B = args["--B_scale"]

    if args["--N"] is not None:
        dm = args["--M"] * unit_mass / args["--N"]
        num_cloud_cells = int(args["--N"])
    else:
        dm = args["--dm"] * unit_mass
        num_cloud_cells = int(round(cloud_mass / dm))
    num_ambient_cells = int(round(ambient_mass / dm))
    num_cells = num_cloud_cells + num_ambient_cells

    x_cloud = cloud_coordinates(num_cloud_cells, box_size, cloud_radius)
    x_ambient = ambient_coordinates(num_ambient_cells, box_size, cloud_radius)
    v_cloud = interpolate_velocity_to_cloud(x_cloud, box_size, cloud_radius)
    masses = np.repeat(dm, num_cells)

    x = np.concatenate([x_cloud, x_ambient])
    v = np.concatenate([v_cloud, np.zeros((num_ambient_cells, 3))]) * unit_speed

    normalize_velocity(v, masses, cloud_mass, cloud_radius, alpha, dens_power)
    B = frac_B * crutcher_Bfield(cloud_density)
    #    print(magnetic_energy(B, cloud_radius).to(u.erg) / grav_energy_sphere(cloud_mass, cloud_radius).to(u.erg))
    B = np.c_[np.zeros(num_cells), np.zeros(num_cells), np.ones(num_cells)] * B

    T_cold = 100.0 * u.K
    T_warm = 1e4 * u.K
    mu_neutral = 1.28
    u_cold = (1.5 * c.k_B * T_cold / (mu_neutral * c.m_p)).to(unit_speed**2)
    u_warm = (1.5 * c.k_B * T_warm / (mu_neutral * c.m_p)).to(unit_speed**2)

    spec_energy = u_cold * np.ones(num_cells)
    spec_energy[-num_ambient_cells:] = u_warm

    rho = cloud_density * np.ones(num_cells)
    rho[-num_ambient_cells:] = ambient_density

    hsml = (dm / rho) ** (1.0 / 3) * 2

    IC_path = f"./M{cloud_mass.value}_R{cloud_radius.value}_N{num_cloud_cells}_alpha{alpha}_B{frac_B}_Z{args["--Z"]}_z{args["--z"]}_I{args["--ISRF"]}.hdf5"
    with h5py.File(IC_path, "w") as F:
        F.create_group("PartType0")
        F.create_group("Header")
        F["Header"].attrs["NumPart_Total"] = [num_cells] + 5 * [0]
        F["Header"].attrs["NumPart_ThisFile"] = [num_cells] + 5 * [0]
        F["Header"].attrs["BoxSize"] = box_size
        F["Header"].attrs["Time"] = 0.0
        p0 = F["PartType0"]
        p0.create_dataset("Masses", data=masses)
        p0.create_dataset("SmoothingLength", data=hsml)
        p0.create_dataset("Density", data=rho)
        p0.create_dataset("Coordinates", data=x)
        p0.create_dataset("InternalEnergy", data=spec_energy)
        p0.create_dataset("ParticleIDs", data=1 + np.arange(num_cells))
        p0.create_dataset("MagneticField", data=B)
        p0.create_dataset("Velocities", data=v)

    make_paramsfile(IC_path, args, box_size, cloud_mass, cloud_radius, unit_time, dm)


def freefall_time(cloud_mass: u.Quantity, cloud_radius: u.Quantity):
    return np.pi / 2 * np.sqrt(cloud_radius**3 / (2 * c.G * cloud_mass))


def make_paramsfile(
    path: str,
    args: dict,
    box_size: u.Quantity,
    cloud_mass: u.Quantity,
    cloud_radius: u.Quantity,
    unit_time: u.Quantity,
    dm: u.Quantity,
):
    # convert both to s due to astropy bug not reducing dimensionless ratios?
    tff_code = (freefall_time(cloud_mass, cloud_radius) / unit_time).to(u.dimensionless_unscaled)
    prefix = path.split(".hdf5")[0].split("/")[1]

    SOFTENING_AU = 20.0
    SOFTENING_PLUMMER_TO_KERNEL = 2.8
    paramsfile_defaults = {
        "InitCondFile": prefix,
        "OutputDir": "output",
        "BoxSize": box_size.value,
        "TimeMax": 10 * tff_code,
        "TimeBetSnapshot": tff_code / 150,
        "MaxSizeTimestep": tff_code / 300,
        "UnitLength_in_cm": 3.09e18,
        "UnitMass_in_g": 1.99e33,
        "UnitVelocity_in_cm_per_s": 1.00e5,
        "UnitMagneticField_in_gauss": 1.0,
        "MaxMemSize": 5000,
        "PartAllocFactor": 5.0,
        "BufferSize": 500,
        "TimeLimitCPU": 604800,
        "CpuTimeBetRestartFile": 7200,
        "DesNumNgb": 32,
        "Softening_Type0": 1e-10,
        "Softening_Type1": 1e-10,
        "Softening_Type2": 1e-10,
        "Softening_Type3": 1e-10,
        "Softening_Type4": 1e-10,
        "Softening_Type5": SOFTENING_AU * u.au.to(u.pc) / SOFTENING_PLUMMER_TO_KERNEL,
        "InterstellarRadiationFieldStrength": float(args["--ISRF"]),
        "InitMetallicity": float(args["--Z"]),
        "SeedBlackHoleMass": dm.value / 10,
        "BAL_wind_particle_mass": dm.value / 10,
        "BAL_wind_particle_mass_MS": dm.value / 100,
        "Redshift_RT_Background": args["--z"],
        "CritPhysDensity": 1e13,
    }

    with open(f"params_{prefix}.txt", "w") as F:
        for k, i in paramsfile_defaults.items():
            F.write(f"{k}    {i}\n")


if __name__ == "__main__":
    args = docopt(__doc__)
    make_IC_and_paramsfile(args)
