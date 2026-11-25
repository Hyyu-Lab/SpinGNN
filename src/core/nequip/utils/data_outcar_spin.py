from io import open
import re
import numpy as np
from ase.io import read
import warnings
import pandas as pd
import os


def voigt_6_to_full_3x3_piezo(stress_vector):
    """
    Form a 3x3 stress matrix from a 6 component vector in Voigt notation
    """
    s1, s2, s3, s4, s5, s6 = np.transpose(stress_vector)
    return np.transpose([[s1, s4, s6], [s4, s2, s5], [s6, s5, s3]])


def get_outcar_magnetization_born(filename):
    """
    Get magnetization in OUTCAR.
    Based on Pymatgen Outcar Parser.
    Note that this data is not always present in OUTCAR. LORBIT must be set to some
    other value than the default.

    Args:
        filename (str): Path of OUTCAR

    Return:
        mag (np.array): In shape of (natoms,3). For collinear, mag[:,1:] is zero and store magnetization in mag[:,0]

    Original Authors: Rickard Armiento, Shyue Ping Ong
    Revised by Hongyu Yu
    """
    filename = filename

    mag_x = []
    mag_y = []
    mag_z = []
    header = []

    all_lines = []
    for line in open(filename, "r").readlines():
        clean = line.strip("-").strip()
        all_lines.append(clean)

    # For single atom systems, VASP doesn't print a total line, so
    # reverse parsing is very difficult
    read_mag_x = False
    read_mag_y = False  # for SOC calculations only
    read_mag_z = False
    lconvergence = False
    nsw = 0
    nelm = 0
    iteraton_nsw = 0
    iteraton_nelm = 0
    lfinish_job = False

    lread_born = False
    born = []

    lread_piezo = False
    piezo = []

    for clean in all_lines:
        if clean == "General timing and accounting informations for this job:":
            lfinish_job = True
        if clean.startswith("NSW"):
            nsw = [int(i) for i in re.findall(r"[\d]+", clean)][0]
        if clean.startswith("NELM"):
            nelm = [int(i) for i in re.findall(r"[\d]+", clean)][0]
        if clean.startswith("Iteration"):
            m = re.match(r"Iteration\s*\d+\(\s+\d+\)\s*", clean)
            if m:
                tmp = [int(i) for i in re.findall(r"[\d]+", clean)]
                iteraton_nsw = tmp[0]
                iteraton_nelm = tmp[1]
        if read_mag_x or read_mag_y or read_mag_z:
            if clean.startswith("# of ion"):
                header = re.split(r"\s{2,}", clean.strip())
                header.pop(0)
            else:
                m = re.match(r"\s*(\d+)\s+(([\d\.\-]+)\s+)+", clean)
                if m:
                    toks = [float(i) for i in re.findall(r"[\d\.\-]+", clean)]
                    toks.pop(0)
                    if read_mag_x:
                        mag_x.append(dict(zip(header, toks)))
                    elif read_mag_y:
                        mag_y.append(dict(zip(header, toks)))
                    elif read_mag_z:
                        mag_z.append(dict(zip(header, toks)))
                elif clean.startswith("tot"):
                    read_mag_x = False
                    read_mag_y = False
                    read_mag_z = False
        if clean == "magnetization (x)":
            mag_x = []
            read_mag_x = True
            read_mag_y, read_mag_z = False, False
        elif clean == "magnetization (y)":
            mag_y = []
            read_mag_y = True
            read_mag_x, read_mag_z = False, False
        elif clean == "magnetization (z)":
            mag_z = []
            read_mag_z = True
            read_mag_x, read_mag_y = False, False

        if lread_born:
            if clean.startswith("ion") or clean.startswith("--------"):
                pass
            elif (
                clean.startswith("CHARGE:")
                or clean == "\n"
                or clean.startswith("LOOP+:")
            ):
                lread_born = False
            else:
                tmp = [float(i) for i in re.findall(r"[\d\.\-]+", clean)]
                if len(tmp) == 4:  # tmp should be in length of 4
                    born.append(tmp[1:])
        if clean.startswith("BORN EFFECTIVE CHARGES"):
            lread_born = True

        if lread_piezo:
            if clean.startswith("XX") or clean.startswith("--------"):
                pass
            elif clean.startswith(
                "PIEZOELECTRIC TENSOR  for field in x, y, z        (C/m^2)"
            ) or clean.startswith(
                "PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z        (C/m^2)"
            ):
                lread_piezo = False
            else:
                tmp = [float(i) for i in re.findall(r"[\d\.\-]+", clean)]
                if len(tmp) == 6:  # tmp should be in length of 4
                    piezo.append(tmp)
        if clean.startswith(
            "PIEZOELECTRIC TENSOR  for field in x, y, z        (e  Angst)"
        ) or clean.startswith(
            "PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z        (e  Angst)"
        ):
            lread_piezo = True

    if born:
        lborn = True
        born = np.array(born, dtype=np.float64).reshape(-1, 9)
    else:
        lborn = False

    if piezo:
        lpiezo = True
        piezo_tensor = np.zeros((3, 3, 3), dtype=np.float64)
        for i in range(3):
            piezo_tensor[i] = voigt_6_to_full_3x3_piezo(piezo[i])
    else:
        lpiezo = False
        piezo_tensor = None

    lconvergence = (
        (nelm > iteraton_nelm) and ((nsw > iteraton_nsw) or nsw == 0) and lfinish_job
    )

    # merge x, y and z components of magmoms if present (SOC calculation)
    key = "tot"
    natom = len(mag_x)
    mag = []
    if mag_y and mag_z:
        for idx in range(natom):
            mag.append([mag_x[idx][key], mag_y[idx][key], mag_z[idx][key]])
    elif mag_x:
        for idx in range(natom):
            mag.append([mag_x[idx][key], 0, 0])
    if mag:
        mag = np.array(mag, dtype=np.float64)
        assert mag.shape == (natom, 3)
        lmagnetization = True
    else:
        lmagnetization = False

    return mag, lconvergence, lmagnetization, born, lborn, piezo_tensor, lpiezo


def _read_oszicar_atom_block(df, start, atoms_num, marker, filename):
    block = df.iloc[start : start + atoms_num, :4]
    if len(block) != atoms_num:
        raise ValueError(
            f"{filename}: {marker} block has {len(block)} atom rows; "
            f"expected {atoms_num}"
        )

    numeric = block.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{filename}: {marker} block contains non-finite values")

    raw_atom_ids = numeric[:, 0]
    if not np.equal(raw_atom_ids, np.floor(raw_atom_ids)).all():
        raise ValueError(f"{filename}: {marker} block contains non-integer atom IDs")
    atom_ids = raw_atom_ids.astype(np.int64) - 1
    if len(np.unique(atom_ids)) != atoms_num:
        raise ValueError(f"{filename}: {marker} block contains duplicate atom IDs")
    if np.any(atom_ids < 0) or np.any(atom_ids >= atoms_num):
        raise ValueError(f"{filename}: {marker} atom IDs must be in [1, {atoms_num}]")

    values = np.zeros((atoms_num, 3), dtype=np.float64)
    values[atom_ids] = numeric[:, 1:4]
    return values


def read_OSZICAR(filename, vasp6=False, on_missing="error"):
    if os.path.exists(filename) is False:
        raise FileNotFoundError(f"OSZICAR file not found: {filename}")
    if on_missing not in ("error", "skip"):
        raise ValueError("on_missing must be either 'error' or 'skip'")

    df = pd.read_table(
        filename, header=None, sep=r"\s+", names=range(20), on_bad_lines="skip"
    )
    marker_rows = {
        marker: df.index[df[1].eq(marker)].to_list()
        for marker in ("lambda*MW_perp", "MW_int")
    }
    missing_markers = [marker for marker, rows in marker_rows.items() if not rows]
    if missing_markers:
        if on_missing == "skip":
            return None, None
        raise ValueError(
            f"{filename}: missing required OSZICAR marker(s): "
            + ", ".join(missing_markers)
        )

    spin_forces_idx = marker_rows["lambda*MW_perp"][-1]
    if spin_forces_idx < 4:
        raise ValueError(
            f"{filename}: lambda*MW_perp marker has no preceding atom count"
        )
    try:
        atoms_num_float = float(df.iloc[spin_forces_idx - 4, 0])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{filename}: invalid atom count before lambda*MW_perp marker"
        ) from exc
    if (
        not np.isfinite(atoms_num_float)
        or atoms_num_float <= 0
        or atoms_num_float != np.floor(atoms_num_float)
    ):
        raise ValueError(f"{filename}: invalid atom count before lambda*MW_perp marker")
    atoms_num = int(atoms_num_float)

    lambda_MW_perp = _read_oszicar_atom_block(
        df,
        spin_forces_idx + 1,
        atoms_num,
        "lambda*MW_perp",
        filename,
    )
    MW_idx = marker_rows["MW_int"][-1]
    MW_int = _read_oszicar_atom_block(df, MW_idx + 1, atoms_num, "MW_int", filename)
    # If you are using vasp6+, the factor 2 is NOT needed.
    spin_force = lambda_MW_perp if vasp6 else 2 * lambda_MW_perp
    # spin force -dE/dS, in unit of eV/uB here. Should use -|M|*dE/dS for network prediction in unit eV
    return spin_force, MW_int


def read_vasp_atom(
    filename,
    non_mag_atomic_number=[],
    min_mag_norm=0.5,
    lmagnetization_data=False,
    lnormalize_spin=True,
    lread_born=False,
    lread_piezo=False,
    lforce_convergence=False,
    lread_OSZICAR=True,
    vasp6=False,
):
    """
    Get atoms with spin in OUTCAR.
    ASE get everything except spin.
    Get spin from magnetization in OUTCAR by revised Pymatgen Outcar parser.
    Note that magnetization is not always present in OUTCAR. LORBIT must be set to some
    other value than the default.

    Args:
        filename (str): Path of OUTCAR
        non_mag_atomic_number (list): atomic numbers of atoms without magnetization which Will be set to zero.
        min_mag_norm (float): if ind_non_mag is not provided, magnetization lower than min_mag_norm will be set to zero.
        lmagnetization_data (bool): whether to read magnetization data from OUTCAR.
        lnormalize_spin (bool): whether to normalize spin to 1.
        lread_born (bool): whether to read born effective charges from OUTCAR.
        lread_piezo (bool): whether to read piezoelectric tensor from OUTCAR.
        lforce_convergence (bool): whether to check convergence.
        lread_OSZICAR (bool): whether to read OSZICAR to get the spin forces.

    Return:
        Ase.atoms with `initial_magmoms` as spin
    """
    if "OUTCAR" in filename:
        (
            spin,
            lconvergence,
            lmagnetization,
            born,
            lborn,
            piezo_tensor,
            lpiezo,
        ) = get_outcar_magnetization_born(filename)
        if lforce_convergence:
            lconvergence = True
    elif "vasprun.xml" in filename:
        lconvergence = True
        warnings.warn(
            "Convergence of vasprun.xml is not checked. Please make sure it is converged."
        )
        assert (
            lmagnetization_data is False
        ), "magnetization data is not supported in vasprun.xml process yet"
        assert (
            lread_born is False
        ), "bec data is not supported in vasprun.xml process yet"
    else:
        raise ValueError("Only OUTCAR and vasprun.xml are supported now")

    if lconvergence:
        # Read atoms use ase
        try:
            atoms = read(filename)
        except Exception:
            print("Fail to read with ASE:", filename)
            return None, False
        # Born effective charge related
        if lread_born:
            if lborn:
                atoms.set_array("born_effective_charges", born)
            else:
                warnings.warn(
                    "{} don't have born effective charge while dataset require to have. Here we treat this data as not convergent".format(
                        filename
                    )
                )
                lconvergence = False

        # piezo related
        if lread_piezo:
            if lpiezo:
                atoms.info["piezoelectric"] = piezo_tensor
            else:
                warnings.warn(
                    "{} don't have piezoelectric while dataset require to have. Here we treat this data as not convergent".format(
                        filename
                    )
                )
                lconvergence = False

        # Spin related
        if lmagnetization_data:
            # Check lmagnetization of OUTCAR and dataset requirement
            if lmagnetization:
                # Generate ind_non_mag
                if len(non_mag_atomic_number) != 0:
                    atomic_numbers = atoms.get_atomic_numbers()
                    ind_non_mag = [
                        num in non_mag_atomic_number for num in atomic_numbers
                    ]
                else:
                    ind_non_mag = []

                # Remove the magmoms that ase read. Use get_outcar_magnetization_born then instead
                atoms._calc.results["magmoms"] = None
                atoms._calc.results["magmom"] = None

                # Set zero to non-magmom atoms
                mag_norm = np.linalg.norm(spin, axis=-1)
                mag_norm_save = mag_norm.copy()
                if len(ind_non_mag) != 0:
                    mag_norm_save[ind_non_mag] = 0
                    spin[ind_non_mag] = 0
                else:
                    mag_norm_save[mag_norm < min_mag_norm] = 0
                    spin[mag_norm < min_mag_norm] = 0

                if lnormalize_spin:
                    mag_index = mag_norm > 0.01
                    spin[mag_index] = spin[mag_index] / mag_norm[mag_index].reshape(
                        -1, 1
                    )
                atoms.set_array("spin_length", mag_norm_save)
                atoms.set_initial_magnetic_moments(spin)

                # Read OSZICAR to get spin forces
                if lread_OSZICAR:
                    read_OSZICAR_path = filename.replace("OUTCAR", "OSZICAR")
                    if os.path.exists(read_OSZICAR_path):
                        spin_force, mw = read_OSZICAR(
                            read_OSZICAR_path,
                            vasp6=vasp6,
                            on_missing="skip",
                        )  # in eV/uB
                        if spin_force is None:
                            warnings.warn(
                                f"{read_OSZICAR_path} has no complete "
                                "lambda*MW_perp/MW_int block; skipping spin-force labels"
                            )
                        else:
                            mw_norm = np.linalg.norm(mw, axis=-1)
                            spin_force = mw_norm.reshape(-1, 1) * spin_force  # in eV
                            warnings.warn(
                                "Here, spin_forces in unit of eV instead of eV/uB. Magnetic potential only predicts -dE/d(unit_vector_spin)=Spin_forces*|M|"
                            )
                            atoms.set_array("spin_forces_vert", spin_force)
            else:
                warnings.warn(
                    "{} don't have magnetization while dataset require to have. Here we treat this data as not convergent".format(
                        filename
                    )
                )
                lconvergence = False
    else:
        atoms = None

    return atoms, lconvergence


def bec_logout_convergence(file_path, criteria=1e-4):
    """
    Check the convergence of VASP calculations using pandas.

    Parameters
    ----------
    file_path : str
        The path to the VASP output file.
    criteria : float
        The convergence criteria.

    Returns
    -------
    bool
        True if converged, False otherwise.
    """
    if os.path.exists(file_path) is False:
        print(f"The file {file_path} does not exist.")
        return False
    df = pd.read_table(
        file_path, header=None, sep=r"\s+", names=range(20), on_bad_lines=None
    )
    linear_index = df[df[0].isin(["Linear"])]
    for i in range(1, len(linear_index) + 1):
        if i < 5:
            ind = linear_index.index[i] - 1
        elif i < len(linear_index):
            ind = linear_index.index[i] - 3
        else:
            ind = df.shape[0] - 4
        try:
            dE = float(df.iloc[ind, 3])
        except ValueError:
            print(
                f"Error: The value {df.iloc[ind, 3]} is not a float, see error in {file_path}"
            )
            return False
        # print(dE)
        if abs(dE) > criteria:
            return False
    return True
