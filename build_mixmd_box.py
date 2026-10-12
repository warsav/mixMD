import os
import sys
import glob
import math
import argparse
import subprocess
import shutil
import re

def parse_args():
    description_text = (
        "Purpose: Script to calculate required box volume for MixMD based on GROMACS vdwradii.dat\n"
        "and iteratively insert ligands using gmx insert-molecules."
    )
    
    epilog_text = (
        "Author: Dr. Andrii Zaremba (ORCID: 0000-0003-0186-1235)\n"
        "Acknowledgment: Code implementation assisted by Gemini AI."
    )
    
    parser = argparse.ArgumentParser(
        description=description_text,
        epilog=epilog_text,
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Required Arguments
    req_group = parser.add_argument_group('Required Arguments')
    req_group.add_argument("--acpype_dir", required=True, metavar="ACPYPE_DIR", 
                           help="Input dir containing acpype subfolders with *_GMX.gro\n(Example: ./acpype_results)")
    req_group.add_argument("--clean_top_dir", required=True, metavar="TOP_DIR", 
                           help="Dir containing clean_topologies/ and merged_atomtypes/ subfolders\n(Example: ./clean_topologies)")
    req_group.add_argument("--top", required=True, metavar="TOP_FILE", 
                           help="Main system .top file (protein topology)\n(Example: ./topol.top)")
    req_group.add_argument("--protein", required=True, metavar="GRO_FILE", 
                           help="Protein .gro file\n(Example: ./protein.gro)")
    req_group.add_argument("--out_dir", required=True, metavar="OUT_DIR", 
                           help="Output directory for the assembled system\n(Example: ./mixmd_box)")
    
    # Optional Output/Logging
    opt_group = parser.add_argument_group('Optional Settings')
    opt_group.add_argument("-l", "--log_file", default="build_mixmd_box.log", metavar="LOG_FILE",
                           help="Path to the output log file\n(Default: ./build_mixmd_box.log)")
    
    # System Physics/Geometry Parameters
    opt_group.add_argument("--v_v_ratio", type=float, default=5.0, metavar="RATIO",
                           help="Volume percentage of probes vs total solvent\n(Default: 5.0)")
    opt_group.add_argument("--copies", type=int, default=1, metavar="NUM",
                           help="Number of copies per ligand to insert\n(Default: 1)")
    opt_group.add_argument("--box_type", type=str, default="octahedron", metavar="TYPE",
                           choices=['cubic', 'triclinic', 'octahedron', 'dodecahedron'], 
                           help="Box type for editconf\n(Default: octahedron)")
    opt_group.add_argument("--editconf_args", type=str, default="-c", metavar="ARGS",
                           help="Extra arguments for editconf\n(Default: '-c' for centering)")
    
    # GROMACS execution settings
    gmx_group = parser.add_argument_group('GROMACS Execution Options')
    gmx_group.add_argument("--try_limit", type=int, default=100, metavar="LIMIT",
                           help="Max insertion attempts per molecule\n(Default: 100)")
    gmx_group.add_argument("--gmx_path", type=str, default="gmx", metavar="PATH",
                           help="Path to GROMACS executable\n(Default: gmx)")
    gmx_group.add_argument("--vdwradii", type=str, default=None, metavar="FILE",
                           help="Path to vdwradii.dat\n(Default: auto-detected from gmx_path)")
    
    # Print short usage and error if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        print(f"{os.path.basename(__file__)}: error: the following arguments are required: --acpype_dir, --clean_top_dir, --top, --protein, --out_dir", file=sys.stderr)
        sys.exit(1)
        
    return parser.parse_args()

def find_vdwradii(gmx_path):
    try:
        gmx_bin = subprocess.check_output(['which', gmx_path]).decode().strip()
        base_dir = os.path.dirname(os.path.dirname(gmx_bin))
        potential_path = os.path.join(base_dir, 'share', 'gromacs', 'top', 'vdwradii.dat')
        if os.path.exists(potential_path):
            return potential_path
    except Exception:
        pass
    return None

def load_vdw_radii(filepath):
    radii = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                resname = parts[0]
                atomname = parts[1]
                try:
                    radius = float(parts[2])
                    radii[(resname, atomname)] = radius
                except ValueError:
                    pass
    return radii

def get_atom_radius(radii_dict, resname, atomname):
    for length in range(len(atomname), 0, -1):
        sub_name = atomname[:length]
        if (resname, sub_name) in radii_dict:
            return radii_dict[(resname, sub_name)]
        if ('AAA', sub_name) in radii_dict:
            return radii_dict[('AAA', sub_name)]
        if ('???', sub_name) in radii_dict:
            return radii_dict[('???', sub_name)]
        if ('*', sub_name) in radii_dict:
            return radii_dict[('*', sub_name)]
    return 0.15

def calculate_gro_volume(gro_path, radii_dict):
    vol = 0.0
    with open(gro_path, 'r') as f:
        lines = f.readlines()
        if len(lines) < 3:
            return 0.0
        for line in lines[2:-1]:
            resname = line[5:10].strip()
            atomname = line[10:15].strip()
            r = get_atom_radius(radii_dict, resname, atomname)
            vol += (4.0 / 3.0) * math.pi * (r ** 3)
    return vol

def main():
    args = parse_args()
    
    log_file = os.path.abspath(args.log_file)
    with open(log_file, "w") as f:
        f.write("=== MixMD Box Assembly Log ===\n")
        
    def log_msg(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
            
    if not os.path.isdir(args.acpype_dir):
        log_msg(f"Error: Input directory '{args.acpype_dir}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(args.clean_top_dir):
        log_msg(f"Error: Topologies directory '{args.clean_top_dir}' does not exist.")
        sys.exit(1)
    if not os.path.isfile(args.top):
        log_msg(f"Error: Topology file '{args.top}' does not exist.")
        sys.exit(1)
    if not os.path.isfile(args.protein):
        log_msg(f"Error: Protein file '{args.protein}' does not exist.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    
    log_msg("=== Phase 1: Analyzing Topologies and Volumes ===")
    
    vdw_path = args.vdwradii
    if not vdw_path:
        vdw_path = find_vdwradii(args.gmx_path)
        
    if not vdw_path or not os.path.exists(vdw_path):
        log_msg("Error: Could not locate vdwradii.dat. Please provide it manually via --vdwradii")
        sys.exit(1)
        
    log_msg(f"Using GROMACS VdW radii library: {vdw_path}")
    radii_dict = load_vdw_radii(vdw_path)
    
    search_pattern = os.path.join(args.acpype_dir, "*", "*_GMX.gro")
    raw_ligand_gro_files = sorted(glob.glob(search_pattern))
    if not raw_ligand_gro_files:
        search_pattern_recursive = os.path.join(args.acpype_dir, "**", "*_GMX.gro")
        raw_ligand_gro_files = sorted(glob.glob(search_pattern_recursive, recursive=True))
        
    if not raw_ligand_gro_files:
        log_msg(f"Error: No *_GMX.gro files found in {args.acpype_dir} or its subdirectories.")
        sys.exit(1)
        
    clean_topologies_dir = os.path.join(args.clean_top_dir, "clean_topologies")
    merged_atomtypes_file = os.path.join(args.clean_top_dir, "merged_atomtypes", "merged_atomtypes.itp")
    
    if not os.path.exists(merged_atomtypes_file):
        log_msg(f"Error: Merged atomtypes file not found at {merged_atomtypes_file}")
        sys.exit(1)

    ligand_gro_files = []
    for gro_path in raw_ligand_gro_files:
        basename = os.path.basename(gro_path)
        clean_itp_name = basename.replace("_GMX.gro", "_GMX_clean.itp")
        clean_itp_path = os.path.join(clean_topologies_dir, clean_itp_name)
        
        if os.path.exists(clean_itp_path):
            ligand_gro_files.append(gro_path)
        else:
            log_msg(f"WARNING: Skipping {basename} because its topology ({clean_itp_name}) is missing in {clean_topologies_dir}")

    if not ligand_gro_files:
        log_msg("Error: No valid ligands with matching clean topologies found.")
        sys.exit(1)

    log_msg(f"Detected {len(ligand_gro_files)} valid unique ligand types ready for insertion.")
    
    protein_vol = calculate_gro_volume(args.protein, radii_dict)
    log_msg(f"Protein VdW volume: {protein_vol:.4f} nm^3")
    
    single_set_ligands_vol = sum([calculate_gro_volume(gro, radii_dict) for gro in ligand_gro_files])
    total_ligands_vol = single_set_ligands_vol * args.copies
    log_msg(f"Total Ligands VdW volume ({args.copies} copies each): {total_ligands_vol:.4f} nm^3")
    
    if args.v_v_ratio <= 0 or args.v_v_ratio >= 100:
        log_msg("Error: --v_v_ratio must be strictly between 0 and 100.")
        sys.exit(1)
        
    water_vol = total_ligands_vol * ((100.0 - args.v_v_ratio) / args.v_v_ratio)
    log_msg(f"Target Water VdW volume (for {args.v_v_ratio}% v/v ratio): {water_vol:.4f} nm^3")
    
    total_system_vol = protein_vol + total_ligands_vol + water_vol
    log_msg(f"Target Total System Volume: {total_system_vol:.4f} nm^3")
    
    if args.box_type in ['cubic', 'triclinic']:
        d = total_system_vol ** (1.0/3.0)
    elif args.box_type in ['octahedron', 'dodecahedron']:
        d = (total_system_vol / 0.70710678) ** (1.0/3.0)
        
    log_msg(f"Calculated optimal box dimension 'd' for {args.box_type}: {d:.4f} nm\n")
    
    log_msg("=== Phase 2: Updating Topology (.top) ===")
    
    ligand_mol_names = []
    clean_itp_includes = []
    
    for gro_path in ligand_gro_files:
        basename = os.path.basename(gro_path)
        base_prefix = basename.replace("_GMX.gro", "")
        clean_itp_name = basename.replace("_GMX.gro", "_GMX_clean.itp")
        clean_itp_path = os.path.join(clean_topologies_dir, clean_itp_name)
        
        clean_itp_includes.append(clean_itp_path)
        
        mol_name = base_prefix
        with open(clean_itp_path, 'r') as f:
            lines = f.readlines()
            found_moltype = False
            for line in lines:
                stripped = line.strip()
                if not found_moltype:
                    if stripped.startswith('[ moleculetype ]'):
                        found_moltype = True
                else:
                    # Ignore empty lines and comments to get the real name
                    if stripped and not stripped.startswith(';'):
                        mol_name = stripped.split()[0]
                        break
        ligand_mol_names.append(mol_name)

    new_top_path = os.path.join(args.out_dir, "system_mixmd.top")
    
    with open(args.top, 'r') as f:
        top_lines = f.readlines()
        
    ff_include_idx = -1
    system_idx = -1
    
    for i, line in enumerate(top_lines):
        if ff_include_idx == -1 and line.startswith('#include') and re.search(r'\.ff|forcefield\.itp', line):
            ff_include_idx = i
        if re.match(r'^\s*\[\s*system\s*\]', line.strip()):
            system_idx = i
            
    if ff_include_idx == -1 or system_idx == -1:
        log_msg("Error: Could not parse main .top file structure (missing forcefield include or [ system ] block).")
        sys.exit(1)
        
    with open(new_top_path, 'w') as f:
        for line in top_lines[:ff_include_idx + 1]:
            f.write(line)
            
        f.write(f'\n; MixMD Merged Atomtypes\n#include "{merged_atomtypes_file}"\n')
        
        for line in top_lines[ff_include_idx + 1:system_idx]:
            f.write(line)
            
        f.write('\n; MixMD Cleaned Ligand Topologies\n')
        for itp in clean_itp_includes:
            f.write(f'#include "{itp}"\n')
        f.write('\n')
        
        for line in top_lines[system_idx:]:
            f.write(line)
            
        f.write('\n; MixMD Ligands\n')
        for mol_name in ligand_mol_names:
            f.write(f"{mol_name:<20} {args.copies}\n")
            
    log_msg(f"Successfully generated new topology: {new_top_path}\n")

    log_msg("=== Phase 3: Spatial Box Assembly ===")
    
    base_box = os.path.join(args.out_dir, "box_base.gro")
    cmd_editconf = [args.gmx_path, "editconf", "-f", args.protein, "-o", base_box, 
                    "-box", str(d), str(d), str(d), "-bt", args.box_type]
    
    if args.editconf_args:
        cmd_editconf.extend(args.editconf_args.split())
        
    log_msg(f"Running: {' '.join(cmd_editconf)}")
    result = subprocess.run(cmd_editconf, capture_output=True, text=True)
    if result.returncode != 0:
        log_msg(f"Error in editconf:\n{result.stderr}")
        sys.exit(1)

    current_box = base_box
    
    log_msg(f"Starting sequential insertion of {len(ligand_gro_files)} ligands ({args.copies} copies each)...")
    for i, gro_path in enumerate(ligand_gro_files, 1):
        next_box = os.path.join(args.out_dir, f"box_iter_{i}.gro")
        
        cmd_insert = [
            args.gmx_path, "insert-molecules", 
            "-f", current_box, 
            "-ci", gro_path, 
            "-nmol", str(args.copies), 
            "-o", next_box, 
            "-try", str(args.try_limit)
        ]
        
        msg_start = f"[{i}/{len(ligand_gro_files)}] Inserting {os.path.basename(gro_path)}..."
        print(msg_start, end=" ", flush=True)
        
        result = subprocess.run(cmd_insert, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(next_box):
            print("Done.")
            with open(log_file, "a") as f:
                f.write(msg_start + " Done.\n")
            if current_box != base_box:
                os.remove(current_box)
            current_box = next_box
        else:
            print("\nWARNING: Insertion failed or partial. Box might be too dense!")
            with open(log_file, "a") as f:
                f.write(msg_start + " WARNING: Insertion failed or partial.\n")
                f.write("GROMACS Error details:\n" + result.stderr + "\n")
            if current_box != base_box:
                shutil.copy(current_box, next_box)
            current_box = next_box
            
    final_box = os.path.join(args.out_dir, "box_mixmd_ready.gro")
    shutil.move(current_box, final_box)
    
    final_summary = (
        "\n" + "="*50 + "\n" +
        "ASSEMBLY SUMMARY:\n" +
        f"System Topology: {new_top_path}\n" +
        f"Assembled Box: {final_box}\n" +
        f"Log file: {log_file}\n" +
        "The system is now ready for manual solvation (gmx solvate).\n" +
        "="*50
    )
    log_msg(final_summary)

if __name__ == "__main__":
    main()