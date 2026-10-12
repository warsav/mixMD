import os
import sys
import argparse
import subprocess
import shutil
import glob

def parse_arguments():
    description_text = (
        "Purpose: Script for sequential mass parametrization of ligands via acpype.\n"
        "Uses acpype's internal multithreading for quantum calculations."
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
    
    # Group 1: Script arguments (Directories and Resources)
    script_group = parser.add_argument_group('Script Settings (Directories & Resources)')
    script_group.add_argument("--in_dir", required=True, metavar="IN_DIR",
                              help="Path to the input directory with .mol2 files\n(Example: ./mixmd_probes)")
    script_group.add_argument("--out_dir", required=True, metavar="OUT_DIR",
                              help="Path to the output directory for results\n(Example: ./acpype_results)")
    script_group.add_argument("--threads", type=int, default=None, metavar="THREADS",
                              help="Limit the number of OpenMP threads for quantum calculations\n(Default: all available cores)")
    script_group.add_argument("--resume", action="store_true",
                              help="Resume mode: skip ligands that already have parameter files,\nsuccessful logs, or were marked as failed")

    # Group 2: Original acpype arguments
    acpype_group = parser.add_argument_group('Original acpype parameters')
    acpype_group.add_argument("-c", "--charge_method", type=str, default="bcc", metavar="METHOD",
                              choices=["bcc", "gasteiger", "user"],
                              help="Charge calculation method [acpype: -c]\n(Default: bcc)")
    acpype_group.add_argument("-a", "--atom_type", type=str, default="gaff2", metavar="TYPE",
                              choices=["gaff", "gaff2", "amber"],
                              help="AMBER force field type [acpype: -a]\n(Default: gaff2)")
    acpype_group.add_argument("-n", "--net_charge", type=int, default=None, metavar="CHARGE",
                              help="Net charge of the molecule. If not specified, acpype auto-detects [acpype: -n]")
    acpype_group.add_argument("-m", "--multiplicity", type=int, default=1, metavar="MULT",
                              help="Multiplicity (2S+1) [acpype: -m]\n(Default: 1)")
    acpype_group.add_argument("-o", "--outtop", type=str, default="gmx", metavar="FORMAT",
                              choices=["all", "gmx", "cns", "charmm"],
                              help="Output topology formats [acpype: -o]\n(Default: gmx)")
    acpype_group.add_argument("-q", "--qprog", type=str, default="sqm", metavar="PROG",
                              choices=["sqm", "mopac"],
                              help="Quantum chemical program [acpype: -q]\n(Default: sqm)")
    acpype_group.add_argument("-k", "--keyword", type=str, default=None, metavar="KEYWORD",
                              help="Additional keywords for SQM/MOPAC [acpype: -k]")
    acpype_group.add_argument("-f", "--force", action="store_true",
                              help="Force overwrite existing files [acpype: -f]")
    acpype_group.add_argument("-t", "--tolerant", action="store_true",
                              help="Tolerant mode (ignore some non-critical errors/checks) [acpype: -t]")
    
    # Print short usage and error if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        print(f"{os.path.basename(__file__)}: error: the following arguments are required: --in_dir, --out_dir", file=sys.stderr)
        sys.exit(1)
        
    return parser.parse_args()

def process_parametrization():
    args = parse_arguments()
    
    input_dir = os.path.abspath(args.in_dir)
    output_dir = os.path.abspath(args.out_dir)
    failed_dir = os.path.join(output_dir, "failed_ligands")
    logs_dir = os.path.join(output_dir, "logs")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    all_mol2_files = sorted(glob.glob(os.path.join(input_dir, "*.mol2")))
    total_files = len(all_mol2_files)
    
    print(f"=== Starting sequential acpype parametrization ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Molecules found for processing: {total_files}")
    
    # Create a custom environment to limit threads
    custom_env = os.environ.copy()
    if args.threads is not None:
        custom_env["OMP_NUM_THREADS"] = str(args.threads)
        custom_env["MKL_NUM_THREADS"] = str(args.threads)
        custom_env["OPENBLAS_NUM_THREADS"] = str(args.threads)
        print(f"Thread limit (OpenMP): {args.threads}\n")
    else:
        print("Thread limit: Not set (using all available cores)\n")
        
    if args.resume:
        print("INFO: Resume mode (--resume) enabled. Already processed structures will be skipped.\n")
    
    if total_files == 0:
        print(f"Error: No .mol2 files found in {input_dir}.")
        sys.exit(1)
        
    total_success = []
    total_failed = []
    
    for idx, mol2_path in enumerate(all_mol2_files, 1):
        basename = os.path.splitext(os.path.basename(mol2_path))[0]
        log_file_path = os.path.join(logs_dir, f"{basename}_acpype.log")
        top_file = os.path.join(output_dir, f"{basename}.acpype", f"{basename}_GMX.top")
        
        # RESUME LOGIC
        if args.resume:
            # 1. Check if the molecule was already marked as failed
            failed_mol2_path = os.path.join(failed_dir, f"{basename}.mol2")
            if os.path.exists(failed_mol2_path):
                total_failed.append(basename)
                print(f"[{idx}/{total_files}] Skipping {basename} (previously failed).")
                continue
                
            # 2. Check for successful parametrization
            itp_file = os.path.join(output_dir, f"{basename}.acpype", f"{basename}_GMX.itp")
            if os.path.exists(top_file) and os.path.exists(itp_file) and os.path.exists(log_file_path):
                with open(log_file_path, "r", encoding="utf-8", errors="ignore") as log_f:
                    log_content = log_f.read()
                    if "Total time of execution" in log_content or "Writing" in log_content:
                        total_success.append(basename)
                        print(f"[{idx}/{total_files}] Skipping {basename} (already parametrized successfully).")
                        continue
        
        # Dynamically build the command
        cmd = ['acpype', '-i', mol2_path, '-b', basename, '-d']
        
        if args.charge_method:
            cmd.extend(['-c', args.charge_method])
        if args.atom_type:
            cmd.extend(['-a', args.atom_type])
        if args.net_charge is not None:
            cmd.extend(['-n', str(args.net_charge)])
        if args.multiplicity is not None:
            cmd.extend(['-m', str(args.multiplicity)])
        if args.outtop:
            cmd.extend(['-o', args.outtop])
        if args.qprog:
            cmd.extend(['-q', args.qprog])
        if args.keyword:
            cmd.extend(['-k', args.keyword])
        if args.force:
            cmd.append('-f')
        if args.tolerant:
            cmd.append('-t')
            
        print(f"[{idx}/{total_files}] Processing {basename}...", end=" ", flush=True)
            
        try:
            # Run with the custom environment
            result = subprocess.run(
                cmd, 
                cwd=output_dir, 
                capture_output=True, 
                text=True,
                env=custom_env
            )
            
            # Write log
            with open(log_file_path, "w") as log_f:
                log_f.write(f"Command: {' '.join(cmd)}\n")
                log_f.write("=== STDOUT ===\n")
                log_f.write(result.stdout)
                log_f.write("\n=== STDERR ===\n")
                log_f.write(result.stderr)
            
            # Check for success
            if result.returncode == 0 and os.path.exists(top_file):
                total_success.append(basename)
                print("Success.")
                
                # Clean up heavy temporary SQM/MOPAC files
                sqm_files = glob.glob(os.path.join(output_dir, f"{basename}.acpype", "*.sqm*"))
                mopac_files = glob.glob(os.path.join(output_dir, f"{basename}.acpype", "*.mop*"))
                for tmp_f in sqm_files + mopac_files:
                    os.remove(tmp_f)
            else:
                raise Exception("GROMACS topology files were not generated.")
                
        except Exception as e:
            total_failed.append(basename)
            print(f"FAILED! (Details in {log_file_path})")
            # Isolate failed structures
            shutil.copy(mol2_path, os.path.join(failed_dir, f"{basename}.mol2"))

    # Summary
    print("\n" + "="*50)
    print("ACPYPE PARAMETRIZATION SUMMARY:")
    print(f"Total processed: {total_files}")
    print(f"Successfully generated/skipped: {len(total_success)}")
    print(f"Errors (moved to failed_ligands/): {len(total_failed)}")
    print("="*50)
    
    if len(total_failed) > 0:
        print("\nList of structures that failed:")
        for f in total_failed:
            print(f" - {f}")

if __name__ == "__main__":
    process_parametrization()