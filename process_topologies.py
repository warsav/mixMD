import os
import sys
import glob
import argparse
import datetime
import re
from collections import defaultdict

def parse_args():
    description_text = (
        "Purpose: Script to merge GROMACS system parameters from multiple acpype .itp files\n"
        "and clean ligand topologies for mixMD systems."
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
    
    parser.add_argument("-i", "--input_dir", required=True, metavar="INPUT_DIR",
                        help="Input directory containing acpype *_GMX.itp files\n(Example: ./acpype_results)")
    parser.add_argument("-o", "--output_dir", required=True, metavar="OUTPUT_DIR",
                        help="Output directory for cleaned .itp files and merged atomtypes\n(Example: ./clean_topologies)")
    parser.add_argument("-l", "--log_file", default="topology_merge.log", metavar="LOG_FILE",
                        help="Path to the output log file\n(Default: ./topology_merge.log)")
    
    # Print short usage and error if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        print(f"{os.path.basename(__file__)}: error: the following arguments are required: -i/--input_dir, -o/--output_dir", file=sys.stderr)
        sys.exit(1)
        
    return parser.parse_args()

def get_param_key(section, normalized_line):
    key_lengths = {
        "atomtypes": 1,
        "bondtypes": 2,
        "angletypes": 3,
        "dihedraltypes": 4,
        "pairtypes": 2,
        "constrainttypes": 2,
        "cmaptypes": 5
    }
    parts = normalized_line.split()
    length = key_lengths.get(section, 1)
    if len(parts) < length:
        return normalized_line
    return " ".join(parts[:length])

def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    clean_topologies_dir = os.path.join(output_dir, "clean_topologies")
    merged_atomtypes_dir = os.path.join(output_dir, "merged_atomtypes")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(clean_topologies_dir, exist_ok=True)
    os.makedirs(merged_atomtypes_dir, exist_ok=True)
    
    log_path = os.path.abspath(args.log_file)
    
    itp_files = sorted(glob.glob(os.path.join(input_dir, "**/*_GMX.itp"), recursive=True))
    if not itp_files:
        print(f"Error: No *_GMX.itp files found in {input_dir}")
        sys.exit(1)
        
    print(f"=== Starting Topology Processing ===")
    print(f"Found {len(itp_files)} .itp files.")
    
    SYSTEM_SECTIONS = {"atomtypes", "bondtypes", "angletypes", "dihedraltypes", "pairtypes", "constrainttypes", "cmaptypes"}
    # Added sections that absolutely must not be included in clean.itp
    DISCARD_SECTIONS = {"defaults", "system", "molecules"}
    
    system_tracker = defaultdict(lambda: defaultdict(set))
    system_sources = defaultdict(lambda: defaultdict(list))
    moleculetypes_tracker = defaultdict(list)
    
    corrupted_files = []
    clean_itp_paths = []
    processed_files_log = []
    
    for itp_path in itp_files:
        basename = os.path.basename(itp_path)
        processed_files_log.append(basename)
        clean_itp_name = basename.replace(".itp", "_clean.itp")
        clean_itp_path = os.path.join(clean_topologies_dir, clean_itp_name)
        
        with open(itp_path, 'r') as f:
            lines = f.readlines()
            
        current_section = None
        clean_lines = []
        mol_name = None
        has_moleculetype = False
        
        for line in lines:
            stripped = line.strip()
            
            # Use regex for accurate parsing of the section name
            match = re.match(r'\[\s*(\w+)\s*\]', stripped)
            if match:
                current_section = match.group(1).lower()
                if current_section not in SYSTEM_SECTIONS and current_section not in DISCARD_SECTIONS:
                    clean_lines.append(line)
                continue
                
            if current_section in DISCARD_SECTIONS:
                continue
                
            if current_section in SYSTEM_SECTIONS:
                if stripped and not stripped.startswith(';'):
                    normalized_line = " ".join(stripped.split())
                    param_key = get_param_key(current_section, normalized_line)
                    
                    system_tracker[current_section][param_key].add(normalized_line)
                    system_sources[current_section][normalized_line].append(basename)
                    
            elif current_section == "moleculetype":
                clean_lines.append(line)
                if stripped and not stripped.startswith(';') and mol_name is None:
                    mol_name = stripped.split()[0]
                    moleculetypes_tracker[mol_name].append(basename)
                    has_moleculetype = True
                    
            else:
                clean_lines.append(line)
                
        if not has_moleculetype:
            corrupted_files.append(basename)
            
        with open(clean_itp_path, 'w') as f:
            f.writelines(clean_lines)
        clean_itp_paths.append(clean_itp_name)

    print("Analyzing data for conflicts...")
    
    conflicts_found = 0
    ordered_sections = ["atomtypes", "bondtypes", "constrainttypes", "angletypes", "dihedraltypes", "pairtypes", "cmaptypes"]
    
    with open(log_path, 'w') as log:
        log.write("="*50 + "\n")
        log.write("=== GROMACS TOPOLOGY MERGE LOG ===\n")
        log.write("="*50 + "\n")
        log.write(f"Execution Time : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Input Directory: {input_dir}\n")
        log.write(f"Output Directory: {output_dir}\n")
        log.write(f"Total .itp Files: {len(itp_files)}\n\n")
        
        log.write("--- PROCESSED FILES ---\n")
        for f in processed_files_log:
            log.write(f" - {f}\n")
        log.write("\n")

        log.write("--- EXTRACTED SYSTEM PARAMETERS SUMMARY ---\n")
        for sec in ordered_sections:
            if sec in system_tracker:
                log.write(f"  {sec:<15} : {len(system_tracker[sec])} unique entries\n")
        log.write("\n")
        
        log.write("--- 1. SYSTEM PARAMETER CONFLICTS ---\n")
        param_conflicts = 0
        for sec in ordered_sections:
            if sec not in system_tracker: continue
            for param_key, param_sets in system_tracker[sec].items():
                if len(param_sets) > 1:
                    param_conflicts += 1
                    conflicts_found += 1
                    log.write(f"CONFLICT DETECTED: [ {sec} ] for '{param_key}' has {len(param_sets)} different definitions.\n")
                    for param_str in param_sets:
                        sources = ", ".join(system_sources[sec][param_str])
                        log.write(f"  Definition: {param_str}\n")
                        log.write(f"  Found in: {sources}\n\n")
                        
        if param_conflicts == 0:
            log.write("OK: No system parameter conflicts detected. Parameters are unique.\n\n")
            
        log.write("--- 2. MOLECULE NAME CONFLICTS ---\n")
        mol_conflicts = 0
        for mol_name, sources in moleculetypes_tracker.items():
            if len(sources) > 1:
                mol_conflicts += 1
                conflicts_found += 1
                log.write(f"CONFLICT DETECTED: Molecule name '{mol_name}' is declared multiple times.\n")
                log.write(f"  Found in: {', '.join(sources)}\n\n")
        if mol_conflicts == 0:
            log.write("OK: All molecule names are unique.\n\n")
            
        log.write("--- 3. CORRUPTED OR INCOMPLETE FILES ---\n")
        if len(corrupted_files) > 0:
            conflicts_found += len(corrupted_files)
            for f in corrupted_files:
                log.write(f"ERROR: File '{f}' is missing the [ moleculetype ] section.\n")
            log.write("\n")
        else:
            log.write("OK: All files contain valid [ moleculetype ] sections.\n\n")
            
        log.write("="*50 + "\n")
        log.write(f"=== END OF LOG | TOTAL CONFLICTS/ERRORS: {conflicts_found} ===\n")
        log.write("="*50 + "\n")

    merged_atomtypes_path = os.path.join(merged_atomtypes_dir, "merged_atomtypes.itp")
    with open(merged_atomtypes_path, 'w') as f:
        f.write("; Merged system parameters generated automatically\n")
        f.write("; Include this file at the top of your main system.top (after [ defaults ])\n")
        
        for sec in ordered_sections:
            if sec in system_tracker and system_tracker[sec]:
                f.write(f"\n[ {sec} ]\n")
                for param_key in sorted(system_tracker[sec].keys()):
                    for param_str in system_tracker[sec][param_key]:
                        f.write(f"  {param_str}\n")

    print("\n" + "="*50)
    print("PROCESSING SUMMARY:")
    print(f"Cleaned topologies saved to: {clean_topologies_dir}")
    print(f"Merged system parameters saved to: {merged_atomtypes_path}")
    print(f"Detailed log file saved to: {log_path}")
    if conflicts_found > 0:
        print(f"\nWARNING: {conflicts_found} conflict(s) or error(s) detected!")
        print(f"Please review {log_path} carefully before proceeding.")
    else:
        print("\nSUCCESS: All parameters match perfectly. No conflicts detected.")
    print("="*50)

if __name__ == "__main__":
    main()