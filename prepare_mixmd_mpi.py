import os
import sys
import argparse
import re
import subprocess
from mpi4py import MPI
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

def parse_arguments():
    description_text = (
        "Purpose: MPI-enabled script for preparing 3D probes for MixMD.\n"
        "Performs desalting, neutralization, mass filtering, pH protonation (OpenBabel),\n"
        "and 3D coordinate generation/minimization (RDKit ETKDGv3 + MMFF94)."
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
    
    parser.add_argument("-i", "--input", required=True, metavar="INPUT_SDF", 
                        help="Path to the input SDF file\n(Example: ./database.sdf)")
    parser.add_argument("-o", "--output", required=True, metavar="OUT_DIR", 
                        help="Directory for output files\n(Example: ./processed_probes)")
    parser.add_argument("-l", "--log_file", default="preparation.log", metavar="LOG_FILE",
                        help="Path to the output log file\n(Default: ./preparation.log)")
    
    parser.add_argument("--out_format", type=str, default="mol2", choices=["mol2", "pdb", "sdf"], metavar="FORMAT",
                        help="Format for output 3D structures\n(Default: mol2)")
    parser.add_argument("--min_mass", type=float, default=0.0, metavar="MASS", 
                        help="Minimum mass threshold in Da\n(Default: 0.0)")
    parser.add_argument("--max_mass", type=float, default=1500.0, metavar="MASS", 
                        help="Maximum mass threshold in Da\n(Default: 1500.0)")
    parser.add_argument("--ph", type=float, default=7.4, metavar="PH", 
                        help="Target pH for protonation\n(Default: 7.4)")
    parser.add_argument("--remove_salts", action="store_true", 
                        help="Enable removal of salts and counterions")
    parser.add_argument("--mmff_iters", type=int, default=1000, metavar="ITERS", 
                        help="Maximum MMFF94 geometry optimization iterations\n(Default: 1000)")
    parser.add_argument("--seed", type=int, default=42, metavar="SEED", 
                        help="Randomization seed for ETKDG 3D generation\n(Default: 42)")
    
    # Print short usage and error if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        print(f"{os.path.basename(__file__)}: error: the following arguments are required: -i/--input, -o/--output", file=sys.stderr)
        sys.exit(1)
        
    return parser.parse_args()

def process_molecules():
    args = parse_arguments()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    log_file_path = os.path.abspath(args.log_file)
    
    if rank == 0:
        os.makedirs(args.output, exist_ok=True)
        with open(log_file_path, "w") as f:
            f.write("=== Starting structure database processing (RDKit + OpenBabel Hybrid) ===\n")
            f.write(f"Parameters: pH={args.ph}, Mass={args.min_mass}-{args.max_mass}\n\n")
    comm.Barrier()
    
    def log_msg(msg):
        print(msg)
        with open(log_file_path, "a") as f:
            f.write(msg + "\n")
    
    suppl = Chem.SDMolSupplier(args.input, removeHs=True)
    remover = SaltRemover() if args.remove_salts else None
    uncharger = rdMolStandardize.Uncharger()
    
    local_sdf_blocks = []
    
    for idx, mol in enumerate(suppl):
        if idx % size != rank:
            continue
            
        if mol is None:
            continue
            
        # 1. Search for molecule name
        raw_name = ""
        for prop in ['_Name', 'name', 'Name', 'GENERIC_NAME', 'DRUG_NAME', 'SYNONYMS', 'ID']:
            if mol.HasProp(prop):
                val = mol.GetProp(prop).strip()
                if val:
                    raw_name = val
                    break
        
        if not raw_name:
            raw_name = "molecule"
            
        clean_name = raw_name.replace(' ', '_').lower()
        clean_name = re.sub(r'[^a-z0-9_]', '', clean_name)
        clean_name = f"{clean_name}_{idx}"
        
        # 2. Desalting
        if remover:
            mol = remover.StripMol(mol, dontRemoveEverything=True)
            
        # 3. Neutralization (Reset residual charges from database)
        try:
            mol = uncharger.uncharge(mol)
        except Exception:
            pass 
            
        mol.SetProp('_Name', clean_name)
        
        # 4. Mass filtration
        mass = Descriptors.MolWt(mol)
        if not (args.min_mass <= mass <= args.max_mass):
            continue 
            
        # 5. Protonation via OpenBabel
        tmp_2d_sdf = os.path.join(args.output, f"tmp_2d_{rank}_{idx}.sdf")
        tmp_prot_sdf = os.path.join(args.output, f"tmp_prot_{rank}_{idx}.sdf")
        
        writer = Chem.SDWriter(tmp_2d_sdf)
        writer.write(mol)
        writer.close()
        
        # Call obabel for pH calculation and ionization (-p)
        cmd_prot = ['obabel', '-isdf', tmp_2d_sdf, '-osdf', '-O', tmp_prot_sdf, '-p', str(args.ph)]
        subprocess.run(cmd_prot, capture_output=True)
        
        if not os.path.exists(tmp_prot_sdf):
            log_msg(f"[Rank {rank}] OpenBabel protonation error for {clean_name}")
            os.remove(tmp_2d_sdf)
            continue
            
        # 6. Return to RDKit for 3D generation (removeHs=False keeps added protonated atoms)
        prot_suppl = Chem.SDMolSupplier(tmp_prot_sdf, removeHs=False)
        if len(prot_suppl) == 0 or prot_suppl[0] is None:
            log_msg(f"[Rank {rank}] Error reading protonated structure {clean_name}")
            os.remove(tmp_2d_sdf); os.remove(tmp_prot_sdf)
            continue
            
        prot_mol = prot_suppl[0]
        
        # 3D space generation (ETKDGv3)
        params = AllChem.ETKDGv3()
        params.randomSeed = args.seed 
        if AllChem.EmbedMolecule(prot_mol, params) == -1:
            log_msg(f"[Rank {rank}] 3D generation error for {clean_name}")
            os.remove(tmp_2d_sdf); os.remove(tmp_prot_sdf)
            continue
            
        # Geometry minimization (MMFF94)
        try:
            AllChem.MMFFOptimizeMolecule(prot_mol, maxIters=args.mmff_iters)
        except Exception as e:
            log_msg(f"[Rank {rank}] Optimization error for {clean_name}: {e}")
            os.remove(tmp_2d_sdf); os.remove(tmp_prot_sdf)
            continue
            
        # 7. Export final molecule
        tmp_final_sdf = os.path.join(args.output, f"tmp_final_{rank}_{idx}.sdf")
        writer = Chem.SDWriter(tmp_final_sdf)
        writer.write(prot_mol)
        writer.close()
        
        output_file = os.path.join(args.output, f"{clean_name}.{args.out_format}")
        
        cmd_export = ['obabel', '-isdf', tmp_final_sdf, f'-o{args.out_format}', '-O', output_file, '--title', clean_name]
        result = subprocess.run(cmd_export, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_file):
            # Save SDF block for final checking
            with open(tmp_final_sdf, 'r') as tmp_f:
                local_sdf_blocks.append((idx, tmp_f.read()))
            log_msg(f"[Rank {rank}] Success: {clean_name} (Mass: {mass:.2f} Da)")
        else:
            log_msg(f"[Rank {rank}] Error writing {args.out_format} for {clean_name}:\n{result.stderr.strip()}")
            
        # Cleanup temporary files
        for tmp_file in [tmp_2d_sdf, tmp_prot_sdf, tmp_final_sdf]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    # Final result gathering
    comm.Barrier()
    gathered_blocks = comm.gather(local_sdf_blocks, root=0)
    
    if rank == 0:
        all_blocks = []
        for blocks in gathered_blocks:
            all_blocks.extend(blocks)
            
        all_blocks.sort(key=lambda x: x[0])
        
        combined_sdf_path = os.path.abspath("combined_probes.sdf")
        with open(combined_sdf_path, "w") as f_out:
            for _, block in all_blocks:
                f_out.write(block)
                
        log_msg(f"\n=== Successfully generated combined file for checking: {combined_sdf_path} ({len(all_blocks)} structures) ===")

if __name__ == "__main__":
    process_molecules()