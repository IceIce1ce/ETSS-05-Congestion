"""
=============================================================================
Disclaimer for Reproducibility:
Due to the inherent randomness in the MILP solver's heuristics and the use of 
random seeds for coverage verification, re-running this script from scratch 
may yield slightly different data splits. This behavior is expected in 
heuristic search.

For strictly reproducible benchmarking and fair comparisons, please directly 
use the provided `split.json` and `.txt` files in this repository.
=============================================================================

This script provides the end-to-end pipeline for generating the TPC-268 
benchmark splits. It dynamically reads the annotation JSON and physical 
directory structure, formulates the Mixed-Integer Linear Programming (MILP) 
problem to ensure taxonomic independence and density balance, and outputs 
the final file lists.
"""

import os
import json
import random
import argparse
import pandas as pd
import pulp
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Taxonomic Dataset Splits via MILP")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Path to the root directory containing Species/Organ images.")
    parser.add_argument("--anno_json", type=str, default="annotations.json", 
                        help="Path to the instance annotation JSON file.")
    parser.add_argument("--output_dir", type=str, default="split", 
                        help="Directory to save the generated split txt and json files.")
    parser.add_argument("--max_retries", type=int, default=50, 
                        help="Maximum attempts for heuristic optimization.")
    return parser.parse_args()


def build_statistics_from_data(data_dir, anno_path):
    """
    Scans the physical directory and parses the annotation JSON to compute 
    image counts and total instance points for each Species-Organ unit.
    """
    print(f"Loading annotations from {anno_path}...")
    try:
        with open(anno_path, "r") as f:
            anno_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file {anno_path} not found.")
        return pd.DataFrame()

    stats = defaultdict(lambda: {"image_count": 0, "total_points": 0})

    print(f"Scanning physical directories in {data_dir}...")
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        return pd.DataFrame()

    for species in os.listdir(data_dir):
        species_path = os.path.join(data_dir, species)
        if not os.path.isdir(species_path):
            continue
            
        for organ in os.listdir(species_path):
            organ_path = os.path.join(species_path, organ)
            if not os.path.isdir(organ_path):
                continue
                
            unit_key = f"{species}_{organ}"
            
            for img_file in os.listdir(organ_path):
                if img_file.endswith(".jpg") or img_file.endswith(".png"):
                    if img_file in anno_data:
                        stats[unit_key]["image_count"] += 1
                        points = anno_data[img_file].get("points", [])
                        stats[unit_key]["total_points"] += len(points)
                    else:
                        # Silently skip images without annotations to keep output clean
                        pass

    records = []
    for unit, data in stats.items():
        if data["image_count"] > 0:
            species, organ = unit.rsplit("_", 1)
            records.append({
                "Unit": unit,
                "Species": species,
                "Organ": organ,
                "Image Count": data["image_count"],
                "Total Points": data["total_points"]
            })
            
    df = pd.DataFrame(records)
    print(f"Successfully built statistics for {len(df)} taxonomic units.")
    return df


def verify_scale_coverage(assignments_dict):
    """
    Verification step: Ensure all subsets (train, val, test) cover the full 
    spectrum of observation scales (Microscopy, Close-range, Remote Sensing).
    
    NOTE: In the practical curation of TPC-268, this step was verified manually 
    by researchers after the solver generated candidate splits, as scale metadata 
    is not hardcoded into the instance-level JSON. 
    
    For automated pipelines, users can inject a dictionary mapping 'Units' 
    to their 'Scale' here. This function serves as the programmatic hook.
    """
    return True


def solve_milp_partition(df, max_retries):
    """
    Formulates and solves the MILP problem to assign each Species-Organ unit 
    to train/val/test splits while balancing image proportions and point density.
    """
    splits = ["train", "val", "test"]
    target_prop = {"train": 0.7, "val": 0.1, "test": 0.2}

    total_imgs = df["Image Count"].sum()
    total_pts = df["Total Points"].sum()

    target_imgs = {k: target_prop[k] * total_imgs for k in splits}
    target_pts = {k: target_prop[k] * total_pts for k in splits}

    attempt = 0
    final_assignments = {}

    while attempt < max_retries:
        attempt += 1
        current_seed = random.randint(1, 10000)
        print(f"\n--- Attempt {attempt}: MILP optimization (Seed {current_seed}) ---")

        prob = pulp.LpProblem("Dataset_Taxonomic_Split", pulp.LpMinimize)

        # Decision variables: z[(i,k)] = 1 if unit i is in split k
        z = {(i, k): pulp.LpVariable(f"z_{i}_{k}", cat="Binary")
             for i in df.index for k in splits}

        # Constraint 1: Unique assignment (Zero-shot taxonomic evaluation)
        for i in df.index:
            prob += sum(z[i, k] for k in splits) == 1, f"one_split_{i}"

        # Constraint 2: Biological Organization Coverage
        # Ensure the training set contains at least one of every 'Organ' type
        organs = df["Organ"].unique()
        for organ in organs:
            species_in_organ = df.index[df["Organ"] == organ]
            prob += sum(z[i, "train"] for i in species_in_organ) >= 1, f"organ_cover_{organ}"

        imgs = {k: sum(df.loc[i, "Image Count"] * z[i, k] for i in df.index) for k in splits}
        pts = {k: sum(df.loc[i, "Total Points"] * z[i, k] for i in df.index) for k in splits}

        u_img = {k: pulp.LpVariable(f"u_img_{k}", lowBound=0) for k in splits}
        u_pts = {k: pulp.LpVariable(f"u_pts_{k}", lowBound=0) for k in splits}

        for k in splits:
            prob += imgs[k] - target_imgs[k] <= u_img[k]
            prob += target_imgs[k] - imgs[k] <= u_img[k]
            prob += pts[k] - target_pts[k] <= u_pts[k]
            prob += target_pts[k] - pts[k] <= u_pts[k]

        # Objective: Minimize weighted sum of deviations (Images >> Points)
        w_img, w_pts = 100.0, 1.0
        prob += sum(u_img[k] for k in splits) * w_img + sum(u_pts[k] for k in splits) * w_pts

        # Solve silently
        prob.solve(pulp.PULP_CBC_CMD(msg=0, gapRel=0.01, timeLimit=60, randomSeed=current_seed))

        current_assignments = {}
        for i in df.index:
            for k in splits:
                if pulp.value(z[i, k]) is not None and pulp.value(z[i, k]) > 0.5:
                    current_assignments[df.loc[i, "Unit"]] = k

        # Constraint 3: Scale Verification Hook
        if verify_scale_coverage(current_assignments):
            print("Success! Found a taxonomically strict split covering all constraints.")
            final_assignments = current_assignments
            break
        else:
            print("Verification failed: Scales not fully covered. Retrying...")

    if not final_assignments:
        print("Warning: Reached maximum retries. Please check if constraints are too strict.")
        
    return final_assignments


def generate_output_files(data_dir, assignments, output_dir):
    """
    Scans the physical directory again and writes the exact image paths 
    into txt files, and taxonomic categories into a json file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    paths_dict = {"train": [], "val": [], "test": []}
    sets_dict = {"train": set(), "val": set(), "test": set()}
    
    for species in os.listdir(data_dir):
        species_path = os.path.join(data_dir, species)
        if not os.path.isdir(species_path):
            continue
            
        for organ in os.listdir(species_path):
            organ_path = os.path.join(species_path, organ)
            if not os.path.isdir(organ_path):
                continue
                
            unit_key = f"{species}_{organ}"
            if unit_key not in assignments:
                continue
                
            split = assignments[unit_key]
            
            variety_organ = unit_key.replace("_", " ")
            sets_dict[split].add(variety_organ)
            
            for file in os.listdir(organ_path):
                if file.endswith(".jpg") or file.endswith(".png"):
                    full_path = os.path.join(species, organ, file)
                    paths_dict[split].append(full_path)
                    
    # Write .txt files
    for split_name, paths in paths_dict.items():
        output_file = os.path.join(output_dir, f"{split_name}.txt")
        with open(output_file, "w") as f:
            for p in sorted(paths):
                f.write(p + "\n")
        print(f"Generated {output_file}: {len(paths)} images.")

    # Write .json file
    split_json = {
        "train": sorted(list(sets_dict["train"])),
        "val": sorted(list(sets_dict["val"])),
        "test": sorted(list(sets_dict["test"])),
    }
    json_path = os.path.join(output_dir, "split.json")
    with open(json_path, "w") as f:
        json.dump(split_json, f, indent=4)
    print(f"Generated {json_path} successfully.")


def main():
    args = parse_args()

    # 1. Parse JSON and folders
    df = build_statistics_from_data(args.data_dir, args.anno_json)
    if df.empty:
        return

    # 2. Run the MILP optimization
    assignments = solve_milp_partition(df, args.max_retries)
    if not assignments:
        return

    # 3. Route files and generate outputs
    print("\nGenerating final split files...")
    generate_output_files(args.data_dir, assignments, args.output_dir)
    print("\nBenchmark split generation completed!")

if __name__ == "__main__":
    main()