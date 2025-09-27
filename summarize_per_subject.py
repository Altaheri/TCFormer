#!/usr/bin/env python3
import os
import re
import sys
import csv

# === Configuration defaults (edit here if needed) ===
DEFAULT_ROOT_DIR = "/results/"
DEFAULT_OUT_FILE = "subject_summary.txt"
MODELS = [
    "TCFormer", "ATCNet", "EEGNet", "ShallowNet",
    "BaseNet", "EEGTCNet", "EEGConformer",
    "TSSEFFNet", "CTNet/heads=4, F1=20, emb_size=40", "MSCFormer"
]
DATASETS = ["2a", "2b", "HGD"]
# ======================================================================

# Regex to match subject lines in results.txt
SUBJ_LINE_RE = re.compile(
    r"Subject\s*(?P<id>\d+)\s*=>.*?Test Acc:\s*(?P<acc>\d*\.?\d+)",
    re.IGNORECASE | re.DOTALL
)

# Parse folder names of the form:
#  ..._loso_..._seed-0_..._aug-True_... (or aug-False)
# Returns dict with keys 'seed' (int), 'aug' (bool), 'loso' (bool) or None
# indicating leave-one-subject-out vs sub-dependent

def parse_folder_name(name):
    parts = name.split('_')
    seed = None
    aug = None
    loso = False
    for seg in parts:
        if seg.startswith('seed-'):
            try:
                seed = int(seg.split('seed-')[1])
            except ValueError:
                pass
        elif seg.startswith('aug-'):
            val = seg.split('aug-')[1]
            if val in ('True', 'False'):
                aug = (val == 'True')
        elif seg.lower() == 'loso':
            loso = True
    if seed is None or aug is None:
        return None
    return {'seed': seed, 'aug': aug, 'loso': loso}

# Collect accuracies per run, grouping by augmentation and loso flags
# Returns a nested dict: data[aug_flag][loso_flag][seed] = { subj_id: acc, ... }

def collect_run_results(root_dir):
    data = {False: {False: {}, True: {}}, True: {False: {}, True: {}}}
    for dirpath, _, files in os.walk(root_dir):
        if 'results.txt' not in files:
            continue
        folder = os.path.basename(dirpath)
        info = parse_folder_name(folder)
        if not info:
            continue
        seed = info['seed']
        aug = info['aug']
        loso = info['loso']
        txt_path = os.path.join(dirpath, 'results.txt')
        with open(txt_path, 'r') as f:
            text = f.read()
        subj_map = {}
        for m in SUBJ_LINE_RE.finditer(text):
            sid = int(m.group('id'))
            acc = float(m.group('acc'))
            subj_map[sid] = acc
        if subj_map:
            data[aug][loso][seed] = subj_map
    return data

# Write summary divided into 4 groups:
#  Without Aug - Without LOSO
#  Without Aug - LOSO
#  With Aug    - Without LOSO
#  With Aug    - LOSO

def write_summary(data, root_dir, out_file):
    out_path = os.path.join(root_dir, out_file)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for aug_flag in (False, True):
            for loso_flag in (False, True):
                # Header for this subgroup
                label = (
                    ('With Aug' if aug_flag else 'Without Aug') +
                    (' - LOSO' if loso_flag else ' - sub-dependent')
                )
                f.write(label + '\n')
                seeds = sorted(data[aug_flag][loso_flag].keys())
                if not seeds:
                    f.write('No runs found.\n\n')
                    continue
                # All subject IDs in this subgroup
                subjects = sorted({sid for run in data[aug_flag][loso_flag].values() for sid in run.keys()})
                # Column header: Subject, seed-0, seed-1, ...
                header = ['Subject'] + [f"seed-{s}" for s in seeds]
                writer.writerow(header)
                # Rows per subject
                for sid in subjects:
                    row = [str(sid)]
                    for s in seeds:
                        subj_map = data[aug_flag][loso_flag].get(s, {})
                        acc = subj_map.get(sid)
                        row.append(f"{acc:.4f}" if acc is not None else '')
                    writer.writerow(row)
                f.write('\n')
    print(f"Summary per subject written to {out_path}")

# Main execution
def main(root_dir, out_file):
    data = collect_run_results(root_dir)
    # Check that any runs were found
    found = any(data[aug][loso] for aug in data for loso in data[aug])
    if not found:
        print(f"No results.txt found under {root_dir}")
        return
    write_summary(data, root_dir, out_file)

if __name__ == '__main__':
    # Determine base directory from CLI or defaults
    base_dir = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_ROOT_DIR
    for model in MODELS:
        for ds in DATASETS:
            # Construct folder name and output file
            rt = os.path.join(base_dir, f"{model}/{ds}")
            out = rt + "/" + DEFAULT_OUT_FILE
            print(f"Processing model={model}, dataset={ds} under {rt}")
            main(rt, out)
