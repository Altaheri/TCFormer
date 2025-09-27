#!/usr/bin/env python3
import os
import re
import sys

# === Configuration defaults (edit here if you prefer hard-coded paths) ===
DEFAULT_ROOT_DIR = "/results/TCFormer/2a"
DEFAULT_OUT_FILE = "summary.txt"
# ======================================================================

def parse_folder_name(name):
    """
    Parse run folder names like:
      ShallowNet_bcic2a_loso_seed-0_z-scale-meth2_aug-False_GPU0_20250709_0442

    Returns a dict with keys: model, dataset, loso (bool), seed, aug, or None if invalid.
    """
    parts = name.split('_')
    if len(parts) < 5:
        return None
    model   = parts[0]
    dataset = parts[1]
    # determine loso mode
    loso = any(part.lower() == 'loso' for part in parts)
    # find seed- and aug- segments
    seed = None
    aug  = None
    for seg in parts:
        if seg.startswith('seed-'):
            try:
                seed = int(seg.split('seed-')[1])
            except ValueError:
                return None
        if seg.startswith('aug-'):
            val = seg.split('aug-')[1]
            if val in ('True', 'False'):
                aug = (val == 'True')
    if seed is None or aug is None:
        return None
    return {
        'model': model,
        'dataset': dataset,
        'loso': loso,
        'seed': seed,
        'aug': aug,
    }


def extract_results(txt):
    """
    Extract #Params, Average Test Accuracy (mean ± std) and Average Test Kappa (mean ± std) from results.txt
    """
    out = {}
    # #Params
    m = re.search(r'#Params:\s*([0-9]+)', txt)
    if m:
        out['params'] = int(m.group(1))
    # Average Test Accuracy: mean ± std
    m = re.search(r'Average Test Accuracy:\s*([0-9]+\.?[0-9]*)\s*±\s*([0-9]+\.?[0-9]*)', txt)
    if m:
        out['acc_mean'] = float(m.group(1))
        out['acc_std']  = float(m.group(2))
    # Average Test Kappa: mean ± std
    m = re.search(r'Average Test Kappa:\s*([0-9]+\.?[0-9]*)\s*±\s*([0-9]+\.?[0-9]*)', txt)
    if m:
        out['kappa_mean'] = float(m.group(1))
        out['kappa_std']  = float(m.group(2))
    return out


def main(root_dir, out_file):
    # data[aug_flag][loso_flag][seed] = results dict
    data = {False: {False: {}, True: {}}, True: {False: {}, True: {}}}
    seeds = list(range(5))

    # walk through all subdirectories and select only those with results.txt
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'results.txt' not in filenames:
            continue
        name = os.path.basename(dirpath)
        info = parse_folder_name(name)
        if not info:
            continue
        loso = info['loso']
        aug  = info['aug']
        seed = info['seed']
        result_path = os.path.join(dirpath, 'results.txt')
        with open(result_path, 'r') as f:
            txt = f.read()
        res = extract_results(txt)
        if 'params' not in res or 'acc_mean' not in res or 'kappa_mean' not in res:
            continue
        data[aug][loso][seed] = res

    with open(os.path.join(root_dir, out_file), 'w') as f:
        # header
        header = (
            "#Aug?\t#Params\t" +
            "\t".join(f"sub-dep s{sd}" for sd in seeds) +
            "\t" +
            "\t".join(f"loso s{sd}" for sd in seeds)
        )
        f.write(header + "\n")

        for aug_flag in (False, True):
            tag = "'-Aug" if not aug_flag else "'+Aug"
            # find any params value
            params = None
            for loso_flag in (False, True):
                for sd in seeds:
                    if sd in data[aug_flag][loso_flag]:
                        params = data[aug_flag][loso_flag][sd]['params']
                        break
                if params is not None:
                    break

            # assemble accuracy row: acc_mean ± acc_std for each seed
            row_acc = []
            for loso_flag in (False, True):
                for sd in seeds:
                    res = data[aug_flag][loso_flag].get(sd)
                    if res:
                        row_acc.append(f"{res['acc_mean']:.2f} ± {res['acc_std']:.2f}")
                    else:
                        row_acc.append("0.00 ± 0.00")
            # assemble kappa row
            row_kappa = []
            for loso_flag in (False, True):
                for sd in seeds:
                    res = data[aug_flag][loso_flag].get(sd)
                    if res:
                        row_kappa.append(f"{res['kappa_mean']:.3f} ± {res['kappa_std']:.3f}")
                    else:
                        row_kappa.append("0.00 ± 0.00")

            # write rows
            f.write(f"{tag}\t{params}\t" + "\t".join(row_acc) + "\n")
            f.write(f"\t\t" + "\t".join(row_kappa) + "\n")

    print(f"Summary written to {out_file}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        root_dir, out_file = sys.argv[1], sys.argv[2]
    else:
        root_dir, out_file = DEFAULT_ROOT_DIR, DEFAULT_OUT_FILE
        print(f"Using defaults -> root_dir: {root_dir}, out_file: {out_file}")
    main(root_dir, out_file)
