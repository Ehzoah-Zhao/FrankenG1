"""Setup G1-Frankenstein dataset from mapped annotations.

Usage: python setup_g1_dataset.py
"""
import os, sys, json, shutil, random, subprocess

def setup():
    base = "datasets/annotations"
    fk_dir = os.path.join(base, "frankenstein-dataset")
    g1_dir = os.path.join(base, "g1-frankenstein")
    # HuggingFace nested layout
    fk_annotations = os.path.join(fk_dir, "annotations", "annotations.json")
    g1_annotations_mapped = os.path.join(fk_dir, "annotations_g1.json")

    print("=" * 60)
    print("Step 1: Checking prerequisites...")
    if not os.path.exists(fk_annotations):
        print("ERROR: not found: " + fk_annotations)
        print("Download: git clone https://huggingface.co/datasets/Coral79/frankenstein-dataset " + fk_dir)
        sys.exit(1)
    print("  Frankenstein annotations found!")

    print("\nStep 2: Running annotation mapping...")
    if not os.path.exists(g1_annotations_mapped):
        print("  Running build_g1_part_annotations.py ...")
        result = subprocess.run([sys.executable, "build_g1_part_annotations.py"],
                                capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        print(result.stdout)
        if result.returncode != 0:
            print("Mapping failed!"); print(result.stderr); sys.exit(1)
    else:
        print("  annotations_g1.json already exists, skipping.")

    print("\nStep 3: Creating G1 dataset directory...")
    os.makedirs(os.path.join(g1_dir, "splits"), exist_ok=True)
    os.makedirs(os.path.join(g1_dir, "text_embeddings", "clip"), exist_ok=True)

    with open(g1_annotations_mapped, "r", encoding="utf-8") as f:
        g1_annotations = json.load(f)
    g1_keys = list(g1_annotations.keys())
    print("  " + str(len(g1_keys)) + " G1 entries with part-level annotations")

    annotations_out = {}
    for key in g1_keys:
        entry = g1_annotations[key]
        annotations_out[key] = {
            "annotations": entry["annotations"],
            "duration": entry["duration"],
        }
    with open(os.path.join(g1_dir, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(annotations_out, f, indent=2, ensure_ascii=False)
    print("  Saved " + str(len(annotations_out)) + " entries")

    print("\nStep 4: Generating splits...")
    random.seed(42)
    random.shuffle(g1_keys)
    n = len(g1_keys)
    splits = {
        "train": g1_keys[:int(n * 0.80)],
        "val":   g1_keys[int(n * 0.80):int(n * 0.90)],
        "test":  g1_keys[int(n * 0.90):],
    }
    for sn, keys in splits.items():
        sp = os.path.join(g1_dir, "splits", sn + ".txt")
        with open(sp, "w") as f:
            for k in keys: f.write(k + "\n")
        print("  " + sn + ": " + str(len(keys)) + " samples")

    print("\nStep 5: Linking CLIP embeddings...")
    fk_clip = os.path.join(fk_dir, "text_embeddings", "clip")
    g1_clip = os.path.join(g1_dir, "text_embeddings", "clip")
    for fn in ["clip.npy", "clip_index.json", "clip_slice.npy"]:
        src = os.path.join(fk_clip, fn)
        dst = os.path.join(g1_clip, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            try: os.symlink(os.path.abspath(src), dst); print("  Symlinked: " + fn)
            except OSError: shutil.copy2(src, dst); print("  Copied: " + fn)
        elif os.path.exists(dst):
            print("  Already exists: " + fn)
        else:
            print("  WARNING: missing " + src)

    print("\n" + "=" * 60)
    print("Setup complete! Train with:")
    print("  python train_g1.py dataset=g1-frankenstein")
    print("=" * 60)

if __name__ == "__main__":
    setup()
