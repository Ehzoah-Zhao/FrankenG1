"""Frankenstein -> G1 part-level annotation mapping.

Frankenstein annotations.json format:
  Key: "00000" (numeric index)
  Value: {path, start, end, duration, annotations: [{text, start, end, bodypart, ...}]}

G1 index.csv format:
  source_path, start_frame, end_frame, new_name
  ./pose_data_g1/ACCAD/.../A1_poses.npy, 0, 117, 000000.npy

Match: Frankenstein "path" field vs G1 source_path (stripped)
"""
import csv, json, os

FK_ANNOTATIONS = "datasets/annotations/frankenstein-dataset/annotations/annotations.json"
G1_INDEX_CSV = "F:/HumanML3D_G1/index.csv"
OUTPUT_JSON = "datasets/annotations/frankenstein-dataset/annotations_g1.json"
OUTPUT_REPORT = "datasets/annotations/frankenstein-dataset/mapping_report.txt"

def clean_g1_path(source_path):
    """ ./pose_data_g1/ACCAD/.../A1_poses.npy -> ACCAD/.../A1_poses """
    p = source_path.replace("./pose_data_g1/", "").replace("\\", "/")
    if p.endswith(".npy"):
        p = p[:-4]
    return p

def build_mapping():
    # Load Frankenstein
    print("=" * 60)
    print("Loading Frankenstein annotations...")
    with open(FK_ANNOTATIONS, "r", encoding="utf-8") as f:
        fk_data = json.load(f)
    print("  " + str(len(fk_data)) + " entries")

    # Build index: path -> fk_entry
    fk_by_path = {}
    for key, entry in fk_data.items():
        p = entry.get("path", "").replace("\\", "/")
        fk_by_path[p] = entry
    print("  " + str(len(fk_by_path)) + " unique paths")

    # Load G1
    print("\nLoading G1 index...")
    g1_entries = []
    with open(G1_INDEX_CSV, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            source_path, start_frame, end_frame, new_name = row
            g1_entries.append({
                "source_path": source_path,
                "clean_path": clean_g1_path(source_path),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame) if end_frame != "-1" else -1,
                "new_name": new_name.replace(".npy", ""),
            })
    print("  " + str(len(g1_entries)) + " G1 entries")

    # Match
    print("\nMatching...")
    matched = {}
    unmatched = []
    exact = 0
    fuzzy = 0

    for g1 in g1_entries:
        cp = g1["clean_path"]
        fk_entry = fk_by_path.get(cp)
        
        if not fk_entry:
            # Try fuzzy: with/without _poses suffix variations
            for suffix in ["", "_poses", "_stageii"]:
                alt = cp
                for s in ["_poses", "_stageii"]:
                    if alt.endswith(s):
                        alt = alt[:-len(s)]
                fk_entry = fk_by_path.get(alt)
                if fk_entry:
                    fuzzy += 1
                    break
                alt = cp + suffix
                fk_entry = fk_by_path.get(alt)
                if fk_entry:
                    fuzzy += 1
                    break
        
        if fk_entry:
            exact += 1
            fps = 20.0
            g1_start_sec = g1["start_frame"] / fps
            g1_end_sec = (g1["end_frame"] / fps) if g1["end_frame"] > 0 else fk_entry.get("duration", 10.0)

            adjusted = []
            for ann in fk_entry.get("annotations", []):
                ac = dict(ann)
                s = max(0.0, ann.get("start", 0.0) - g1_start_sec)
                e = min(g1_end_sec - g1_start_sec, max(s + 0.01, ann.get("end", 0.0) - g1_start_sec))
                if e > s:
                    ac["start"] = round(s, 3)
                    ac["end"] = round(e, 3)
                    adjusted.append(ac)

            matched[g1["new_name"]] = {
                "g1_file": g1["new_name"],
                "g1_source": g1["source_path"],
                "fk_path": fk_entry.get("path", ""),
                "start_frame": g1["start_frame"],
                "end_frame": g1["end_frame"],
                "annotations": adjusted,
                "duration": g1_end_sec - g1_start_sec,
            }
        else:
            unmatched.append(g1)

    # Report
    print("\n" + "=" * 60)
    print("MAPPING RESULTS")
    print("=" * 60)
    mc = len(matched)
    total = len(g1_entries)
    print("  G1 entries:     " + str(total))
    print("  FK entries:     " + str(len(fk_data)))
    print("  FK unique paths:" + str(len(fk_by_path)))
    print("  Matched:        " + str(mc) + " (exact=" + str(exact) + " fuzzy=" + str(fuzzy) + ")")
    print("  Unmatched:      " + str(len(unmatched)))
    print("  Match rate:     {:.1f}%".format(mc/total*100 if total else 0))

    # Save JSON
    print("\nSaving " + OUTPUT_JSON)
    output = {}
    for key, entry in matched.items():
        output[key] = {
            "annotations": entry["annotations"],
            "duration": entry["duration"],
        }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("  " + str(len(output)) + " entries saved")

    # Save report
    print("Saving " + OUTPUT_REPORT)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("Frankenstein -> G1 Mapping Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("G1 entries: " + str(total) + "\n")
        f.write("FK entries: " + str(len(fk_data)) + "\n")
        f.write("Matched: " + str(mc) + " (exact=" + str(exact) + " fuzzy=" + str(fuzzy) + ")\n")
        f.write("Unmatched: " + str(len(unmatched)) + "\n")
        f.write("Match rate: {:.1f}%\n".format(mc/total*100 if total else 0))
        f.write("\n--- Sample matches ---\n")
        for key, entry in list(matched.items())[:20]:
            f.write("\n  G1 " + entry["g1_file"] + " <- FK " + entry["fk_path"] + "\n")
            bps = sorted(set(a.get("bodypart","?") for a in entry["annotations"]))
            f.write("    Parts: " + str(bps) + "\n")
        f.write("\n--- Unmatched ---\n")
        for g1 in unmatched[:20]:
            f.write("  " + g1["new_name"] + ": " + g1["source_path"] + "\n")

    print("\nDone!")

if __name__ == "__main__":
    build_mapping()
