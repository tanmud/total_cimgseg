import os
import shutil
import random
import json
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("/work/10572/tmudali/vista/carecomp/dataset")
NNUNET_RAW   = Path(os.environ["nnUNet_raw"])
SEED         = 42
TEST_RATIO   = 0.2   # 20% held out as test set

# Folder → (modality, dataset_id, dataset_name)
FOLDER_MAP = {
    "A ct_train":    ("CT",  "Dataset001_CARE_CT"),
    "B ct_train":    ("CT",  "Dataset001_CARE_CT"),
    "G ct_train":    ("CT",  "Dataset001_CARE_CT"),
    "C and D mr_train": ("MRI", "Dataset002_CARE_MR"),
    "E mr_train":    ("MRI", "Dataset002_CARE_MR"),
}

DATASET_META = {
    "Dataset001_CARE_CT": {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "target": 1},
        "file_ending": ".nii.gz",
    },
    "Dataset002_CARE_MR": {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "target": 1},
        "file_ending": ".nii.gz",
    },
}

# ── COLLECT ALL CASES PER DATASET ───────────────────────────────────────────
dataset_cases = {"Dataset001_CARE_CT": [], "Dataset002_CARE_MR": []}

for folder_name, (modality, dataset_name) in FOLDER_MAP.items():
    folder = RAW_DATA_DIR / folder_name
    images = sorted(folder.glob("*_image.nii.gz"))
    for img_path in images:
        case_id = img_path.name.replace("_image.nii.gz", "")
        lbl_path = folder / f"{case_id}_label.nii.gz"
        assert lbl_path.exists(), f"Missing label for {case_id}"
        dataset_cases[dataset_name].append((case_id, img_path, lbl_path))

# ── SPLIT AND COPY ───────────────────────────────────────────────────────────
for dataset_name, cases in dataset_cases.items():
    out_dir = NNUNET_RAW / dataset_name
    for split in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    shuffled = cases.copy()
    random.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * TEST_RATIO))
    test_cases  = shuffled[:n_test]
    train_cases = shuffled[n_test:]

    for case_id, img_path, lbl_path in train_cases:
        shutil.copy(img_path, out_dir / "imagesTr" / f"{case_id}_0000.nii.gz")
        shutil.copy(lbl_path, out_dir / "labelsTr" / f"{case_id}.nii.gz")

    for case_id, img_path, lbl_path in test_cases:
        shutil.copy(img_path, out_dir / "imagesTs" / f"{case_id}_0000.nii.gz")
        shutil.copy(lbl_path, out_dir / "labelsTs" / f"{case_id}.nii.gz")  # keep for eval

    # ── dataset.json ────────────────────────────────────────────────────────
    meta = DATASET_META[dataset_name]
    dataset_json = {
        "channel_names": meta["channel_names"],
        "labels": meta["labels"],
        "numTraining": len(train_cases),
        "file_ending": meta["file_ending"],
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"{dataset_name}: {len(train_cases)} train, {len(test_cases)} test")

print("Done! Now run:")
print("  nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity")
print("  nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity")