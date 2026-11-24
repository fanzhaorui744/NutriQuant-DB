# NutriQuant-DB Dataset Card

## 1. Dataset Summary

- **Dataset name:** `NutriQuant-DB`
- **Scope:** Monocular smartphone RGB images of portioned cafeteria meals with electronic-scale ground-truth mass and USDA SR28 energy coefficients.
- **Scale:** **786 annotated cafeteria images**, **2,094 food instances** across **10 cooked-food categories**.
- **Categories (`10`):**
  - `rice` (Rice), `tudou` (Potato), `tudousi` (Shredded potato), `lajiao` (Pepper), `qingcai` (Leafy vegetables), `xilanhua` (Broccoli), `tofu` (Tofu), `qiezi` (Eggplant), `jirou` (Chicken), `egg` (Egg).

---

## 2. Password-Protected Full Release

The complete dataset is packaged in this repository as an encrypted archive (tracked via Git LFS):

- **Archive:** [`dataset/NutriQuant-DB_images_protected.zip`](dataset/NutriQuant-DB_images_protected.zip)
- **Checksum:** [`dataset/NutriQuant-DB_images_protected.zip.sha256`](dataset/NutriQuant-DB_images_protected.zip.sha256)

### Archive Contents

```text
NutriQuant-DB/
|- images/                              # 786 cafeteria meal photos (.jpg)
|- annotations/
|  |- annotations_10_categories.xml     # CVAT XML polygon + fiducial marker annotations
|  |- annotation_manifest.csv           # Flattened instance-level manifest (2,094 rows)
|  |- annotation_counts.csv             # Per-category instance & image counts
|  `- annotation_inventory.json         # Summary statistics & Shannon entropy
`- README_DATASET.txt
```

### How to Request the Decryption Password

To comply with institutional data-governance and privacy policies, the full image archive is password-protected. Researchers requesting access for non-commercial academic use should contact:

- **Email:** `fanzhaorui744@gmail.com` (or the corresponding author listed in the manuscript)
- Please include your **full name, institutional affiliation, and intended research use**.

Once you receive the password, extract the archive with standard `unzip`, 7-Zip, or Python `zipfile`.

---

## 3. Ready-to-Run Demo Samples

For immediate testing of the local HTML web demo (`python demo_app.py`) and CLI demo (`python main.py`), a public subset of sample images and ground-truth weights is included directly in:
- `data/test_images/`
- `data/ground_truth.csv`

---

## 4. Terms of Use & Limitations

- **Code license:** Research & academic use.
- **Data license:** Non-commercial academic research only (CC BY-NC 4.0); redistribution of the decrypted images without permission is prohibited.
- **Energy note:** Calorie values are calculated from measured served mass and traceable USDA SR28 reference entries (`nutrition.py`), not bomb calorimetry.
