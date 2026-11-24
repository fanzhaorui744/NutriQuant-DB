# NutriQuant-DB Dataset Card

## 1. Dataset Summary

- **Dataset name:** `NutriQuant-DB`
- **Scope:** Monocular smartphone RGB images of portioned cafeteria meals with electronic-scale ground-truth mass and USDA SR28 energy coefficients.
- **Scale:** **786 annotated cafeteria images**, **2,094 food instances** across **10 cooked-food categories**.
- **Categories (`10`):**
  - `rice` (Rice), `tudou` (Potato), `tudousi` (Shredded potato), `lajiao` (Pepper), `qingcai` (Leafy vegetables), `xilanhua` (Broccoli), `tofu` (Tofu), `qiezi` (Eggplant), `jirou` (Chicken), `egg` (Egg).

---

## 2. Restricted & Password-Protected Full Release (Google Drive)

To keep the GitHub repository lightweight and enforce academic access control, the complete dataset archive (**256.37 MB**) is hosted on Google Drive with **Restricted** sharing permissions and **ZIP password encryption**:

- **Google Drive Link (Restricted):** [https://drive.google.com/file/d/1uRg9CPDsKoWgD67H2SViRSOsQahn2OAR/view?usp=sharing](https://drive.google.com/file/d/1uRg9CPDsKoWgD67H2SViRSOsQahn2OAR/view?usp=sharing)

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

### How to Request Access & Decryption Password

1. Click the Google Drive link above and submit a **Request access** form, or email **`fanzhaorui744@gmail.com`** (or the corresponding author listed in the manuscript) with your **full name, institutional affiliation, and intended research use**.
2. Upon approval, Google Drive access will be granted and the extraction password will be sent to you.

---

## 3. Ready-to-Run Demo Samples

For immediate testing of the live interactive demo (`https://fanzhaorui744.github.io/NutriQuant-DB/`), local HTML web demo (`python demo_app.py`), and CLI demo (`python main.py`), **4 curated sample images** and their ground-truth weights are included directly in:
- `data/test_images/` (4 curated samples)
- `data/ground_truth.csv` (11 component rows for the 4 samples)

---

## 4. Terms of Use & Limitations

- **Code license:** Research & academic use.
- **Data license:** Non-commercial academic research only (CC BY-NC 4.0); redistribution of the decrypted images without permission is prohibited.
- **Energy note:** Calorie values are calculated from measured served mass and traceable USDA SR28 reference entries (`nutrition.py`), not bomb calorimetry.
