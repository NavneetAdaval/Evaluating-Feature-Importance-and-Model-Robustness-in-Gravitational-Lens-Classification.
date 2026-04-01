# "Texture vs. Morphology: Evaluating Feature Importance and Model Robustness in Gravitational Lens Classification.

Multi-class classification of strong gravitational lensing images using DenseNet121, with analysis of inductive bias and corruption robustness.

\---

## Table of Contents

1. [Task Overview](#task-overview)
2. [Dataset](#dataset)
3. [Approach \& Architecture](#approach--architecture)
4. [Augmentation Strategy](#augmentation-strategy)
5. [Results](#results)
6. [Research Question 1 — Inductive Bias](#research-question-1--inductive-bias-texture-vs-morphology)
7. [Research Question 2 — Robustness Under Corruption](#research-question-2--robustness-under-corruption)
8. [Discussion \& Conclusions](#discussion--conclusions)
9. [Repository Structure](#repository-structure)

\---

## Task Overview

The goal is to build a classifier that distinguishes three classes of strong gravitational lensing images:

|Class|Label|Description|
|-|-|-|
|0|**No Substructure**|Smooth Einstein ring / arc|
|1|**Subhalo Substructure**|Perturbations from dark matter subhalos|
|2|**Vortex Substructure**|Perturbations from vortex-type substructure|

Beyond accuracy, two research questions are investigated:

* **Q1 (Inductive Bias):** Is classification driven by fine-grained local texture or large-scale global morphology?
* **Q2 (Robustness):** How stable is the model under realistic observational corruptions?

\---

## Dataset

* **Source:** [DeepLense](https://drive.google.com/...)
* **Classes:** 3 (no substructure, subhalo, vortex)
* **Preprocessing:** Images are pre-normalized using min-max normalization
* **Format:** Single-channel (grayscale) numpy arrays

\---

## Approach \& Architecture

**Model:** DenseNet121 (pretrained on ImageNet, fine-tuned for this task)

DenseNet121 was chosen for several reasons suited to this problem:

* **Dense connectivity** encourages feature reuse across scales — important when both local perturbations and global ring shape may carry signal
* **Compact and parameter-efficient** relative to ResNets at similar capacity
* **Strong transfer learning baseline** even for single-channel scientific images (channel replicated to 3 for compatibility)

**Training details:**

|Hyperparameter|Value|
|-|-|
|Optimizer|AdamW|
|Learning Rate|1e-4|
|Scheduler|CosineAnnealingLR|
|Batch Size|32|
|Epochs|15|
|Loss|Cross-Entropy|
|Metric|Macro-averaged AUC (OvR)|

Two separate models were trained:

* **Clean model** — no data augmentation
* **Aug model** — with the augmentation pipeline described below

\---

## Augmentation Strategy

Augmentations were designed to respect the physical symmetries of gravitational lensing images:

```python
@staticmethod
def augment\\\_tensor(t):
    # Random horizontal / vertical flip
    if random.random() > 0.5:
        t = TF.hflip(t)
    if random.random() > 0.5:
        t = TF.vflip(t)
    # Random 90° rotation (lensing images are rotationally symmetric)
    angle = random.choice(\\\[0, 90, 180, 270])
    t = TF.rotate(t, angle)
    # Mild Gaussian noise (σ sampled uniformly in \\\[0.01, 0.05])
    if random.random() > 0.5:
        sigma = random.uniform(0.01, 0.05)
        t = t + torch.randn\\\_like(t) \\\* sigma
    # Mild Gaussian blur (σ sampled uniformly in \\\[0.3, 1.0])
    if random.random() > 0.4:
        arr = t.numpy()
        sigma\\\_b = random.uniform(0.3, 1.0)
        arr = gaussian\\\_filter(arr, sigma=\\\[0, sigma\\\_b, sigma\\\_b])
        t = torch.tensor(arr)
    return t
```

**Design rationale:**

* Flips and 90° rotations are physically valid — a lensed image has no preferred orientation
* Gaussian noise simulates telescope detector noise
* Mild blur simulates seeing/PSF variation across observations
* Augmentations are conservative — no cropping or intensity rescaling that could destroy the radial lensing profile

\---

## Results

### Baseline AUC (No Corruption)

|Model|AUC|
|-|-|
|Clean (no augmentation)|**0.9768**|
|Aug (with augmentation)|**0.9897**|

Both models achieve strong classification performance. The augmented model improves by \~0.013 AUC, indicating the augmentations generalise without overfitting.

\---

## Research Question 1 — Inductive Bias: Texture vs Morphology?

To determine whether the model relies on **local texture** or **global spatial structure**, three probes were applied at inference time on held-out data:

### 1\. Low-Pass Filtering (Blurring)

Increasingly strong Gaussian blur progressively destroys fine-grained detail while preserving global layout. If the model relies on texture, performance should degrade quickly; if it relies on morphology, it should be more tolerant.

|σ|Clean AUC|Aug AUC|
|-|-|-|
|0 (baseline)|0.9768|0.9897|
|0.5|0.9710|0.9906|
|1|0.9595|0.9907|
|2|0.8840|0.9019|
|3|0.7434|0.6631|
|5|0.5704|0.5371|

Both models are **remarkably tolerant** up to σ=1–2, confirming that coarse morphological features carry the bulk of the signal. The sharp drop only at σ≥3 is consistent with morphology-driven classification where very heavy smoothing washes out the large-scale ring structure itself.

### 2\. High-Pass Filtering (Texture Residuals Only)

The low-frequency component is subtracted, leaving only high-frequency edges and texture residuals. If classification were texture-driven, performance should be above chance here.

|σ|Clean AUC|Aug AUC|
|-|-|-|
|1|0.5142|0.5145|
|2|0.5147|0.5074|
|3|0.5248|0.5084|
|5|0.5595|0.5248|

Both models perform **at chance** (≈0.50–0.56) on high-pass residuals alone across all σ. This decisively rules out local texture as a primary classification cue.

### 3\. Patch Shuffle

The image is divided into a grid and patches are randomly permuted, destroying global spatial layout while keeping all local patch statistics intact. A morphology-driven model should fail under this probe.

|Grid|Clean AUC|Aug AUC|
|-|-|-|
|2×2|0.6273|0.5482|
|3×3|0.5690|0.5166|
|5×5|0.5233|0.5164|
|7×7|0.5047|0.5021|
|10×10|0.4961|0.4937|

Performance collapses to chance even with coarse 2×2 or 3×3 shuffles. This is the strongest evidence that classification depends on the **coherent global arrangement** of lensing features — not any local statistics that would survive patch permutation.

### 4\. Grad-CAM Visualisation

Grad-CAM saliency maps were computed for both models across all three classes to directly visualise which spatial regions drive predictions.

**Key observations:**

* **Both models attend to the Einstein ring/arc**, not to point-like features or background regions. This is the direct visual confirmation of the morphology-driven finding from the filtering and patch-shuffle probes.
* For **`\\\[no]` substructure**, attention is distributed broadly and symmetrically along the full ring — the model is evaluating global ring coherence.
* For **`\\\[sphere]` subhalo** and **`\\\[vort]` vortex**, attention shifts toward the **perturbed arc segments** — the regions where substructure breaks the ring's symmetry — which is exactly where the physically discriminating signal is located.
* The **augmented model** shows more spatially distributed, ring-following attention maps, suggesting it has learned a more robust and complete representation of the ring morphology.
* The **clean model** tends to concentrate attention into a single dominant region rather than following the full arc, indicating a more brittle, less spatially consistent representation — consistent with its lower robustness under rotation.

This confirms that neither model is using spurious cues: both have learned to interrogate the global morphological structure of the lensed image.

### Q1 Summary

|Probe|Clean AUC (extreme)|Aug AUC (extreme)|Interpretation|
|-|-|-|-|
|Baseline|0.9768|0.9897|—|
|Low-pass (σ=5)|0.5704|0.5371|Heavy blur destroys morphology → collapse|
|High-pass (σ=5)|0.5595|0.5248|Texture alone → at chance|
|Patch shuffle (10×10)|0.4961|0.4937|Broken layout → at chance|

> \\\*\\\*Conclusion:\\\*\\\* Classification is driven by \\\*\\\*large-scale global morphology\\\*\\\* (the shape, continuity, and radial profile of the Einstein ring/arc). Local texture contributes negligibly. This is physically intuitive — subhalo and vortex perturbations manifest as coherent distortions in the ring shape, not as local textural differences.

\---

## Research Question 2 — Robustness Under Corruption

Five corruption types are applied at five severity levels each, testing how both models degrade under realistic and adversarial perturbations.

### Gaussian Noise

|σ|Clean AUC|Aug AUC|
|-|-|-|
|0.00|0.9768|0.9897|
|0.02|0.6867|0.9617|
|0.05|0.5758|0.8165|
|0.10|0.5220|0.6223|
|0.20|0.5035|0.5086|

The augmented model is substantially more robust — at σ=0.02 it retains 0.96 vs 0.69 for clean. Both collapse at σ=0.20. Training noise (σ∈\[0.01, 0.05]) directly transfers to test robustness.

### Gaussian Blur

|σ|Clean AUC|Aug AUC|
|-|-|-|
|0.0|0.9768|0.9897|
|0.5|0.9710|0.9906|
|1.0|0.9595|0.9907|
|2.0|0.8840|0.9019|
|3.0|0.7434|0.6631|

Both models are highly tolerant up to σ=2. The aug model slightly overtakes at moderate blur but they converge at heavy blur. Blur robustness is comparable between the two.

### Rotation

|Angle|Clean AUC|Aug AUC|
|-|-|-|
|0°|0.9768|0.9897|
|45°|0.7468|0.8750|
|90°|0.7279|0.9896|
|135°|0.7408|0.9184|
|180°|0.9761|0.9888|

The most striking result. The clean model degrades to \~0.73 at 45°/90°/135° but recovers at 180° (since 180° is equivalent to a double-flip). The aug model, trained with 90° rotation augmentation, is essentially invariant across all angles (≥0.875 throughout). This is a direct and clean demonstration of augmentation-induced invariance.

### Brightness Shift

|Offset|Clean AUC|Aug AUC|
|-|-|-|
|−0.20|0.5962|0.5431|
|−0.10|0.7783|0.7149|
|0.00|0.9768|0.9897|
|+0.10|0.8381|0.7641|
|+0.20|0.7763|0.7251|

Both models are sensitive to brightness shift, but the clean model is slightly more robust. Brightness augmentation was not included in the training pipeline, and neither model handles it well at large offsets. This is a clear gap for future work.

### Center Masking

|Masked fraction|Clean AUC|Aug AUC|
|-|-|-|
|0.0|0.9768|0.9897|
|0.2|0.6183|0.5910|
|0.4|0.5153|0.5072|
|0.6|0.5022|0.5069|
|0.8|0.5065|0.5068|

Both models collapse to chance upon masking just 20–40% of the center. This is physically expected — the Einstein ring, which is the primary discriminating feature, is concentrated in the central region of the image. Masking it removes the classification signal entirely, and this is consistent with the morphology-driven finding from Q1.

### Q2 Summary Table

|Corruption|Mean AUC (Clean)|Mean AUC (Aug)|Winner|Δ|
|-|-|-|-|-|
|Gaussian Noise|0.5720|0.7273|**Aug**|+0.155|
|Gaussian Blur|0.8895|0.8866|Clean|−0.003|
|Rotation|0.7979|0.9429|**Aug**|+0.145|
|Brightness Shift|0.8424|0.7984|Clean|−0.044|
|Center Masking|0.5356|0.5280|Clean|−0.008|

> \\\*\\\*Conclusion:\\\*\\\* Augmentation provides large and meaningful robustness gains for corruptions it was trained against (noise: +0.155, rotation: +0.145). For corruptions not seen during training (brightness shift, center masking), both models perform similarly and neither is robust. This validates the role of augmentation as an inductive bias tool — it generalises to the specific invariances it encodes, not universally.

\---

## Discussion \& Conclusions

### What the results tell us physically

Gravitational lensing substructure classification is a **morphological task**. The discriminating signal is encoded in the global shape of the Einstein ring — its continuity, symmetry, and the spatial pattern of distortions — rather than in fine-grained pixel-level statistics. This aligns with the physics: subhalo and vortex perturbations create coherent, extended distortions in the lensed arc, which are not reducible to local texture differences.

### Model behaviour

DenseNet121's dense feature reuse across scales appears well-suited to this: the model captures both the fine structure needed to detect small perturbations and the global context needed to evaluate ring coherence. Both the clean and augmented models learn equivalent representations (as shown by comparable high-pass and patch-shuffle results), with the augmented model additionally encoding rotation and noise invariances.

### Augmentation is a targeted tool

The augmented model is not universally more robust. It is specifically more robust to the transformations it was trained with (noise and rotation) and no more robust than the clean model to unseen corruptions (brightness, masking). This is the expected behaviour of inductive-bias regularisation and reinforces that augmentation should be designed to match the expected distribution shifts in deployment.

### Limitations and future work

* **Brightness robustness is poor for both models** — adding brightness/contrast jitter to augmentations would likely close this gap
* **Center masking causes full collapse** — this is a fundamental limitation given the physics; it is not fixable through augmentation. Methods that attend to the ring periphery or use multiple spatial hypotheses could help
* Exploring **equivariant architectures** (e.g., group-equivariant CNNs) could yield further rotation robustness beyond augmentation

\---

