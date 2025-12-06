## 📌 EXPLAINABLE AI FOR EARLY STROKE DETECTION: CAUSAL AND CLINICIAN-CENTERED INTERPRETATION OF BRAIN IMAGING

## 🧠 Overview

Acute ischemic stroke affects 13.7 million people each year, and outcomes depend heavily on fast and accurate diagnosis. Although deep learning models reach Dice scores between 0.60 and 0.75, recent work shows that strong accuracy does not ensure safe clinical use. This repository describes a causal intervention framework that evaluates whether stroke-detection models rely on meaningful clinical evidence rather than misleading image patterns.
This project implements:

* A full 3D U-Net pipeline for stroke lesion segmentation
* Preprocessing and normalization for multi-center MRI
* Evaluation across Dice, IoU, precision, recall, lesion volume correlation
* Gradient-based and perturbation-based XAI methods
* A causal intervention technique to measure whether regions of the image cause the prediction
* A simulated clinician review process to judge reasoning quality

### Key Innovation

We demonstrate that a model achieving **92% Dice score** can be **unanimously rejected by clinical experts** due to inappropriate reasoning—relying 56.2% on contralateral hemisphere features (a spurious correlation) rather than actual lesion characteristics. Five widely-used XAI methods (GradCAM, Integrated Gradients, Occlusion, LIME, Attention Rollout) **failed to detect this critical flaw**.

## 🎯 Problem Statement

Current deep learning models for stroke detection face a fundamental challenge:

- ⚠️ **High accuracy ≠ Clinical safety**: Models can achieve excellent performance by learning shortcuts
- 🔍 **Correlation ≠ Causation**: Traditional XAI methods show *what* models look at, not *whether they should*
- 🏥 **Regulatory gap**: FDA requires reasoning validation, but existing tools cannot provide it
- 🚨 **Patient safety risk**: Models relying on spurious features fail unpredictably on atypical cases

## ✨ Key Features

### 1. Causal Intervention Analysis
- Systematic perturbation of anatomically-defined vascular territories
- Quantification of causal effects on model predictions
- Mapping to explicit clinical appropriateness criteria

### 2. Clinician-Centered Evaluation
- Structured expert review protocol
- Alignment with diagnostic standards
- Simulated multi-specialty panel (neuroradiologists + stroke neurologists)

### 3. Regulatory-Compliant Audit Trail
- Complete event logging (FDA 21 CFR Part 11 compliant)
- Full provenance from input to deployment decision
- Transparent, reproducible validation pipeline

### 4. Two-Tier Validation Framework
| **Tier 1: Performance** | **Tier 2: Reasoning** |
|------------------------|----------------------|
| Dice coefficient, precision, recall | Causal appropriateness scores |
| Traditional accuracy metrics | Domain-specific feature validation |
| Necessary but insufficient | Essential for clinical safety |

## 📊 Results

### Main Findings

Our framework successfully identified dangerous reasoning patterns in a high-performing model:

| Metric | Value | Clinical Assessment |
|--------|-------|-------------------|
| **Dice Score** | 0.9232 | ✅ Excellent accuracy |
| **Lesion Dependency** | 9.1% | ❌ Critically low |
| **Contralateral Dependency** | 56.2% | ❌ **Spurious correlation** |
| **Expert Rejection Rate** | 100% (5/5) | ❌ Unsafe for deployment |
| **Clinical Trust Score** | 1.16/5 | ❌ Very low trust |

### XAI Method Comparison

| Method | Spurious Detection | Clinical Usability |
|--------|-------------------|-------------------|
| GradCAM | ❌ Failed | Low (diffuse patterns) |
| Integrated Gradients | ❌ Failed | Low (baseline-dependent) |
| Occlusion | ❌ Failed | Medium (no validation) |
| LIME | ❌ Failed | Low (unstable) |
| Attention Rollout | ❌ Failed | Low (attention ≠ causation) |
| **Causal Intervention** | ✅ **Detected** | **High (100% approval)** |

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 1.9
CUDA >= 11.0 (for GPU support)
```

### Installation

```bash
pip install -r requirements.txt
```

### How to Run

```python
from src.model import StrokeSegmentationModel
from src.causal_analysis import CausalInterventionFramework
from src.evaluation import ClinicalEvaluator

# Load model and data
model = StrokeSegmentationModel.load_pretrained("checkpoints/best_model.pth")
dwi_volume = load_dwi_scan("path/to/scan.nii.gz")

# Run causal intervention analysis
framework = CausalInterventionFramework(
    model=model,
    territories=['ACA', 'MCA', 'PCA', 'lesion', 'contralateral']
)
causal_results = framework.analyze(dwi_volume)

# Clinical evaluation
evaluator = ClinicalEvaluator()
assessment = evaluator.evaluate(causal_results)
print(f"Deployment Decision: {assessment.decision}")
print(f"Clinical Trust Score: {assessment.trust_score}/5")
```

## 📁 Dataset

This project uses the **ISLES 2022** dataset:

- **Source**: [Kaggle - ISLES 2022](https://www.kaggle.com/datasets/orvile/isles-2022-brain-stoke-dataset/)
- **Size**: 250 patients with diffusion-weighted MRI scans
- **Split**: 175 training / 37 validation / 38 test
- **Annotations**: Expert-labeled lesion masks
- **Characteristics**: Multi-center, diverse protocols, varying lesion sizes (0.1-100 mL)

### Preprocessing Pipeline

```
Input DWI → Resize (80³) → Skull Stripping → Intensity Normalization → Z-score Standardization
```

## 🏗️ Project Structure

```
explainable-stroke-detection/
├── src/
│   ├── model.py                 # 3D U-Net architecture
│   ├── causal_analysis.py       # Causal intervention framework
│   ├── xai_methods.py           # Baseline XAI implementations
│   ├── evaluation.py            # Clinical evaluation protocol
│   ├── audit_trail.py           # Logging and compliance
│   └── utils.py                 # Data loading and preprocessing
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_causal_analysis.ipynb
│   └── 04_expert_evaluation.ipynb
├── configs/
│   ├── model_config.yaml        # Model hyperparameters
│   └── evaluation_config.yaml   # Evaluation criteria
├── checkpoints/                 # Trained model weights
├── results/                     # Experimental outputs
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔬 Methodology

### 1. Model Architecture
- **Base Model**: 3D U-Net with skip connections
- **Parameters**: 7.8M
- **Input Size**: 80 × 80 × 80 voxels
- **Loss Function**: Focal-Dice combined loss
- **Optimizer**: Adam with cosine annealing

### 2. Causal Intervention Protocol

For each vascular territory T:

1. **Baseline Prediction**: Compute Dice score on original image
2. **Intervention**: Replace territory T voxels with matched noise
3. **Counterfactual Prediction**: Re-compute Dice score
4. **Causal Effect**: ΔDice = Baseline - Counterfactual

### 3. Clinical Appropriateness Criteria

| Territory | Expected Effect | Clinical Rationale |
|-----------|----------------|-------------------|
| **DWI Lesion** | >60% drop | Primary diagnostic feature |
| **Affected Vascular Territory** | 20-80% drop | Anatomical context |
| **Adjacent Territories** | 10-40% drop | Related anatomy |
| **Contralateral Hemisphere** | <20% drop | Minimal diagnostic role |

### 4. Simulated Expert Evaluation

Structured questionnaire assessing:
- Anatomical appropriateness (1-5 scale)
- Feature importance ranking
- Spurious correlation detection
- Deployment recommendation (Approve/Conditional/Reject)
- Clinical trust score (1-5 scale)

## 📈 Key Findings

### 1. The Accuracy-Trustworthiness Gap

High segmentation accuracy does not guarantee clinically appropriate reasoning. Our framework identified:

- ✅ **Statistical Success**: 92% Dice coefficient
- ❌ **Clinical Failure**: 56.2% dependence on non-diagnostic features
- ⚠️ **Safety Risk**: Would fail on bilateral strokes, anatomical variants

### 2. Limitations of Correlation-Based XAI

All five tested XAI methods failed to:
- Distinguish valid from spurious feature correlations
- Provide normative assessment against clinical standards
- Enable evidence-based safety evaluation

### 3. Simulated Clinical Expert Consensus

- 100% spurious correlation detection rate (5/5 experts)
- 100% deployment rejection despite 92% accuracy
- 100% preference for causal intervention over gradient-based methods

### 4. Regulatory Compliance

Complete audit trail enables:
- FDA 21 CFR Part 11 compliance
- Full traceability from input to deployment decision
- Evidence-based safety documentation

## 🔮 Future Work

### Immediate Extensions
- [ ] Real-world clinical validation with multi-center expert panels
- [ ] Multimodal integration (DWI + ADC + FLAIR + perfusion)
- [ ] Patient-specific vascular territory segmentation
- [ ] Alternative perturbation strategies (inpainting, counterfactual generation)

### Long-term Research
- [ ] Reasoning-aware training objectives that penalize spurious dependencies
- [ ] Automated appropriateness prediction for scalable evaluation
- [ ] Continuous post-deployment monitoring for reasoning drift
- [ ] Extension to other medical imaging tasks (lung nodules, diabetic retinopathy)

### 📺 Video Presentation
This repository describes the causal inference framework and experimental results for early stroke detection.  
Watch the project presentation: https://youtu.be/-f_VkAsTw7k 

### 🎤 Presentation
Access the project presentation slides here: [Explainable-AI-for-Early-Stroke-Detection.pdf](Explainable-AI-for-Early-Stroke-Detection.pdf)


## 📝 Citation

If you use this work in your research, please cite:

Owusu, D. Blemano, T.A.D (2025). Explainable AI for Stroke Detection: Causal and Clinician-Centered Interpretation of Brain Imaging.


## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ISLES 2022 Challenge** for providing the dataset
- **Clinical collaborators** for domain expertise
- **Open-source community** for foundational tools (PyTorch, scikit-learn, matplotlib)

## 📧 Contact

For questions or collaboration inquiries:

- **Email**: [denniso@mtu.edu]
- **Issues**: [GitHub Issues](https://github.com/royaldennis/explainable-stroke-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/royaldennis/explainable-stroke-detection/discussions)

---

**⚠️ Clinical Use Disclaimer**: This is a research tool and not approved for clinical diagnosis. Any clinical application requires appropriate regulatory approval and prospective validation.

**🌟 Star this repo** if you find it useful for your research!

*Made with ❤️ for safer medical AI*

</div>

