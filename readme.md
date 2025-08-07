# Final Project: Muscle Stem Cell Behavior Analysis

This repository contains the complete computational pipeline for quantifying and modeling muscle stem cell (muSC) motility and morphology, developed as part of the MSc Final Project.

## Repository Structure

```
Final-Project/                  # Root directory
│
├── CellBehaviour_Modelling_and_Analysis.py  # Core Python script with functions for each analysis stage
├── Final_Project_Notebook.ipynb             # Jupyter notebook orchestrating the analysis (sections 1–26)
├── outputs_final/                           # Exported results
│   ├── *.csv                                # Per-step data tables (metrics, model parameters, statistics)
│   └── *.png                                # Figures (plots, overlays, diagnostics)
└── README.md                                # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ShaliniDaw/Final-Project.git
   cd Final-Project
   ```
2. Create a Python virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

## Usage

### 1. Run the Notebook

Open `Final_Project_Notebook.ipynb` in JupyterLab or Jupyter Notebook and run all cells sequentially. Each section corresponds to a report chapter:

- **Sections 1–2:** Environment setup, file discovery, initial inspection
- **Sections 3–5:** Object detection, tracking, metric computation
- **Section 6:** Diffusion-model fitting and spatial analyses
- **Sections 7–9:** Statistical testing and injury-response integration
- **Section 10:** Predictive modeling and feature importance
- **Sections 11–13:** Advanced exploratory analyses (anomalous diffusion, clustering, Markov/survival)

### 2. Run the Python Script

To execute the entire pipeline in a single run (non-interactive), use:

```bash
python CellBehaviour_Modelling_and_Analysis.py
```

Outputs (CSV and PNG files) will be saved under the `outputs_final/` directory.

## Output Files

- \`\`: Data tables containing extracted metrics, model parameters, and statistical test results.
- \`\`: Publication-ready figures for inclusion in the project report.

Refer to each file’s header comments or the notebook text for detailed descriptions.

## Reproducibility

- The notebook and script include section headers and comments explaining each analytical step.
- Clear naming conventions for exported files ensure traceability between code and report.

## Professional & Ethical Notes

- Data used are anonymized and pre-segmented; no human or personal data are involved.
- Analysis adheres to BCS and IET codes of conduct, with transparent documentation and sustainable computing practices.

## Contact

For questions or collaboration, please contact **Shalini Daw** at [**k24045762@kcl.ac.uk**](mailto\:k24045762@kcl.ac.uk).

