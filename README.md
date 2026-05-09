# GPMA: Gastric Cancer Prognostication Model



GPMA (Gastric Prognostic Model Architecture) is a deep learning-based application for predicting the prognosis of gastric cancer patients. By integrating parallel Mamba, residual connections, wavelet transform, and feature fusion modules, this model significantly enhances the efficiency of image feature extraction and the accuracy of model prediction. It predicts patient prognosis by analyzing whole slide images (WSI) of digital pathology alongside clinical data.

## Key Features
- **Information Acquisition Module**: Processes multi-format WSI files (e.g., .svs, .ndpi, .sdpc) and clinical data (Age, Pathological Stage).
- **Image Processing & Feature Extraction**: Utilizes an advanced Wavelet Parallel Mamba Model to efficiently extract spatial features from WSIs.
- **Risk & Prognosis Prediction**: Predicts survival rates over 1, 2, and 3 years, and provides an overall risk score using a Cox Proportional Hazards model.
- **Explainability**: Features visual Risk Heatmaps and SHAP (SHapley Additive exPlanations) analysis to interpret feature contributions clearly.
- **Nomogram**: Integrates the deep learning predictions with clinical factors to create a classical nomogram for clinicians.

## Prerequisites
- Python 3.8 or higher
- CUDA-enabled GPU is highly recommended for faster WSI processing and inference.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/GPMAapp.git
   cd GPMAapp
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   conda create -n gpma_env python=3.10
   conda activate gpma_env
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The installation of `mamba-ssm` and `causal-conv1d` requires a CUDA toolkit and PyTorch configured for GPU. Ensure you install the appropriate versions corresponding to your CUDA setup.)*

4. **Install OpenSlide (for WSI reading):**
   - **Ubuntu/Debian:** `sudo apt-get install openslide-tools`
   - **CentOS/RedHat:** `sudo yum install openslide`
   - **macOS:** `brew install openslide`
   - **Windows:** Download the latest binaries from [OpenSlide](https://openslide.org/download/).

## Usage

1. **Start the Streamlit App:**
   From the root directory of the project, run:
   ```bash
   streamlit run Home.py
   ```

2. **Using the Application:**
   - Navigate to the **Analysis** page via the sidebar.
   - **Upload a WSI Image:** Upload a whole slide image (formats: `.svs`, `.ndpi`, `.sdpc`).
   - **Input Clinical Data:** Use the sliders to input the patient's **Pathological Stage** and **Age**.
   - Click **Submit**.

3. **Interpreting Results:**
   - **Raw WSI:** View the thumbnail of the uploaded slide.
   - **Risk map:** Observe the high-risk and low-risk spatial regions on the slide.
   - **Features Contribution:** Understand the weight of the model's score, patient age, and stage via the SHAP analysis bar chart.
   - **Survival Rate Plot:** View the projected survival probabilities across 1 to 3 years.
   - **Nomogram:** A comprehensive clinical tool summarizing the prognostic factors.

## Project Structure
- `Home.py`: Main entry point for the Streamlit web application.
- `pages/`: Contains the subpages for the application, including the Analysis logic (`2_Analysis.py`).
- `components/`, `vis_utils/`, `wsi_core/`, `utils/`: Helper modules for visualization, WSI handling, and core pipeline logic.
- `models/`, `part/`, `mamba/`: The deep learning architectures and custom layers (Mamba, Wavelet Transform, etc.).
- `config_template2.yaml`: Configuration file specifying patch size, model hyperparameters, etc.
- `s_1_checkpoint_GPMA.pth` & `cox_model.pkl`: Pre-trained model weights.

## Citation
If you find this project useful for your research, please cite our upcoming paper:
> **Topology-aware computational pathology reveals spatial ecological states underlying prognosis in gastric cancer**

## Acknowledgement
We thank the investigators who generated and publicly shared the single-cell datasets used in this study through repositories including GEO. Their commitment to open data made this work possible.

## License
© 2025 Zhongnan Hospital. All rights reserved.
