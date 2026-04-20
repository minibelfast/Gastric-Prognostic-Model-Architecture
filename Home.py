import streamlit as st
from components.shared import add_contact_info

# 设置页面配置
st.set_page_config(
    page_title="GPMA for Gastric Cancer Prognostication",
    page_icon="🔬",
    layout="wide"
)

# 添加联系信息到侧边栏
add_contact_info()

# 页面内容
st.title("GPMA: Gastric Cancer Prognostication Model")

# 模型简介
st.markdown("""
### Model Introduction
GPMA (Gastric Prognostic Model Architecture) is a deep learning-based model for predicting the prognosis of gastric cancer patients. By integrating parallel Mamba, residual connections, wavelet transform, and feature fusion modules, this model significantly enhances the efficiency of image feature extraction and the accuracy of model prediction. The model predicts patient prognosis by analyzing whole slide images (WSI) of digital pathology.
""")

# 显示模型架构图
st.markdown("### Model Architecture")
st.image('pic/model_architecture.png', 
         caption='GPMA Model Architecture',
         use_container_width=True)

# 模型特点
st.markdown("""
### Key Modules
- Information Acquisition Module
- Image Processing Module
- Wavelet Parallel Mamba Model Module
- Result Prediction Module
- Explainability Module
""")

# 添加页脚
st.markdown("---")
st.markdown("<div style='text-align: center;'>© 2025 Zhongnan Hospital. All rights reserved.</div>", unsafe_allow_html=True)