import streamlit as st
from components.shared import add_contact_info

# 设置页面配置
st.set_page_config(
    page_title="GPMA Tutorial",
    page_icon="📖",
    layout="wide"
)

# 添加联系信息到侧边栏
add_contact_info()

# 设置页面样式
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .main > div {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .step-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("GPMA Tutorial")

# 使用教程
st.markdown("## How to Use GPMA")

with st.container():
    st.markdown("### Step 1: Data Preparation")
    st.markdown("""
    Before using GPMA, please ensure you have:
    - Digital pathology images in WSI format (supporting .svs or .ndpi formats)
    - Pathological staging information for the patient
    - Patient age information
    """)

with st.container():
    st.markdown("### Step 2: Data Upload and Analysis")
    st.markdown("""
    1. Upload a WSI image on the Analysis page.
    2. Enter the pathological stage (1-4).
    3. Enter the patient's age.
    4. Click the "Submit" button to start the analysis.
    """)

# 结果解读
st.markdown("## Understanding the Results")

with st.container():
    st.markdown("### Risk Heatmap")
    st.markdown("""
    The heatmap displays the risk level in different regions of the WSI image:
    - Red areas indicate high-risk regions
    - Blue areas indicate low-risk regions
    """)

with st.container():
    st.markdown("### SHAP Analysis")
    st.markdown("""
    SHAP analysis shows the contribution of each feature to the prediction:
    - Positive values (red) indicate increased risk
    - Negative values (blue) indicate decreased risk
    """)

with st.container():
    st.markdown("### Survival Rate Plot")
    st.markdown("""
    The survival curve shows the predicted 1-year, 2-year, and 3-year survival rates:
    - The y-axis represents the probability of survival.
    - The x-axis represents time (in years).
    """)

# 常见问题
st.markdown("## FAQ")
with st.expander("Q: How long does it take to process large WSI images?"):
    st.write("Processing time depends on the size of the image and server load, typically taking 3-5 minutes.")

with st.expander("Q: What image formats are supported?"):
    st.write("Currently supports WSI images in .svs and .ndpi formats.")

with st.expander("Q: How to explain risk scoring?"):
    st.write("A risk score greater than 83.52901 is considered high risk, while a score less than that value is considered low risk.")

# 添加页脚
st.markdown("---")
st.markdown("<div style='text-align: center;'>© 2025 Zhongnan Hospital. All rights reserved.</div>", unsafe_allow_html=True)