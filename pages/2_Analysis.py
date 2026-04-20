import os

import pandas as pd
from lifelines import CoxPHFitter
import shap
import joblib  # or you can use import pickle  
import pandas as pd  


import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import h5py
import tempfile
import shutil
from pathlib import Path
from types import SimpleNamespace

# 导入项目模块
from utils.eval_utils import initiate_model
from models import get_encoder
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.batch_process_utils import initialize_df
from utils.file_utils import save_hdf5
from components.shared import add_contact_info

# 移除PIL库的图像大小限制
Image.MAX_IMAGE_PIXELS = None

# 配置文件和模型路径
CONFIG_PATH = 'config_template2.yaml'
MODEL_PATH = 's_1_checkpoint_GPMA.pth'
Coxmodel="cox_model.pkl"
train_data="9.1.dataset_train.csv"
# 设置页面配置
# 在文件开头的st.set_page_config中修改
st.set_page_config(
    page_title="GPMA Analysis",
    page_icon="📊",
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
        padding: 0;
        max-width: 100%;
        margin: 0;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1.5rem;
        background-color: #2ecc71;
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 0.5rem;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
    [data-testid="stFileUploader"] {
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        border: 2px dashed #bdc3c7;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
    }
    [data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stExpander"] > div:first-child {
        border-radius: 0.5rem 0.5rem 0 0;
        background-color: #f8f9fa;
        padding: 1rem;
    }
    h1 {
        color: #2c3e50;
        margin-bottom: 2rem;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
    }
    h3 {
        color: #34495e;
        margin: 1.5rem 0 1rem;
        font-size: 1.25rem;
        font-weight: 600;
    }
    [data-testid="stSlider"] {
        padding: 1.5rem 0;
    }
    [data-testid="stSlider"] > div:first-child {
        font-weight: 500;
    }
    .element-container {
        margin-bottom: 1.5rem;
    }
    [data-testid="column"]:first-child {
        background-color: #2c3e50;
        color: white;
        padding: 1.5rem;
        border-radius: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        height: 100vh;
    }
    [data-testid="column"]:not(:first-child) {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .shap-importance {
        margin-top: 2rem;
    }
    .nomogram-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin: 1rem 0;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
    .high-risk {
        color: #e74c3c;
    }
    .low-risk {
        color: #2ecc71;
    }
    /* 添加联系信息样式 */
    .contact-info {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.2);
        color: rgba(255,255,255,0.7);
    }
    .contact-info p {
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    /* 修改标题样式 */
    [data-testid="stExpander"] > div:first-child {
        background-color: #3498db;
        color: white;
    }
    /* 修改折叠图标颜色 */
    [data-testid="stExpander"] label.st-emotion-cache-1egp7rm {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 设置标题
st.title("GPMA for Gastric Cancer Prognostication")

# 创建两列布局
left_col, right_col = st.columns([1, 3])

# 左侧列：输入参数
with left_col:
    # 创建一个容器用于输入参数
    with st.container():
        st.markdown("### Input Parameters")
        
        # 创建可滚动的输入区域
        with st.form("input_form"):
            st.markdown("#### Upload WSI Image")
            upload_file = st.file_uploader("", type=["svs","ndpi","sdpc"], 
                help="Upload a WSI image in SVS format", 
                label_visibility='hidden', 
                accept_multiple_files=False)
            
            st.markdown("#### Pathological Stage")
            stage = st.slider("", 1, 4, 1, 
                help="Select the pathological stage", 
                label_visibility='hidden')
            
            st.markdown("#### Age(years)")
            age = st.slider("", 0, 100, 50, 
                help="Input patient's age", 
                label_visibility='hidden')
            
            # 提交按钮
            submit_button = st.form_submit_button("Submit", type="primary")
        
        # 添加一个分隔线
        st.markdown("<hr style='margin: 2rem 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

# 右侧列：结果展示
with right_col:
    # 创建两行两列的布局
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    # 第一行第一列：原始WSI图像
    with row1_col1:
        raw_wsi_container = st.expander("Raw WSI", expanded=True)
    
    # 第一行第二列：风险热图
    with row1_col2:
        risk_map_container = st.expander("Risk map", expanded=True)
    
    # 第二行第一列：特征贡献度分析
    with row2_col1:
        feature_container = st.expander("Features Contribution", expanded=True)
    
    # 第二行第二列：生存率曲线
    with row2_col2:
        survival_container = st.expander("Survival Rate Plot", expanded=True)
    
    # 第三行：Nomogram
    nomogram_container = st.expander("Nomogram", expanded=True)

# 创建临时目录用于处理上传的文件
temp_dir = Path("temp")
temp_dir.mkdir(exist_ok=True)


# 拟合cox比例风险模型


# 示例：加载数据（确保数据中包含'time'和'event'列）








# 处理函数
def process_wsi(wsi_path, stage, age):
    # 加载配置文件
    with open(CONFIG_PATH, "r", encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 准备参数
    patch_args = SimpleNamespace(**config_dict['patching_arguments'])
    data_args = SimpleNamespace(**config_dict['data_arguments'])
    model_args = config_dict['model_arguments']
    model_args.update({'n_classes': config_dict['exp_arguments']['n_classes']})
    model_args = SimpleNamespace(**model_args)
    encoder_args = SimpleNamespace(**config_dict['encoder_arguments'])
    exp_args = SimpleNamespace(**config_dict['exp_arguments'])
    heatmap_args = SimpleNamespace(**config_dict['heatmap_arguments'])
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    
    # 加载特征提取器
    feature_extractor, img_transforms = get_encoder(encoder_args.model_name, target_img_size=encoder_args.target_img_size)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # 创建临时保存目录
    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 定义分割掩码路径
    seg_mask_path = temp_dir / 'seg_mask.pkl'
    
    # 初始化WSI对象
    seg_params = config_dict['segmentation_arguments']
    filter_params = config_dict['filter_arguments']
    wsi_object = initialize_wsi(wsi_path, seg_mask_path=seg_mask_path, seg_params=seg_params, filter_params=filter_params)
    
    # 设置补丁大小和步长
    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    
    # 设置WSI处理参数
    wsi_kwargs = {
        'top_left': None, 
        'bot_right': None, 
        'patch_size': patch_size, 
        'step_size': step_size,
        'custom_downsample': patch_args.custom_downsample, 
        'level': patch_args.patch_level, 
        'use_center_shift': heatmap_args.use_center_shift
    }
    
    # 创建临时保存目录
    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 计算特征和热图
    with st.spinner("Processing WSI and generating heatmap..."):
        # 定义特征文件路径
        features_path = output_dir / f"{Path(wsi_path).stem}_features.pt"
        h5_path = output_dir / f"{Path(wsi_path).stem}_features.h5"
        
        if features_path.exists() and h5_path.exists():
            # 如果特征文件已存在，直接加载
            st.info("Loading pre-computed features...")
            features = torch.load(features_path)
            file = h5py.File(h5_path, "r")
            coords = file['coords'][:]
            file.close()
        else:
            # 如果特征文件不存在，计算并保存
            st.info("Computing features...")
            # 计算特征
            _, _, wsi_object = compute_from_patches(
                wsi_object=wsi_object,
                model=model,
                feature_extractor=feature_extractor,
                img_transforms=img_transforms,
                batch_size=config_dict['exp_arguments']['batch_size'],
                **wsi_kwargs,
                attn_save_path=None,
                feat_save_path=h5_path,
                ref_scores=None
            )
            
            # 保存特征
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:])
            torch.save(features, features_path)
            coords = file['coords'][:]
            file.close()

        # 加载特征到设备并进行预测
        features = features.to(device)
        
        # 模型预测
        with torch.no_grad():
            # 处理特征
            features2 = features.expand(1, -1, -1).float()
            features2 = model._fc1(features2)  # [B, n, 512]
            
            # 通过模型层处理
            with torch.no_grad():
                for layer in model.layers:
                    A_ = features2
                    A = layer[0](features2)
                    A = layer[1](A, rate=model.rate)
                    outputs = A + A_
                A = model.normA(A)
                for layer in model.layers2:
                    B_ = features2
                    B = layer[0](features2)
                    B = layer[1](B).squeeze(-1).permute(0, 2, 1)
                    B = B + B_
                B = model.normA(B)
                features2 = model.DFF(A.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1),
                                     B.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1).permute(0, 2, 1)
                A = model.classifier(features2)  # [B, n_classes]
                A = A[:, :, 0].unsqueeze(2)
            
            # 获取预测结果
            _, survival, Y_hat, A2, Y_prob = model(features)
            Y_hat = Y_hat.item()
            A = A.view(-1, 1).cpu().numpy()
            
            # 获取概率
            probs, ids = torch.topk(Y_prob, config_dict['exp_arguments']['n_classes'])
            Y_probs = probs[-1].cpu().numpy()
            Y_hats = ids[-1].cpu().numpy()
        
        # 计算风险
        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()[0]
        
        # 保存注意力分数
        block_map_save_path = output_dir / "blockmap.h5"
        asset_dict = {'attention_scores': A, 'coords': coords}
        save_hdf5(str(block_map_save_path), asset_dict, mode='w')
        
        # 生成热图
        file = h5py.File(block_map_save_path, 'r')
        scores = file['attention_scores'][:]
        coords = file['coords'][:]
        file.close()
        
        # 生成可视化热图
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_object.level_downsamples[patch_args.patch_level]) * 
                              patch_args.custom_downsample).astype(int))
        
        heatmap = drawHeatmap(
            scores, coords, wsi_path, wsi_object=wsi_object, 
            cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, 
            use_holes=True, binarize=False, vis_level=-1, 
            blank_canvas=False, thresh=-1, patch_size=vis_patch_size, 
            convert_to_percentiles=True
        )
        
        # 保存热图
        heatmap_path = output_dir / "heatmap.png"
        heatmap.save(heatmap_path)
        
        # 返回结果
        return {
            'wsi_object': wsi_object,
            'heatmap': heatmap,
            'heatmap_path': heatmap_path,
            'risk': risk,
            'Y_prob': Y_prob.cpu().numpy(),
            'Y_hat': Y_hat,
            'survival': survival.cpu().numpy()
        }

# 主程序逻辑
if upload_file is not None:
    # 显示原始WSI图像
    with raw_wsi_container:
        try:
            # 保存上传的图像
            wsi_path = temp_dir / upload_file.name
            with open(wsi_path, "wb") as f:
                f.write(upload_file.getbuffer())
            
            # 显示WSI缩略图
            wsi_object = WholeSlideImage(str(wsi_path))
            if str(wsi_path).endswith('.sdpc'):
                thumbnail = wsi_object.wsi.get_thumbnail((3))
            else:
                thumbnail = wsi_object.wsi.get_thumbnail((1000, 1000))
            st.image(thumbnail, caption="Raw WSI Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading WSI image: {str(e)}")
    
    # 处理提交
    if submit_button:
        try:
            # 处理WSI
            results = process_wsi(str(wsi_path), stage, age)
            
            # 显示风险热图
            with risk_map_container:
                st.image(results['heatmap'], caption="Risk Heatmap", use_container_width=True)
            
            # 计算分数和生存率
            #high_prob = results['Y_prob'][0][0]  # 高风险概率
            high_prob = results['risk']
            point1 = 25 * high_prob + 100
            point2 = 4.98 * stage - 4.98
            point3 = 0.140784869 * age - 4.223546065
            points = point1 + point2 + point3
            
            # 计算生存率
            def calculate_survival(points, time):
                if time == 1:
                    return (0.00000039 * points**3 - 0.000302657 * points**2 + 
                            0.033240274 * points - 0.043659898)
                elif time == 2:
                    return (0.000001702 * points**3 - 0.000513225 * points**2 + 
                            0.031320647 * points + 0.403112562)
                else:  # time == 3
                    return (0.000001702 * points**3 - 0.000437112 * points**2 + 
                            0.017152554 * points + 0.761625108)
            
            survival_1 = calculate_survival(points, 1)
            survival_2 = calculate_survival(points, 2)
            survival_3 = calculate_survival(points, 3)
            cph = joblib.load('cox_model.pkl') 
            train_data = pd.read_csv("9.1.dataset_train.csv")  
            train_data = train_data[["GPMA", "Stage", "Age", "survival", "censorship"]]  
            def predict_risk(X):
                # 返回预测的风险比分（线性预测器或部分风险值）
                return cph.predict_partial_hazard(X)
            # 从训练数据中选择特征数据（排除'time'和'event'）
            background = train_data.drop(['survival', 'censorship'], axis=1)

            # 创建 KernelExplainer 对象
            explainer = shap.KernelExplainer(predict_risk, background)

            new_patient = pd.DataFrame({
                'GPMA': [high_prob],
                'Stage': [stage],
                'Age': [age]
            })
            # 计算新患者的 SHAP 值
            new_shap_values = explainer.shap_values(new_patient)
            
            # 显示特征贡献度分析
            with feature_container:
                risk_group = "High Risk" if points > 83.52901 else "Low Risk"
                risk_class = "high-risk" if points > 83.52901 else "low-risk"
                
                st.markdown(f"<div class='nomogram-score {risk_class}'>GPMA Risk Group: {risk_group}</div>", unsafe_allow_html=True)
                
                # SHAP Analysis
                st.subheader("SHAP Analysis")
                
                # 创建水平条形图来显示SHAP值
                fig, ax = plt.subplots(figsize=(10, 4.8))
                feature_names = [f'GPMA={high_prob:.2f}', f'Stage={stage}', f'Age={age}']
                shap_values = new_shap_values[0]
                
                # 绘制条形图
                y_pos = np.arange(len(feature_names))
                colors = ['red' if x > 0 else 'blue' for x in shap_values]
                bars = plt.barh(y_pos, shap_values, color=colors)
                
                # 设置标签和标题
                plt.yticks(y_pos, feature_names)
                plt.xlabel('SHAP value (impact on model output)')
                
                # 添加基准线
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # 添加数值标签
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2,
                           f'{width:.3f}',
                           ha='left' if width > 0 else 'right',
                           va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # 显示生存率曲线
            with survival_container:
                st.markdown(f"<div class='nomogram-score'>Nomogram Score: {points:.1f}</div>", unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                times = [1, 2, 3]
                survivals = [survival_1, survival_2, survival_3]  # 5年生存率设为0作为示例
                
                plt.plot(times, survivals, 'ro-', linewidth=2, markersize=8)
                plt.fill_between(times, survivals, alpha=0.2, color='red')
                plt.xlabel('Time (years)')
                plt.ylabel('Survival Rate')
                plt.title('Survival Rates over Time')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1)
                
                # 添加数值标签
                for i, txt in enumerate(survivals):
                    plt.annotate(f'{txt:.2f}', 
                                (times[i], survivals[i]),
                                textcoords="offset points",
                                xytext=(0,10),
                                ha='center')
                
                st.pyplot(fig)
            
            # 显示Nomogram
            with nomogram_container:
                # 创建两列布局
                nom_col1, nom_col2 = st.columns(2)
                
                with nom_col1:
                    # SHAP重要性图
                    st.subheader("SHAP Importance")
                    
                    # 显示SHAP summary plot PNG
                    st.image('pic/shap_summary_plot.png', 
                            caption='SHAP Summary Plot',
                            use_container_width=True)
                
                with nom_col2:
                    # Nomogram
                    st.subheader("Nomogram")
                    
                    # 显示Nomogram PNG
                    st.image('pic/nomogram_classical.png', 
                            caption='Classical Nomogram',
                            use_container_width=True)
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# 添加页脚
st.markdown("---")
st.markdown("<div style='text-align: center;'>© 2025 Zhongnan Hospital. All rights reserved.</div>", unsafe_allow_html=True)