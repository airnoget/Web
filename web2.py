#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import shutil
import traceback
import time

import json
import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

import matplotlib

# reuse project modules
import toolkit_main as tkm
import toolkit_3D as tk3
from contact3D import calculate_3D_contact
from contact2D import calculate_2D_contact
import sam_segmenter
from matplotlib.patches import Patch

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    IMAGE_COORDS_AVAILABLE = True
except ImportError:
    st.error("错误：缺少 'streamlit-image-coordinates' 库。")
    st.error("请在您的环境中运行: pip install streamlit-image-coordinates")
    IMAGE_COORDS_AVAILABLE = False


# --- 3D Rendering Imports ---
import plotly.graph_objects as go
try:
    from skimage.measure import marching_cubes
    SKIMAGE_AVAILABLE = True
    _skimage_import_error = None
except ImportError as e:
    SKIMAGE_AVAILABLE = False
    _skimage_import_error = e

# --- Skeleton Analysis Import ---
try:
    from SkeletonAnalysis import skeleton_analysis, get_skeleton_img
    SKELETON_AVAILABLE = True
    _skeleton_import_error = None
except ImportError as e:
    SKELETON_AVAILABLE = False
    _skeleton_import_error = e

# Make tk3.get_nii tolerant to a 'rotate' keyword if the underlying function doesn't accept it.
# This avoids TypeError in places that call tk3.get_nii(..., rotate=True)
st.set_page_config(
    page_title="胰腺肿瘤可切除性分析",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)


def display_detailed_results(results):
    st.markdown("### 📊 详细分析结果")

    # 1. 显示 2D 分析结果
    if "2D" in results:
        st.markdown("#### 2D 接触分析")
        for target, data in results["2D"].items():
            if "error" in data:
                st.error(f"{target.capitalize()} 2D 分析失败: {data['error']}")
            else:
                st.write(f"- **{target.capitalize()}**")
                st.write(f"  - 最大接触比例: {data.get('max_ratio', 'N/A'):.2f}")
                st.write(f"  - 最大接触切片索引: {data.get('max_slice', 'N/A')}")

    # 2. 显示 3D 分析结果
    if "3D" in results:
        st.markdown("#### 3D 接触分析")
        for target, data in results["3D"].items():
            if "error" in data:
                st.error(f"{target.capitalize()} 3D 分析失败: {data['error']}")
            elif isinstance(data, list) and len(data) > 0:  # 检查是否为非空列表
                st.write(f"- **{target.capitalize()}**")
                st.write(f"  - 接触比例: {data[0].get('contact_ratio', 'N/A'):.2f}")
                st.write(f"  - 接触体积: {data[0].get('contact_volume', 'N/A')}")
            else:
                st.warning(f"{target.capitalize()} 3D 分析结果为空或格式无效")

    # 3. 显示骨架分析结果
    if "skeleton" in results:
        st.markdown("#### 骨架分析")
        for target, data in results["skeleton"].items():
            if "error" in data:
                st.error(f"{target.capitalize()} 骨架分析失败: {data['error']}")
            else:
                st.write(f"- **{target.capitalize()}**")
                st.write(f"  - 骨架长度: {data.get('length', 'N/A'):.2f}")
                st.write(f"  - 分支数量: {data.get('branches', 'N/A')}")

    # 4. 可视化结果（替换为热力图）
    if "3D" in results and isinstance(results["3D"], dict):
        st.markdown("#### 可视化结果")
        # 检查数据是否有效
        artery_data = results["3D"].get("artery", [])
        vein_data = results["3D"].get("vein", [])
        if len(artery_data) > 0 and len(vein_data) > 0:
            # 用热力图展示接触比例
            import plotly.express as px
            import pandas as pd

            data = {
                "血管类型": ["动脉", "静脉"],
                "接触比例": [
                    artery_data[0].get("contact_ratio", 0),
                    vein_data[0].get("contact_ratio", 0)
                ]
            }
            df = pd.DataFrame(data)

            fig = px.imshow(
                df.pivot_table(values="接触比例", index=None, columns="血管类型"),
                labels=dict(x="血管类型", y="", color="接触比例"),
                color_continuous_scale="Viridis",
                title="3D 接触比例热力图"
            )
            fig.update_layout(width=500, height=300)
            st.plotly_chart(fig)
        else:
            st.warning("3D 分析数据不足，无法生成可视化图表")





def display_score_card(score, label):
    st.markdown("### 切除性评估")
    score_color = "#FF4B4B" if score < 0.4 else ("#FFA500" if score < 0.7 else "#2ECC71")
    st.markdown(f"""
    <div style="border-left: 5px solid {score_color}; padding: 10px; background: #F8F9FA;">
        <p style="font-size: 16px; margin: 0;">评分: <span style="font-weight: bold; color: {score_color}; font-size: 24px;">{score:.2f}</span></p>
        <p style="font-size: 14px; margin: 0;">结论: {label}</p>
    </div>
    """, unsafe_allow_html=True)

try:
    _orig_get_nii = tk3.get_nii


    def _get_nii_compat(path, *args, **kwargs):
        if 'rotate' in kwargs:
            try:
                return _orig_get_nii(path, *args, **kwargs)
            except TypeError:
                # underlying implementation doesn't accept 'rotate'
                kwargs.pop('rotate')
                return _orig_get_nii(path, *args, **kwargs)
        else:
            return _orig_get_nii(path, *args, **kwargs)


    tk3.get_nii = _get_nii_compat
except Exception:
    # If monkeypatching fails for some reason, continue — safe_get_nii also handles compatibility.
    pass

# skeleton analysis may fail if kimimaro/cloud-volume incompatible
try:
    from SkeletonAnalysis import skeleton_analysis

    SKELETON_AVAILABLE = True
except Exception:
    SKELETON_AVAILABLE = False

# 标题与上传
st.title("🩺 胰腺肿瘤可切除性分析")
uploaded = st.file_uploader("📤 上传分割文件 (.nii)", type=["nii", "nii.gz"])
#
# st.markdown("上传分割好的 NIfTI (.nii 或 .nii.gz)，标签为：1=动脉，2=肿瘤，3=静脉。")
#
# uploaded = st.file_uploader("上传分割 NIfTI", type=["nii", "nii.gz"])


with st.sidebar:
    with st.expander("⚙️ 参数设置", expanded=True):
        contour_thickness = st.slider("轮廓厚度", 0.5, 5.0, 1.5)
        contact_range = st.slider("接触范围（体素）", 0, 10, 2)
        axis = st.selectbox("切片查看轴", ["z", "x"], index=0)
        do_2d = st.checkbox("运行 2D 接触分析", value=True)
    with st.expander("🔍 分析选项"):
        do_3d = st.checkbox("运行 3D 接触分析", value=True)
        do_skeleton = st.checkbox("运行骨架分析", value=False)
    if do_skeleton and not SKELETON_AVAILABLE:
        st.warning("无法导入 SkeletonAnalysis，在此环境中骨架分析将被跳过。")
        do_skeleton = False


def save_uploaded(tmpdir, uploaded_file):
    dest = os.path.join(tmpdir, uploaded_file.name)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def load_normalized_nii(path):
    """
    Load a NIfTI and return a 3D numpy array with shape (H, W, Z) (label volume).
    Handles:
      - channel-first one-hot e.g. (C, H, W) or (C, H, W, T) -> move channel to last, argmax
      - channel-last one-hot e.g. (H, W, C) -> argmax
      - 2D arrays -> expand to (H, W, 1)
      - already (H, W, Z) -> pass-through
    """
    img = nib.load(path)
    data = np.asarray(img.get_fdata())
    # 4D where first dim is small -> treat as channel-first
    if data.ndim == 4 and data.shape[0] <= 8:
        data = np.moveaxis(data, 0, -1)
    # 3D but one axis is small (<=8) and others large -> likely channel-first
    if data.ndim == 3:
        small_axes = [i for i, s in enumerate(data.shape) if s <= 8]
        if small_axes and max(data.shape) > 20:
            ch = small_axes[0]
            if ch != (data.ndim - 1):
                data = np.moveaxis(data, ch, -1)
    # If last axis looks like channels (<=8), convert one-hot/multi-channel to labels
    if data.ndim == 3 and data.shape[-1] <= 8:
        try:
            data = np.argmax(data, axis=-1)
        except Exception:
            data = np.squeeze(data)
    # If 2D -> expand to single-slice 3D
    if data.ndim == 2:
        data = data[..., np.newaxis]
    # final check/reshape
    data = np.asarray(data)
    if data.ndim == 3:
        return data
    data = np.squeeze(data)
    if data.ndim == 2:
        return data[..., np.newaxis]
    raise ValueError(f"Unsupported nifti shape after normalization: {data.shape}")


def display_slice_from_nii(nii_path, axis='z', slice_index=None, title=None):
    img = load_normalized_nii(nii_path)
    # rotate/assume orientation similar to toolkit_3D.get_nii
    if axis == 'z':
        depth = img.shape[2]
        if slice_index is None:
            slice_index = depth // 2
        slice_img = img[:, :, slice_index]
    else:
        depth = img.shape[0]
        if slice_index is None:
            slice_index = depth // 2
        slice_img = img[slice_index, :, :]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(slice_img.T, cmap='gray', origin='lower')
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis('off')
    return fig


def make_overlay_fig(origin_nii, contact_nii, axis='z', slice_index=None):
    oimg = load_normalized_nii(origin_nii)
    cimg = load_normalized_nii(contact_nii)
    if axis == 'z':
        if slice_index is None:
            slice_index = oimg.shape[2] // 2
        o = oimg[:, :, slice_index]
        c = cimg[:, :, slice_index]
    else:
        if slice_index is None:
            slice_index = oimg.shape[0] // 2
        o = oimg[slice_index, :, :]
        c = cimg[slice_index, :, :]

    # origin: show tumor in red, artery/vein contours and contacts with colors
    # Correctly create a transposed base image for display
    base = np.zeros((o.shape[1], o.shape[0], 3), dtype=np.float32)

    # Transpose masks before assigning to channels
    tumor_mask = (o == 2).T
    artery_contour = ((c == 2) | (c == 4)).T
    artery_contact = ((c == 3) | (c == 5)).T
    vein_contour = (c == 2).T  # fallback
    vein_contact = (c == 3).T

    base[..., 0] = tumor_mask.astype(float) * 0.8
    base[..., 1] = artery_contour.astype(float) * 0.6 + vein_contact.astype(float) * 0.4
    base[..., 2] = artery_contact.astype(float) * 0.8 + vein_contour.astype(float) * 0.3

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # Display the base image without incorrect transposition
    ax.imshow(base, origin='lower')
    ax.axis('off')
    return fig


def make_contact_only_fig(contact_nii, axis='z', slice_index=None, title=None):
    """
    Show the contact NIfTI slice in color-coded way:
      - background -> black
      - contour voxels -> yellow
      - contact voxels -> red
    This helps visualize where contacts are detected.
    """
    cimg = load_normalized_nii(contact_nii)
    if axis == 'z':
        if slice_index is None:
            slice_index = cimg.shape[2] // 2
        c = cimg[:, :, slice_index]
    else:
        if slice_index is None:
            slice_index = cimg.shape[0] // 2
        c = cimg[slice_index, :, :]

    # build rgb
    rgb = np.zeros((c.shape[1], c.shape[0], 3), dtype=np.float32)
    # Define simple masks (transpose for correct orientation)
    contour_mask = ((c == 2) | (c == 4)).T
    contact_mask = ((c == 3) | (c == 5)).T
    artery_mask = (c == 2).T
    vein_mask = (c == 3).T

    rgb[..., 0] = contact_mask.astype(float)  # red channel for contact
    rgb[..., 1] = contour_mask.astype(float) * 0.7  # green-ish for contour
    rgb[..., 2] = 0.0
    if title is None:
        title = "Contact Slice"

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(rgb, origin='lower')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    return fig


def safe_get_nii(path, rotate=True):
    """Call tk3.get_nii in a backward-compatible way: try keyword 'rotate' first,
    fall back to calling without keyword if function signature doesn't accept it.
    """
    try:
        return tk3.get_nii(path, rotate=rotate)
    except TypeError:
        # fallback to positional call if rotate kw not supported
        return tk3.get_nii(path)


def display_input_and_contact(origin_path, contact_path, axis='z', slice_index=None):
    """
    Display three images side-by-side:
      - Original segmentation slice (grayscale labels)
      - Contact-only visualization (colored)
      - Overlay (tumor + contacts)
    """
    cols = st.columns(3)
    # Original
    with cols[0]:
        try:
            fig_o = display_slice_from_nii(origin_path, axis=axis, slice_index=slice_index, title="原始分割切片")
            st.pyplot(fig_o)
        except Exception as e:
            st.error(f"无法展示原始切片: {e}")
    # Contact-only
    with cols[1]:
        try:
            fig_c = make_contact_only_fig(contact_path, axis=axis, slice_index=slice_index, title="接触掩码切片")
            st.pyplot(fig_c)
        except Exception as e:
            st.error(f"无法展示接触切片: {e}")
    # Overlay
    with cols[2]:
        try:
            fig_overlay = make_overlay_fig(origin_path, contact_path, axis=axis, slice_index=slice_index)
            st.pyplot(fig_overlay)
        except Exception as e:
            st.error(f"无法展示叠加视图: {e}")

def display_input_and_contact1(origin_path,axis='z', slice_index=None):
    """
    Display three images side-by-side:
      - Original segmentation slice (grayscale labels)
      - Contact-only visualization (colored)
      - Overlay (tumor + contacts)
    """
    cols = st.columns(1)
    # Original
    with cols[0]:
        try:
            fig_o = display_slice_from_nii(origin_path, axis=axis, slice_index=slice_index,
                                           title="原始分割切片")
            st.pyplot(fig_o)
        except Exception as e:
            st.error(f"无法展示原始切片: {e}")

st.header("上传文件")
uploaded1 = st.file_uploader("📤 上传 NIfTI 文件 (.nii/.nii.gz)", type=["nii", "nii.gz"])

if uploaded1 is not None:
    tmpdir = tempfile.mkdtemp(prefix="pancreas_demo_")
    try:
        # Use uploaded1 (matches the if-check) when saving
        saved = save_uploaded(tmpdir, uploaded1)
        st.success(f"已将上传文件保存到 {saved}")

        file_dict = {
            "img_id": os.path.splitext(os.path.basename(saved))[0],
            "img_path": saved,
            "img_contact_path": None
        }
        contact_dir = tmpdir

        # --- Progress Bar Setup ---
        total_steps = 1 + do_2d + do_3d + do_skeleton  # 1 for contact generation
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_step = 0

        def update_progress(step_name):
            # At module/script top level use global, not nonlocal
            global current_step
            current_step += 1
            progress = current_step / max(1, total_steps)
            progress_bar.progress(progress)
            status_text.text(f"步骤 {current_step}/{total_steps}: {step_name}")

        # mark progress complete for this quick path
        progress_bar.progress(1.0)

        # --- Post-processing and Results ---
        # tumor info
        try:
            origin_struct = None
            try:
                origin_struct = safe_get_nii(file_dict["img_path"], rotate=True)
            except Exception:
                origin_struct = None
            if isinstance(origin_struct, dict) and "tumor" in origin_struct:
                tumor_mask = origin_struct["tumor"]
                tumor_volume = int(np.sum(np.asarray(tumor_mask) > 0))
            else:
                lbl = load_normalized_nii(file_dict["img_path"])
                tumor_volume = int(np.sum(lbl == 2))
        except Exception:
            tumor_volume = None

        # --- Visualization (with corrected indentation) ---
        st.header("可视化")

        # Load image once to determine slice ranges and default
        try:
            oimg = load_normalized_nii(file_dict["img_path"])
            if axis == 'z':
                max_slice = max(0, oimg.shape[2] - 1)
                default_slice = oimg.shape[2] // 2
            else:  # axis == 'x'
                max_slice = max(0, oimg.shape[0] - 1)
                default_slice = oimg.shape[0] // 2
        except Exception:
            # Fallback values if load fails
            max_slice = 0
            default_slice = 0

        # Provide a slider so user can choose the slice to preview
        slice_index = st.slider(
            "选择切片索引",
            min_value=0,
            max_value=max_slice,
            value=min(default_slice, max_slice)
        )

        # 显示折叠区域：在 expander 中再次展示原始切片（或组合视图）
        with st.expander("📂 点击查看上传的图像", expanded=False):
            try:
                # If contact exists, show the three-panel view; otherwise show original only
                if file_dict.get("img_contact_path") and os.path.exists(file_dict["img_contact_path"]):
                    display_input_and_contact(
                        file_dict["img_path"],
                        file_dict["img_contact_path"],
                        axis=axis,
                        slice_index=slice_index
                    )
                else:
                    fig_o = display_slice_from_nii(file_dict["img_path"], axis=axis, slice_index=slice_index, title="原始分割切片")
                    st.pyplot(fig_o)
            except Exception as e:
                st.error(f"无法展示上传的图像: {e}")

    except Exception as e:
        # 捕获 try 块中未处理的异常并显示
        st.error(f"处理上传文件时发生错误: {e}")
        st.text(traceback.format_exc())
    finally:
        # 清理临时目录（根据你的逻辑，你也可以继续使用 checkbox 控制删除）
        if st.checkbox("现在清理临时文件", value=False, key="cleanup_temp_files_1"):
            try:
                shutil.rmtree(tmpdir)
                st.success("临时文件已删除")
            except Exception as e:
                st.error("删除临时目录失败: " + str(e))

if uploaded is not None:
    tmpdir = tempfile.mkdtemp(prefix="pancreas_demo_")
    try:
        saved = save_uploaded(tmpdir, uploaded)
        st.success(f"已将上传文件保存到 {saved}")

        file_dict = {
            "img_id": os.path.splitext(os.path.basename(saved))[0],
            "img_path": saved,
            "img_contact_path": None
        }
        contact_dir = tmpdir

        # --- Progress Bar Setup ---
        total_steps = 1 + do_2d + do_3d + do_skeleton  # 1 for contact generation
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_step = 0


        def update_progress(step_name):
            global current_step
            current_step += 1
            progress = current_step / total_steps
            progress_bar.progress(progress)
            status_text.text(f"步骤 {current_step}/{total_steps}: {step_name}")


        # --- Step 1: Generate Contact Image ---
        update_progress("正在生成接触面图像...")
        with st.spinner("生成接触面中..."):
            try:
                tkm.generate_contour_nii_3D(file_dict, contact_dir, prefix="contact_",
                                            contour_thickness=contour_thickness,
                                            contact_range=contact_range, axis=axis)
            except Exception as e:
                st.error("生成接触面失败: " + str(e))
                st.text(traceback.format_exc())
                st.stop()  # Stop execution if this critical step fails

        st.write("接触 NIfTI:", file_dict.get("img_contact_path"))

        results = {"2D": {}, "3D": {}, "skeleton": {}}

        # --- Step 2: 2D Analysis ---
        if do_2d:
            update_progress("正在执行 2D 接触分析...")
            with st.spinner("⏳ 2D 分析中..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress_bar.progress(i + 1)
                for target in ["vein", "artery"]:
                    try:
                        slice_list, max_ratio, max_slice = tkm.calculate_2D_contact(file_dict, target,
                                                                                    contact_dir=contact_dir,
                                                                                    size_threshold=30, axis=axis)
                        results["2D"][target] = {"slices": slice_list, "max_ratio": float(max_ratio),
                                                 "max_slice": int(max_slice)}
                    except Exception as e:
                        results["2D"][target] = {"error": str(e)}
                        st.error(f"2D 分析失败（{target}）：{e}")

        # --- Step 3: 3D Analysis ---
        if do_3d:
            update_progress("正在执行 3D 接触分析...")
            with st.spinner("⏳ 3D 分析中..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress_bar.progress(i + 1)
                for target in ["vein", "artery"]:
                    try:
                        res3 = calculate_3D_contact(file_dict, target)
                        results["3D"][target] = res3
                    except Exception as e:
                        results["3D"][target] = {"error": str(e)}
                        st.error(f"3D 分析失败（{target}）：{e}")

        # --- Step 4: Skeleton Analysis ---
        if do_skeleton:
            update_progress("正在执行骨架分析...")
            with st.spinner("⏳ 骨架化中..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress_bar.progress(i + 1)
                for target in ["vein", "artery"]:
                    try:
                        skeleton_res = skeleton_analysis(file_dict, target, print_info=False)
                        results["skeleton"][target] = skeleton_res
                    except Exception as e:
                        results["skeleton"][target] = {"error": str(e)}
                        st.error(f"骨架分析失败（{target}）：{e}")

        status_text.text("✅所有分析步骤已完成！")
        progress_bar.progress(1.0)

        # --- Post-processing and Results ---
        # tumor info
        try:
            origin_struct = None
            try:
                origin_struct = safe_get_nii(file_dict["img_path"], rotate=True)
            except Exception:
                origin_struct = None
            if isinstance(origin_struct, dict) and "tumor" in origin_struct:
                tumor_mask = origin_struct["tumor"]
                tumor_volume = int(np.sum(np.asarray(tumor_mask) > 0))
            else:
                lbl = load_normalized_nii(file_dict["img_path"])
                tumor_volume = int(np.sum(lbl == 2))
        except Exception:
            tumor_volume = None

        # simple resectability scoring rule (example)
        score = 0.5
        if "artery" in results["3D"] and isinstance(results["3D"]["artery"], list) and len(results["3D"]["artery"]) > 0:
            c3a = results["3D"]["artery"][0].get("contact_ratio", 0)
            score -= 0.3 * c3a
        if "vein" in results["3D"] and isinstance(results["3D"]["vein"], list) and len(results["3D"]["vein"]) > 0:
            c3v = results["3D"]["vein"][0].get("contact_ratio", 0)
            score -= 0.15 * c3v
        if tumor_volume is not None:
            score -= 0.05 * np.log1p(tumor_volume)
        score = float(max(0.0, min(1.0, score)))

        if score > 0.7:
            label = "可能可切除"
        elif score > 0.4:
            label = "边界性"
        else:
            label = "可能不可切除"

        st.header("结果摘要")
        st.write("切除性评分：", score, "（", label, "）")

        # 显示结果
        col1, col2 = st.columns([1, 2])
        with col1:
            display_score_card(score, label)
        with col2:
            with st.expander("📊 详细分析结果", expanded=True):
                display_detailed_results(results)
#        st.json(results)

        # --- Visualization (with corrected indentation) ---
        st.header("可视化")
        if file_dict.get("img_contact_path") and os.path.exists(file_dict["img_contact_path"]):
            oimg = load_normalized_nii(file_dict["img_path"])
            if axis == 'z':
                max_slice = oimg.shape[2] - 1
                default_slice = oimg.shape[2] // 2
            else:  # axis == 'x'
                max_slice = oimg.shape[0] - 1
                default_slice = oimg.shape[0] // 2

            # Ensure the slider's default value doesn't exceed the max slice index
            slice_index = st.slider(
                "选择切片索引",
                min_value=0,
                max_value=max_slice,
                value=min(default_slice, max_slice)
            )

            # New: display original slice, contact-only slice, and overlay side-by-side
            display_input_and_contact(file_dict["img_path"], file_dict["img_contact_path"], axis=axis,
                                      slice_index=slice_index)
        else:
            st.info("未生成用于可视化的接触 NIfTI。")

    finally:
        if st.checkbox("现在清理临时文件", value=False, key="cleanup_temp_files_2"):
            try:
                shutil.rmtree(tmpdir)
                st.success("临时文件已删除")
            except Exception as e:
                st.error("删除临时目录失败: " + str(e))

else:
    st.info("请上传分割好的 NIfTI 以开始分析。")

# 可视化分栏（页面底部也显示一次，方便快速查看）
if uploaded:
    # Protect against missing variables when user hasn't finished processing
    try:
        if file_dict.get("img_contact_path") and os.path.exists(file_dict["img_contact_path"]):
            oimg = load_normalized_nii(file_dict["img_path"])
            if axis == 'z':
                max_slice = oimg.shape[2] - 1
                default_slice = oimg.shape[2] // 2
            else:
                max_slice = oimg.shape[0] - 1
                default_slice = oimg.shape[0] // 2

            # keep previous chosen slice if exists, else default
            try:
                current_slice = slice_index
            except NameError:
                current_slice = min(default_slice, max_slice)

            st.markdown("---")
            st.subheader("切片快速预览")
            display_input_and_contact(file_dict["img_path"], file_dict["img_contact_path"], axis=axis,
                                      slice_index=current_slice)

            st.markdown("### 评分卡")
            st.metric("切除性评分", f"{score:.2f}", label)
    except Exception:
        # If anything not available yet, silently ignore (already shown above)
        pass
def display_resectability_recommendation(results):
    st.header("可切除性评估建议 (Resectability Assessment)")

    artery_contact_ratio, vein_contact_ratio = 0.0, 0.0
    if "3D" in results and results["3D"]:
        if "artery" in results["3D"] and isinstance(results["3D"]["artery"], list) and results["3D"]["artery"]:
            artery_contact_ratio = max([seg.get("contact_ratio", 0) for seg in results["3D"]["artery"]])
        if "vein" in results["3D"] and isinstance(results["3D"]["vein"], list) and results["3D"]["vein"]:
            vein_contact_ratio = max([seg.get("contact_ratio", 0) for seg in results["3D"]["vein"]])

    UNRESECTABLE_ARTERY_THRESHOLD, BORDERLINE_VEIN_THRESHOLD = 0.5, 0.5
    recommendation, reasons = "🟢 **可切除 (Resectable)**", []

    if artery_contact_ratio > UNRESECTABLE_ARTERY_THRESHOLD:
        recommendation = "🔴 **不可切除 (Unresectable)**"
        reasons.append(f"**主要动脉包裹**: 肿瘤与动脉的最大接触比例为 **{artery_contact_ratio:.2%}**，超过了 180° 包裹的阈值 ({UNRESECTABLE_ARTERY_THRESHOLD:.0%})。")
    elif vein_contact_ratio > BORDERLINE_VEIN_THRESHOLD:
        recommendation = "🟡 **交界可切除 (Borderline Resectable)**"
        reasons.append(f"**主要静脉包裹**: 肿瘤与静脉的最大接触比例为 **{vein_contact_ratio:.2%}**，超过了 180° 包裹的阈值 ({BORDERLINE_VEIN_THRESHOLD:.0%})。")
    elif artery_contact_ratio > 0:
        recommendation = "🟡 **交界可切除 (Borderline Resectable)**"
        reasons.append(f"**动脉邻接**: 肿瘤与动脉存在接触（最大比例 **{artery_contact_ratio:.2%}**），但未达到完全包裹的程度。")
    else:
        reasons.append("肿瘤与主要动脉无接触，且与主要静脉的接触未达到完全包裹的程度，具备良好的手术切除条件。")

    st.markdown(f"### 评估结果: {recommendation}")
    with st.container():
        st.markdown("**评估依据:**")
        for r in reasons: st.markdown(f"- {r}")
        st.markdown(f"**关键参数:**")
        st.markdown(f"  - **动脉最大接触比例**: `{artery_contact_ratio:.2%}`")
        st.markdown(f"  - **静脉最大接触比例**: `{vein_contact_ratio:.2%}`")
        st.caption("注：该建议基于 3D 接触比例。此结果仅供参考。")

@st.cache_data(show_spinner=False)
def perform_full_analysis(_uploaded_file_bytes, _file_name, _contour_thickness, _contact_range, _axis, _do_2d, _do_3d, _do_skeleton, _raw_ct_bytes=None):
    # ... (此函数内容保持不变) ...
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, _file_name)
        with open(file_path, "wb") as f: f.write(_uploaded_file_bytes)
        file_dict = {"img_id": os.path.splitext(_file_name)[0], "img_path": file_path, "img_contact_path": None}

        total_steps = 1 + (1 if _do_2d else 0) + (1 if _do_3d else 0) + (1 if _do_skeleton else 0)
        progress_bar = st.progress(0, text="正在初始化分析...")

        def update_progress(step, name):
            progress = step / total_steps
            progress_bar.progress(progress, text=f"步骤 {step}/{total_steps}: {name}")

        update_progress(1, "正在生成接触面图像...")

        label_map_dict_data = tk3.get_nii(file_dict["img_path"], axis=_axis)

        raw_ct_data = None
        if _raw_ct_bytes:
            raw_path = os.path.join(tmpdir, "raw_ct.nii.gz")
            with open(raw_path, "wb") as f: f.write(_raw_ct_bytes)
            raw_ct_data = tk3.get_any_nii(raw_path, axis=_axis)['img']

        contact_img_data = None
        contact_map_path = file_dict.get("img_contact_path")
        if contact_map_path and os.path.exists(contact_map_path):
            try:
                contact_img_data = tk3.get_any_nii(contact_map_path, axis=_axis)['img']
            except Exception as e:
                st.warning(f"无法加载接触面图像: {e}")

        progress_bar.progress(1.0, text="分析完成！")
        return results, label_map_dict_data, raw_ct_data, artery_skeleton, vein_skeleton, contact_img_data

def save_uploaded_file(uploaded_file, directory):
    try:
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue()) # Use getvalue() for BytesIO
        return file_path
    except Exception as e:
        st.error(f"保存文件时出错: {e}")
        return None


def make_organ_overlay_fig(label_map_dict, raw_ct_data, axis='z', slice_index=None):
    if axis == 'z':
        slice_idx = slice_index if slice_index is not None else label_map_dict["origin"].shape[2] // 2
        labels = label_map_dict["origin"][:, :, slice_idx]
        raw_slice = raw_ct_data[:, :, slice_idx] if raw_ct_data is not None else None
    else:
        slice_idx = slice_index if slice_index is not None else label_map_dict["origin"].shape[0] // 2
        labels = label_map_dict["origin"][slice_idx, :, :]
        raw_slice = raw_ct_data[slice_idx, :, :] if raw_ct_data is not None else None

    fig, ax = plt.subplots(figsize=(10, 10))

    if raw_slice is not None:
        window_center, window_width = 40, 40
        vmin, vmax = window_center - window_width / 2, window_center + window_width / 2
        ax.imshow(raw_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    else:
        ax.imshow(np.zeros_like(labels.T), cmap='gray', origin='lower')

    colors = {'artery':(1,0,0,.6),'vein':(0,0,1,.6),'tumor':(0,1,0,.6),'pancreas':(1,1,0,.5)}

    for organ, label_val in [('pancreas', 4), ('artery', 1), ('vein', 3), ('tumor', 2)]:
        # Check existence using .get for label_map_dict keys
        # For label_val != 4, we assume they might exist in origin
        if (organ == 'pancreas' and label_map_dict.get(organ) is not None) or \
           (label_val != 4 and label_map_dict.get("origin") is not None):
             try:
                 mask = (labels == label_val).T
                 colored_mask = np.zeros(mask.shape + (4,)); colored_mask[mask] = colors[organ]
                 ax.imshow(colored_mask, origin='lower', interpolation='none')
             except IndexError: # Handle case where slice_idx might be out of bounds somehow
                 pass


    ax.set_aspect('equal')

    legend_elements = [Patch(facecolor=c, label=f"{n} ({l})") for n, c, l in [('动脉', 'red', 'Arteries'), ('静脉', 'blue', 'Veins'), ('肿瘤', 'green', 'Tumor'), ('胰腺', 'yellow', 'Pancreas')]]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_title(f"器官分割透视图 - 切片 {slice_idx} (轴={axis})"); ax.axis('off')
    return fig

def make_contact_overlay_fig(origin_data, contact_data, axis='z', slice_index=None):
    if axis == 'z':
        slice_idx = slice_index if slice_index is not None else origin_data.shape[2] // 2
        slice_idx = min(slice_idx, origin_data.shape[2] - 1)
        slice_idx = min(slice_idx, contact_data.shape[2] - 1) if contact_data is not None else slice_idx # Handle None contact_data
        o = origin_data[:, :, slice_idx]
        c = contact_data[:, :, slice_idx] if contact_data is not None else np.zeros_like(o) # Handle None contact_data
    else:
        slice_idx = slice_index if slice_index is not None else origin_data.shape[0] // 2
        slice_idx = min(slice_idx, origin_data.shape[0] - 1)
        slice_idx = min(slice_idx, contact_data.shape[0] - 1) if contact_data is not None else slice_idx
        o = origin_data[slice_idx, :, :]
        c = contact_data[slice_idx, :, :] if contact_data is not None else np.zeros_like(o)


    base = np.zeros((o.shape[1], o.shape[0], 3), dtype=np.float32)

    tumor_mask = (o == 2).T
    artery_contour = ((c == 2) | (c == 4)).T
    artery_contact = ((c == 3) | (c == 5)).T
    vein_contour = (c == 2).T # Note: Same label as artery contour in original logic
    vein_contact = (c == 3).T # Note: Same label as artery contact in original logic

    # Consider adjusting the logic if artery/vein contact/contour labels differ
    base[..., 0] = tumor_mask.astype(float) * 0.8
    base[..., 1] = artery_contour.astype(float) * 0.6 + vein_contact.astype(float) * 0.4
    base[..., 2] = artery_contact.astype(float) * 0.8 + vein_contour.astype(float) * 0.3

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(base, origin='lower')
    ax.set_aspect('equal')

    legend_elements = [
        Patch(facecolor=(0.8, 0, 0), label='肿瘤 (Tumor)'),
        Patch(facecolor=(0, 0.6, 0.3), label='动脉/静脉轮廓 (Vessel Contour)'),
        Patch(facecolor=(0, 0.4, 0.8), label='动脉/静脉接触 (Vessel Contact)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_title(f"接触面透视图 - 切片 {slice_idx} (轴={axis})")
    ax.axis('off')
    return fig
def make_skeleton_fig(skeleton_img_3d, title, axis='z', slice_index=None):
    # Ensure slice_index is valid
    if axis == 'z':
        if slice_index is None: slice_index = skeleton_img_3d.shape[2] // 2
        slice_index = min(max(0, slice_index), skeleton_img_3d.shape[2] - 1) # Clamp index
        slice_data = skeleton_img_3d[:, :, slice_index]
    else: # axis == 'x'
        if slice_index is None: slice_index = skeleton_img_3d.shape[0] // 2
        slice_index = min(max(0, slice_index), skeleton_img_3d.shape[0] - 1) # Clamp index
        slice_data = skeleton_img_3d[slice_index, :, :]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(slice_data.T, cmap='hot', origin='lower')
    ax.set_aspect('equal')
    ax.set_title(f"{title} - 切片 {slice_index} (轴={axis})"); ax.axis('off')
    return fig

@st.cache_data(show_spinner="正在生成 3D 模型...")
def make_3d_surface_plot(_label_data_array):
    if not SKIMAGE_AVAILABLE:
        st.error("无法创建 3D 视图：缺少 `scikit-image` 库。请运行 `pip install scikit-image`。")
        return go.Figure()

    plot_data = []
    organ_defs = {
        '动脉 (Artery)':   {'label': 1, 'color': 'red',     'opacity': 1.0},
        '肿瘤 (Tumor)':    {'label': 2, 'color': 'green',   'opacity': 0.5},
        '静脉 (Vein)':     {'label': 3, 'color': 'blue',    'opacity': 1.0},
        '胰腺 (Pancreas)': {'label': 4, 'color': 'yellow',  'opacity': 0.4}
    }

    for organ, props in organ_defs.items():
        label_val = props['label']
        if np.any(_label_data_array == label_val):
            try:
                verts, faces, _, _ = marching_cubes(
                    _label_data_array == label_val,
                    level=0.5,
                    spacing=(1.0, 1.0, 1.0)
                )
                plot_data.append(
                    go.Mesh3d(
                        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                        color=props['color'],
                        opacity=props['opacity'],
                        name=organ
                    )
                )
            except Exception as e:
                st.warning(f"为 {organ} 生成 3D 网格时出错: {e}")

    if not plot_data:
        st.warning("在 NIfTI 文件中未找到可渲染的标签 (1, 2, 3, 4)。")
        return go.Figure()

    fig = go.Figure(data=plot_data)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="交互式 3D 模型 (可拖动旋转)"
    )
    return fig
with st.sidebar:
    st.header("工作模式"); mode = st.radio("选择分割方式", ('上传已分割文件', '使用 SAM 2 分割'))

    do_3d_render = st.checkbox("交互式 3D 渲染", value=True)
    if do_3d_render and not SKIMAGE_AVAILABLE:
        st.warning("`scikit-image` 库未安装。3D 渲染将不可用。")
        do_3d_render = False

if mode == '上传已分割文件':

    st.header("上传已分割的 NIfTI 文件")
    st.markdown("标签定义: 1=动脉, 2=肿瘤, 3=静脉。(可选: 4=胰腺)")
    uploaded_file = st.file_uploader("上传分割文件", type=["nii", "nii.gz"])

    if uploaded_file:
        # Get bytes once
        uploaded_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        results, label_map_dict, raw_ct_data, artery_skeleton, vein_skeleton, contact_img_data = perform_full_analysis(
            uploaded_bytes,
            file_name,
            contour_thickness,
            contact_range,
            axis,
            do_2d,
            do_3d,
            do_skeleton,
            _raw_ct_bytes=None # No raw CT needed unless user uploads it separately
        )
        display_resectability_recommendation(results)

        if do_3d_render:
            if label_map_dict and label_map_dict.get("origin") is not None:
                st.header("交互式 3D 可视化")
                st.info("您可以拖动、旋转和缩放 3D 模型。")
                fig_3d = make_3d_surface_plot(label_map_dict["origin"])
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("无法生成 3D 视图：未加载标签数据。")

# ==============================================================================
# --- 模式 2：使用 SAM 2 进行交互式分割 ---
# ==============================================================================
elif mode == '使用 SAM 2 分割':
    st.header("使用 SAM 2 进行交互式分割")
    st.markdown("请上传一个**原始的、未分割的**医学影像文件 (例如 CT 平扫期)。")

    # 检查依赖库
    if not IMAGE_COORDS_AVAILABLE:
        st.error("缺少 'streamlit-image-coordinates' 库，无法进行交互式点击。请先安装。")
        st.stop()  # 停止执行此模式

    raw_file = st.file_uploader("上传原始影像文件", type=["nii", "nii.gz"], key="sam_raw_uploader")

    # --- 初始化 Session State ---
    # (确保这些只在第一次加载或文件更改时初始化)
    if 'sam_raw_file_id' not in st.session_state or (
            raw_file and raw_file.file_id != st.session_state.get('sam_raw_file_id')):
        st.session_state.masks = {'artery': [], 'tumor': [], 'vein': []}  # 分类存储掩码
        st.session_state.points = []  # 当前切片上的点 [(x, y), ...]
        st.session_state.labels = []  # 当前切片上点的标签 [1, 0, ...] (1=前景, 0=背景)
        st.session_state.current_click_coords = None  # 存储最近一次点击的坐标
        st.session_state.raw_img_data = None
        st.session_state.raw_img_nii = None
        st.session_state.normalized_slices = {}  # 缓存归一化后的切片
        st.session_state.sam_raw_file_id = raw_file.file_id if raw_file else None
        st.session_state.analysis_complete = False  # 标记分析是否完成

    if raw_file and st.session_state.raw_img_data is None:
        # 只在第一次上传或更换文件时加载
        with st.spinner("正在加载原始 CT 影像..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                raw_path = save_uploaded_file(raw_file, tmpdir)
                try:
                    st.session_state.raw_img_nii = nib.load(raw_path)
                    st.session_state.raw_img_data = st.session_state.raw_img_nii.get_fdata().astype(np.float32)
                    st.session_state.normalized_slices = {}  # 清空缓存
                    st.rerun()  # 重新运行以更新界面状态
                except Exception as e:
                    st.error(f"加载 NIfTI 文件失败: {e}")
                    st.session_state.raw_img_data = None  # 标记加载失败

    if st.session_state.raw_img_data is not None and not st.session_state.analysis_complete:
        raw_img_data = st.session_state.raw_img_data
        H, W, Z = raw_img_data.shape

        col1, col2 = st.columns([3, 1])  # 图像列更宽

        with col1:
            st.subheader("图像交互区域")
            slice_idx = st.slider("选择要标注的切片", 0, Z - 1, Z // 2, key="sam_slice_slider")

            # --- 缓存和获取当前切片 (归一化到 uint8 用于显示) ---
            if slice_idx not in st.session_state.normalized_slices:
                current_slice_raw = raw_img_data[:, :, slice_idx]
                # 使用与 DODnet 类似的窗口化，然后归一化到 0-255
                slice_normalized = np.clip(current_slice_raw, -100, 400)
                min_norm, max_norm = np.min(slice_normalized), np.max(slice_normalized)
                if max_norm > min_norm:
                    slice_uint8 = ((slice_normalized - min_norm) / (max_norm - min_norm) * 255).astype(np.uint8)
                else:
                    slice_uint8 = np.zeros_like(current_slice_raw, dtype=np.uint8)
                st.session_state.normalized_slices[slice_idx] = slice_uint8
            current_slice_uint8 = st.session_state.normalized_slices[slice_idx]
            # --- 结束切片处理 ---

            # --- 使用 streamlit-image-coordinates 进行点击交互 ---
            st.write("在下方图像上点击选择点：")
            value = streamlit_image_coordinates(current_slice_uint8, key="sam_image_click")

            # 如果用户点击了图像，记录坐标
            if value is not None and value != st.session_state.get("_last_click_value_ref"):  # 避免重复添加同一点
                coords = (value["x"], value["y"])
                st.session_state.current_click_coords = coords
                st.session_state._last_click_value_ref = value  # 存储引用以防重复点击
                st.info(f"已选择点: ({coords[0]}, {coords[1]})。请点击下方按钮确认前景或背景。")
                # 不需要 st.rerun()，让用户点击按钮来确认

            # --- 使用 Matplotlib 显示带有点的图像 ---

            # 绘制已确认的点

            # --- 结束 Matplotlib 显示 ---

        with col2:
            st.subheader("标注工具")
            structure_to_label = st.radio("选择要标注的结构", ('肿瘤', '动脉', '静脉'), key="sam_structure_radio")
            label_map = {'肿瘤': 'tumor', '动脉': 'artery', '静脉': 'vein'}

            # --- 添加点按钮 ---
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("添加前景点 (+)", key="sam_add_fg"):
                    if st.session_state.current_click_coords:
                        st.session_state.points.append(st.session_state.current_click_coords)
                        st.session_state.labels.append(1)
                        st.session_state.current_click_coords = None  # 清除待确认点
                        st.session_state._last_click_value_ref = None  # 允许下一次点击
                        st.rerun()
                    else:
                        st.warning("请先在图像上点击选择一个点。")
            with col_btn2:
                if st.button("添加背景点 (-)", key="sam_add_bg"):
                    if st.session_state.current_click_coords:
                        st.session_state.points.append(st.session_state.current_click_coords)
                        st.session_state.labels.append(0)  # 背景点标签为 0
                        st.session_state.current_click_coords = None
                        st.session_state._last_click_value_ref = None
                        st.rerun()
                    else:
                        st.warning("请先在图像上点击选择一个点。")

            # --- 其他控制按钮 ---
            if st.button("清除当前切片所有点", key="sam_clear_points"):
                st.session_state.points = []
                st.session_state.labels = []
                st.session_state.current_click_coords = None
                st.session_state._last_click_value_ref = None
                st.rerun()

            if st.button("运行 SAM 分割当前切片", key="sam_run_slice_seg"):
                if not st.session_state.points:
                    st.warning("请至少添加一个前景或背景点。")
                else:
                    with st.spinner("SAM 正在分割当前切片..."):
                        predictor = sam_segmenter.load_sam2_model()  # 确保模型已加载
                        if predictor:
                            # 传递归一化的 uint8 图像给 SAM
                            current_slice_for_sam = st.session_state.normalized_slices[slice_idx]

                            mask = sam_segmenter.run_sam2_prediction(
                                predictor,
                                current_slice_for_sam,
                                st.session_state.points,
                                st.session_state.labels
                            )

                            if mask is not None:
                                target_key = label_map[structure_to_label]
                                # 创建一个与 3D 图像相同大小的 False 掩码
                                full_mask = np.zeros_like(raw_img_data, dtype=bool)
                                # 将当前切片的 2D 掩码放入 3D 掩码
                                full_mask[:, :, slice_idx] = mask
                                # 将这个 3D 掩码添加到对应结构的列表中
                                st.session_state.masks[target_key].append(full_mask)

                                st.success(f"已为“{structure_to_label}”添加一个掩码 (来自切片 {slice_idx})。")
                                # 分割成功后清除当前切片的点
                                st.session_state.points = []
                                st.session_state.labels = []
                                st.session_state.current_click_coords = None
                                st.session_state._last_click_value_ref = None
                                st.rerun()
                            else:
                                st.error("SAM 分割失败，请检查点或图像。")
                        else:
                            st.error("SAM 模型加载失败，无法进行分割。")

        # --- 显示已完成的掩码数量 ---
        st.markdown("---")
        st.subheader("已完成分割")
        st.write("已添加的 3D 掩码数量:")
        st.write(f"- 肿瘤: {len(st.session_state.masks['tumor'])} 个")
        st.write(f"- 动脉: {len(st.session_state.masks['artery'])} 个")
        st.write(f"- 静脉: {len(st.session_state.masks['vein'])} 个")
        st.caption("每个掩码代表一次成功的切片分割。")

        # --- 完成与分析按钮 ---
        if st.button("完成所有分割，合并掩码并开始分析", key="sam_finalize"):
            if not any(st.session_state.masks.values()):
                st.error("请至少完成一个结构的分割。")
            else:
                with st.spinner("正在合并所有分割掩码..."):
                    # 合并同一结构的所有 3D 掩码
                    final_3d_mask = np.zeros_like(raw_img_data, dtype=np.uint8)
                    for target, label_val in [('artery', 1), ('tumor', 2), ('vein', 3)]:
                        combined_mask_for_target = np.zeros_like(raw_img_data, dtype=bool)
                        for mask_3d in st.session_state.masks[target]:
                            combined_mask_for_target = np.logical_or(combined_mask_for_target, mask_3d)
                        final_3d_mask[combined_mask_for_target] = label_val

                with tempfile.TemporaryDirectory() as tmpdir:
                    # 保存合并后的掩码为 NIfTI
                    sam_nifti_path = os.path.join(tmpdir, "sam_merged_segmentation.nii.gz")
                    # 使用原始 NII 的仿射矩阵和头信息
                    refined_nii = nib.Nifti1Image(final_3d_mask.astype(np.uint8), st.session_state.raw_img_nii.affine,
                                                  st.session_state.raw_img_nii.header)
                    nib.save(refined_nii, sam_nifti_path)

                    with open(sam_nifti_path, "rb") as f:
                        sam_nifti_bytes = f.read()

                    st.success("掩码合并完成，准备进行后续分析。")

                    # --- 调用分析函数 ---
                    st.info("正在运行接触分析和可视化...")
                    # 获取侧边栏参数
                    contour_thickness = st.session_state.get('contour_thickness', 1.5)  # 使用 st.session_state 获取侧边栏值
                    contact_range = st.session_state.get('contact_range', 2)
                    axis = st.session_state.get('axis', 'z')
                    do_2d = st.session_state.get('do_2d', True)
                    do_3d = st.session_state.get('do_3d', True)
                    do_skeleton = st.session_state.get('do_skeleton', True)
                    do_3d_render = st.session_state.get('do_3d_render', True)

                    results, label_map_dict, raw_ct_data_final, artery_skeleton, vein_skeleton, contact_img_data = perform_full_analysis(
                        sam_nifti_bytes,
                        "sam_merged_segmentation.nii.gz",
                        contour_thickness,
                        contact_range,
                        axis,
                        do_2d,
                        do_3d,
                        do_skeleton,
                        _raw_ct_bytes=raw_file.getvalue()  # 传递原始 CT 数据用于显示
                    )
                    st.session_state.analysis_results = (
                    results, label_map_dict, raw_ct_data_final, artery_skeleton, vein_skeleton, contact_img_data)
                    st.session_state.analysis_axis = axis  # 保存用于可视化的轴
                    st.session_state.analysis_do_skeleton = do_skeleton
                    st.session_state.analysis_do_3d_render = do_3d_render

                    # 标记分析完成并重新运行以显示结果
                    st.session_state.analysis_complete = True
                    st.rerun()

    # --- 在分析完成后显示结果 ---
    if st.session_state.get('analysis_complete', False):
        results, label_map_dict, raw_ct_data_final, artery_skeleton, vein_skeleton, contact_img_data = st.session_state.analysis_results
        axis = st.session_state.analysis_axis
        do_skeleton = st.session_state.analysis_do_skeleton
        do_3d_render = st.session_state.analysis_do_3d_render

        st.header("SAM 2 交互分割后的分析结果")
        display_resectability_recommendation(results)
        with st.expander("点击查看详细分析数据"):
            st.json(results)

        if do_3d_render:
            if label_map_dict and label_map_dict.get("origin") is not None:
                st.header("交互式 3D 可视化")
                st.info("您可以拖动、旋转和缩放 3D 模型。")
                fig_3d = make_3d_surface_plot(label_map_dict["origin"])
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("无法生成 3D 视图：未加载标签数据。")

        st.header("2D 切片可视化")
        if label_map_dict and raw_ct_data_final is not None:
            data_shape = label_map_dict["origin"].shape
            if axis == 'z':
                max_slice = data_shape[2] - 1
                default_slice = data_shape[2] // 2
            else:  # axis == 'x'
                max_slice = data_shape[0] - 1
                default_slice = data_shape[0] // 2

            if max_slice >= 0:
                slice_index = st.slider("选择切片索引", 0, max_slice, min(default_slice, max_slice),
                                        key="sam_result_slider")

                col1_res, col2_res = st.columns(2)
                with col1_res:
                    st.pyplot(
                        make_organ_overlay_fig(label_map_dict, raw_ct_data_final, axis=axis, slice_index=slice_index))
                with col2_res:
                    if contact_img_data is not None:
                        st.pyplot(make_contact_overlay_fig(label_map_dict["origin"], contact_img_data, axis=axis,
                                                           slice_index=slice_index))
                    else:
                        st.info("未生成或加载接触面图像。")
            else:
                st.warning("无法确定有效的切片范围用于结果可视化。")

        else:
            st.info("未能加载SAM分割影像或原始CT影像，无法进行可视化。")

        if do_skeleton and SKELETON_AVAILABLE:
            st.markdown("---")
            st.subheader("骨架分析")
            if 'slice_index' in locals() and max_slice >= 0:  # Check if slider was created
                col1_skel_res, col2_skel_res = st.columns(2)
                with col1_skel_res:
                    if st.checkbox("显示动脉骨架", key="sam_res_skel_art") and artery_skeleton is not None:
                        st.pyplot(make_skeleton_fig(artery_skeleton, "动脉骨架", axis, slice_index))
                with col2_skel_res:
                    if st.checkbox("显示静脉骨架", key="sam_res_skel_vein") and vein_skeleton is not None:
                        st.pyplot(make_skeleton_fig(vein_skeleton, "静脉骨架", axis, slice_index))
            else:
                st.info("无法显示骨架，因为切片索引未确定。")

        # --- 添加按钮以开始新的 SAM 分割 ---
        if st.button("开始新的 SAM 交互分割", key="sam_restart"):
            # 清理 session state 以重新开始
            keys_to_reset = ['masks', 'points', 'labels', 'current_click_coords',
                             'raw_img_data', 'raw_img_nii', 'normalized_slices',
                             'sam_raw_file_id', 'analysis_complete', 'analysis_results',
                             'analysis_axis', 'analysis_do_skeleton', 'analysis_do_3d_render',
                             '_last_click_value_ref']
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            # 可能还需要清除上传组件的状态，重新运行通常可以做到
            st.rerun()
