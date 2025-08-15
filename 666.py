import streamlit as st
from diffusers import StableDiffusionXLPipeline, TextToVideoSDPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
import cv2
import numpy as np
import time
import io
import base64
import os

# 加载 Stable Diffusion XL 图像模型
@st.cache_resource
def load_sd_model():
    try:
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        return pipe
    except Exception as e:
        st.error(f"加载图像模型失败: {e}")
        return None

# 加载视频生成模型
@st.cache_resource
def load_video_model():
    try:
        model_id = "damo-vilab/text-to-video-ms-1.7b"
        pipe = TextToVideoSDPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        return pipe
    except Exception as e:
        st.error(f"加载视频模型失败: {e}")
        return None

# 加载中文扩写模型
@st.cache_resource
def load_text_model():
    try:
        model_name = "Qwen/Qwen2-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        if torch.cuda.is_available():
            model = model.cuda()
        return tokenizer, model
    except Exception as e:
        st.error(f"加载扩写模型失败: {e}")
        return None, None

# 加载 LaMa 水印去除模型
@st.cache_resource
def load_lama_model():
    try:
        from lama_cleaner.model import LaMa
        model = LaMa(device="cuda" if torch.cuda.is_available() else "cpu", model_path="F:\\models\\lama\\experiments")
        return model
    except Exception as e:
        st.error(f"加载 LaMa 模型失败: {e}")
        return None

# 加载 RIFE 补帧模型
@st.cache_resource
def load_rife_model():
    try:
        from rife import RIFE
        model = RIFE(device="cuda" if torch.cuda.is_available() else "cpu", model_path="F:\\models\\rife\\train_log")
        return model
    except Exception as e:
        st.error(f"加载 RIFE 模型失败: {e}")
        return None

# 中文扩写
def generate_text(prompt, max_length=150):
    try:
        tokenizer, model = load_text_model()
        if tokenizer is None or model is None:
            return "扩写模型未加载"
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"扩写失败: {e}"

# 生成图像
def generate_image(prompt):
    pipe = load_sd_model()
    if pipe is None:
        return None
    try:
        image = pipe(prompt, num_inference_steps=50, height=1024, width=1024).images[0]
        return image
    except Exception as e:
        st.error(f"生成图像失败: {e}")
        return None

# 生成视频
def generate_video(prompt, duration=5, fps=24):
    pipe = load_video_model()
    if pipe is None:
        return None
    try:
        frames = duration * fps
        frame_time = 1.5 if torch.cuda.is_available() else 6  # RTX 4060: 1.5s/帧
        progress_bar = st.progress(0)
        time_display = st.empty()
        video_frames = pipe(prompt, num_frames=frames, num_inference_steps=50).frames
        for i in range(frames):
            progress = (i + 1) / frames
            progress_bar.progress(progress)
            remaining_time = (frames - (i + 1)) * frame_time
            time_display.write(f"预计剩余时间: {remaining_time:.1f} 秒")
        out_path = "output.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (video_frames[0].shape[1], video_frames[0].shape[0]))
        for frame in video_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        return out_path
    except Exception as e:
        st.error(f"生成视频失败: {e}")
        return None

# 图像+描述生成视频
def generate_image_to_video(image, prompt, duration=5, fps=24):
    pipe = load_video_model()
    if pipe is None:
        return None
    try:
        frames = duration * fps
        frame_time = 1.5 if torch.cuda.is_available() else 6
        progress_bar = st.progress(0)
        time_display = st.empty()
        video_frames = pipe(prompt, num_frames=frames, num_inference_steps=50).frames  # 待优化：结合初始图像
        for i in range(frames):
            progress = (i + 1) / frames
            progress_bar.progress(progress)
            remaining_time = (frames - (i + 1)) * frame_time
            time_display.write(f"预计剩余时间: {remaining_time:.1f} 秒")
        out_path = "output_image_to_video.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (video_frames[0].shape[1], video_frames[0].shape[0]))
        for frame in video_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        return out_path
    except Exception as e:
        st.error(f"生成视频失败: {e}")
        return None

# 水印去除（LaMa）
def remove_watermark(image, mask):
    try:
        model = load_lama_model()
        if model is None:
            image_np = np.array(image)
            mask_np = np.array(mask)[:, :, 3]
            mask_np = mask_np.astype(np.uint8)
            inpainted = cv2.inpaint(image_np, mask_np, 3, cv2.INPAINT_TELEA)
            return Image.fromarray(inpainted)
        image_np = np.array(image)
        mask_np = np.array(mask)[:, :, 3]
        mask_np = mask_np.astype(np.uint8)
        inpainted = model(image_np, mask_np)
        return Image.fromarray(inpainted)
    except Exception as e:
        st.error(f"水印去除失败: {e}")
        return None

# AI 补帧（RIFE）
def ai_frame_interpolation(video_path, target_fps):
    try:
        model = load_rife_model()
        if model is None:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            original_fps = 24
            factor = target_fps / original_fps
            new_frames = []
            for i in range(len(frames)):
                new_frames.append(frames[i])
                if i < len(frames) - 1:
                    for _ in range(int(factor) - 1):
                        new_frames.append(frames[i])
            out_path = "output_interpolated.mp4"
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frames[0].shape[1], frames[0].shape[0]))
            for frame in new_frames:
                out.write(frame)
            out.release()
            return out_path
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        new_frames = []
        for i in range(len(frames) - 1):
            new_frames.append(frames[i])
            interpolated = model.infer(frames[i], frames[i + 1], num_interpolated=int(target_fps / 24 - 1))
            new_frames.extend(interpolated)
        new_frames.append(frames[-1])
        out_path = "output_interpolated.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frames[0].shape[1], frames[0].shape[0]))
        for frame in new_frames:
            out.write(frame)
        out.release()
        return out_path
    except Exception as e:
        st.error(f"补帧失败: {e}")
        return None

# 主题切换
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# UI 样式（华丽版）
theme_styles = {
    "dark": """
        .main {
            background: linear-gradient(135deg, #6b21a8, #3b82f6);
            padding: 30px;
            border-radius: 15px;
            color: white;
            animation: gradient 5s ease infinite;
            background-size: 200% 200%;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .stButton>button {
            background: linear-gradient(45deg, #9333ea, #ec4899);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0 6px 15px rgba(0,0,0,0.4);
            transition: transform 0.3s, box-shadow 0.3s, filter 0.3s;
            animation: pulse 2s infinite;
        }
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(236, 72, 153, 0.6);
            filter: brightness(1.2);
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 10px rgba(236, 72, 153, 0.4); }
            50% { box-shadow: 0 0 20px rgba(236, 72, 153, 0.8); }
            100% { box-shadow: 0 0 10px rgba(236, 72, 153, 0.4); }
        }
        .stTextInput>div>input {
            border: 2px solid #9333ea;
            border-radius: 8px;
            padding: 12px;
            background-color: rgba(30, 64, 175, 0.8);
            color: white;
            font-size: 16px;
        }
        .stSlider>div>div {
            background-color: #9333ea;
        }
        .stProgress .st-bo {
            background-color: #9333ea;
            border-radius: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(30, 64, 175, 0.8);
            color: white;
            border-radius: 8px;
            margin: 0 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #9333ea;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .title {
            animation: fadeIn 1.5s ease-in-out;
            text-shadow: 0 0 10px rgba(236, 72, 153, 0.8);
        }
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(-20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #9333ea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    """,
    "light": """
        .main {
            background: linear-gradient(135deg, #e0e7ff, #f9fafb);
            padding: 30px;
            border-radius: 15px;
            color: #1e3a8a;
            animation: gradient 5s ease infinite;
            background-size: 200% 200%;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .stButton>button {
            background: linear-gradient(45deg, #3b82f6, #93c5fd);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0 6px 15px rgba(0,0,0,0.3);
            transition: transform 0.3s, box-shadow 0.3s, filter 0.3s;
            animation: pulse 2s infinite;
        }
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
            filter: brightness(1.2);
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 10px rgba(59, 130, 246, 0.4); }
            50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.8); }
            100% { box-shadow: 0 0 10px rgba(59, 130, 246, 0.4); }
        }
        .stTextInput>div>input {
            border: 2px solid #3b82f6;
            border-radius: 8px;
            padding: 12px;
            background-color: white;
            color: #1e3a8a;
            font-size: 16px;
        }
        .stSlider>div>div {
            background-color: #3b82f6;
        }
        .stProgress .st-bo {
            background-color: #3b82f6;
            border-radius: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #e0e7ff;
            color: #1e3a8a;
            border-radius: 8px;
            margin: 0 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #93c5fd;
        }
        .card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .title {
            animation: fadeIn 1.5s ease-in-out;
            text-shadow: 0 0 10px rgba(59, 130, 246, 0.8);
        }
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(-20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3b82f6;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    """
}

st.markdown(f"""
    <style>
    {theme_styles[st.session_state.theme]}
    </style>
""", unsafe_allow_html=True)

# 主界面
st.markdown('<h1 class="title">AI 图像处理平台</h1>', unsafe_allow_html=True)
st.write(f"运行于 {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
st.button("切换主题", on_click=toggle_theme)

# 功能选择
with st.container():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["描述生成图像", "描述生成视频", "图像+描述生成视频", "智能扩写", "水印去除", "AI 补帧"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("描述生成图像")
        prompt = st.text_input("输入描述", "月光下的湖泊", key="image_prompt")
        if st.button("生成图像"):
            with st.markdown('<div class="spinner"></div>', unsafe_allow_html=True):
                image = generate_image(prompt)
                if image:
                    st.image(image, caption="生成图像")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("描述生成视频")
        prompt = st.text_input("输入描述", "星空飞船", key="video_prompt")
        duration = st.slider("视频时长（秒）", 1, 20, 5)
        st.write("选择 FPS：")
        col1, col2, col3 = st.columns(3)
        with col1:
            fps_24 = st.button("24 FPS", key="fps_24", help="生成 24 帧/秒视频")
        with col2:
            fps_30 = st.button("30 FPS", key="fps_30", help="生成 30 帧/秒视频")
        with col3:
            fps_60 = st.button("60 FPS", key="fps_60", help="生成 60 帧/秒视频")
        fps = 24
        if fps_30:
            fps = 30
        elif fps_60:
            fps = 60
        if fps_24 or fps_30 or fps_60:
            with st.markdown('<div class="spinner"></div>', unsafe_allow_html=True):
                video_path = generate_video(prompt, duration, fps)
                if video_path:
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    st.video(video_bytes)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("图像+描述生成视频")
        uploaded_image = st.file_uploader("上传图像", type=["png", "jpg", "jpeg"])
        prompt = st.text_input("输入描述", "飞船升空", key="image_video_prompt")
        duration = st.slider("视频时长（秒）", 1, 20, 5, key="image_video_duration")
        st.write("选择 FPS：")
        col1, col2, col3 = st.columns(3)
        with col1:
            fps_24 = st.button("24 FPS", key="fps_24_image", help="生成 24 帧/秒视频")
        with col2:
            fps_30 = st.button("30 FPS", key="fps_30_image", help="生成 30 帧/秒视频")
        with col3:
            fps_60 = st.button("60 FPS", key="fps_60_image", help="生成 60 帧/秒视频")
        fps = 24
        if fps_30:
            fps = 30
        elif fps_60:
            fps = 60
        if fps_24 or fps_30 or fps_60:
            with st.markdown('<div class="spinner"></div>', unsafe_allow_html=True):
                if uploaded_image:
                    image = Image.open(uploaded_image)
                    video_path = generate_image_to_video(image, prompt, duration, fps)
                    if video_path:
                        with open(video_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                else:
                    st.error("请上传图像")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("智能扩写")
        prompt = st.text_input("输入中文描述（扩写）", "海洋")
        if st.button("智能扩写"):
            with st.markdown('<div class="spinner"></div>', unsafe_allow_html=True):
                expanded_text = generate_text(prompt)
                st.write("扩写结果：")
                st.write(expanded_text)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("水印去除")
        uploaded_image = st.file_uploader("上传含水印图像", type=["png", "jpg", "jpeg"], key="watermark_image")
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="原始图像")
            # 临时替换 st_canvas
            mask_file = st.file_uploader("上传掩码图像（透明区域为水印）", type=["png"], key="mask_image")
            if st.button("去除水印"):
                with st.markdown('<div class="spinner"></div>', unsafe_allow_html=True):
                    if mask_file:
                        mask = Image.open(mask_file)
                        result = remove_watermark(image, mask)
                        if result:
                            st.image(result, caption="去除水印后")
                    else:
                        st.error("请上传掩码图像")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab6:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("AI 补帧")
        uploaded_video = st.file_uploader("上传视频", type=["mp4"])
        st.write("选择目标 FPS：")
        col1, col2, col3 = st.columns(3)
        with col1:
            target_fps_24 = st.button("24 FPS", key="target_fps_24", help="补帧到 24 帧/秒")
        with col2:
            target_fps_30 = st.button("30 FPS", key="target_fps_30", help="补帧到 30 帧/秒")
        with col3:
            target_fps_60 = st.button("60 FPS", key="target_fps_60", help="补帧到 60 帧/秒")
        target_fps = 24
        if target_fps_30:
            target_fps = 30
        elif target_fps_60:
            target_fps = 60
        if target_fps_24 or target_fps_30 or target_fps_60:
            with st.markdown('<div class="spinner"></div>', unsafe_allow_html=True):
                if uploaded_video:
                    with open("input_video.mp4", "wb") as f:
                        f.write(uploaded_video.read())
                    video_path = ai_frame_interpolation("input_video.mp4", target_fps)
                    if video_path:
                        with open(video_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                else:
                    st.error("请上传视频")
        st.markdown('</div>', unsafe_allow_html=True)
