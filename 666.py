# 导入所需库
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
import glob
import time

# 设置 Streamlit 页面配置
st.set_page_config(
    page_title="Advanced Personal AI Image & Video Tool",
    layout="wide",
    page_icon="🎥"
)

# 全局标志，用于取消任务
if 'cancel_task' not in st.session_state:
    st.session_state.cancel_task = False
if 'continue_task' not in st.session_state:
    st.session_state.continue_task = False

# 自定义 CSS：卡片式布局、动画按钮
st.markdown(
    """
    <style>
    .card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transform: translateY(-5px);
    }
    .custom-button {
        display: inline-block;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        color: white;
        background: linear-gradient(45deg, #4CAF50, #45a049);
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        transform: scale(1);
    }
    .custom-button:hover {
        transform: scale(1.1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .cancel-button {
        background: linear-gradient(45deg, #f44336, #d32f2f);
    }
    .stButton>button {
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        color: white;
        border-radius: 10px;
    }
    .history-item {
        margin: 5px 0;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 创建默认文件夹
if not os.path.exists("input"):
    os.makedirs("input")
if not os.path.exists("output"):
    os.makedirs("output")


# 初始化 Stable Diffusion 模型
@st.cache_resource
def load_diffusion_model(model_type="generate"):
    """加载 Stable Diffusion 模型，优先 GPU"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_type == "inpaint":
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_auth_token=False
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_auth_token=False
            )
        pipe = pipe.to(device)
        if device == "cpu":
            pipe.enable_attention_slicing()
        st.success(f"🎨 Stable Diffusion {'Inpainting' if model_type == 'inpaint' else 'Generate'} 加载成功（{device}）")
        return pipe
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None


# 分析图像/视频内容
def analyze_content(file, file_type):
    """分析图像/视频内容，提取主色调等特征"""
    try:
        if file_type == "图像":
            image = Image.open(file)
            image_np = np.array(image)
            mean_color = np.mean(image_np, axis=(0, 1))
            return f"dominant colors RGB({int(mean_color[0])},{int(mean_color[1])},{int(mean_color[2])})"
        else:
            cap = cv2.VideoCapture(os.path.join(st.session_state.upload_path, file.name))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return "dynamic scene"
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mean_color = np.mean(frame_rgb, axis=(0, 1))
            return f"dominant colors RGB({int(mean_color[0])},{int(mean_color[1])},{int(mean_color[2])})"
    except Exception as e:
        return "unknown content"


# 智能扩写描述
def expand_prompt(prompt, duration=None, file=None, file_type=None):
    """根据时长和内容动态扩写描述"""
    try:
        if not prompt.strip():
            return "A beautiful scene, highly detailed, cinematic"

        base_desc = prompt
        content_desc = ""
        if file and file_type:
            content_desc = analyze_content(file, file_type)

        target_words = 100 if not duration else int(30 * duration)
        additions = [
            f", highly detailed, vibrant colors, soft lighting, {content_desc}",
            f", cinematic style, realistic textures, 4K resolution, {content_desc}",
            f", in a serene environment, vivid details, dreamy atmosphere, {content_desc}"
        ]
        import random
        expanded = base_desc + random.choice(additions)

        if duration and duration > 10:
            extra_details = [
                ", with intricate patterns and subtle movements",
                ", featuring dynamic lighting and rich textures",
                ", evolving with gentle transitions and vivid contrasts"
            ]
            for _ in range(int(duration / 5)):
                expanded += random.choice(extra_details)

        target_chars = target_words * 5
        if len(expanded) > target_chars:
            expanded = expanded[:target_chars] + "..."

        return expanded
    except Exception as e:
        st.error(f"扩写错误: {e}")
        return prompt


# 描述生成图像
def generate_image_from_text(prompt, pipe):
    """使用 Stable Diffusion 生成图像"""
    try:
        if pipe is None:
            st.error("模型未加载！")
            return None
        if st.session_state.cancel_task:
            st.warning("任务已取消！")
            return None
        with torch.no_grad():
            image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
        return image
    except Exception as e:
        st.error(f"生成图像错误: {e}")
        return None


# 图像+描述生成视频
def generate_video_from_image_and_text(image, prompt, pipe, duration, fps=24):
    """基于图像和描述生成视频"""
    try:
        if pipe is None:
            st.error("模型未加载！")
            return None
        frames = []
        num_frames = int(duration * fps)
        progress = st.progress(0)

        # 初始帧
        init_image = image.resize((512, 512))  # Stable Diffusion 要求 512x512
        mask = np.zeros((512, 512), dtype=np.uint8)  # 全图 inpainting
        frames.append(np.array(init_image.convert("RGB")))

        # 生成后续帧
        for i in range(1, num_frames):
            if st.session_state.cancel_task:
                st.warning("视频生成已取消！")
                return None
            frame_prompt = f"{prompt}, frame {i}, subtle motion"
            with torch.no_grad():
                frame = pipe(
                    prompt=frame_prompt,
                    init_image=init_image,
                    mask_image=Image.fromarray(mask),
                    strength=0.3,  # 轻微修改初始图像
                    num_inference_steps=20
                ).images[0]
            frames.append(np.array(frame.convert("RGB")))
            progress.progress((i + 1) / num_frames)

        height, width = frames[0].shape[:2]
        output_path = os.path.join(st.session_state.output_path,
                                   f"generated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        return output_path
    except Exception as e:
        st.error(f"生成视频错误: {e}")
        return None


# 生成视频（描述）
def generate_video_from_text(prompt, pipe, duration, fps=24):
    """生成图像序列并合成为视频"""
    try:
        if pipe is None:
            st.error("模型未加载！")
            return None
        frames = []
        num_frames = int(duration * fps)
        progress = st.progress(0)
        for i in range(num_frames):
            if st.session_state.cancel_task:
                st.warning("视频生成已取消！")
                return None
            frame_prompt = f"{prompt}, frame {i}"
            image = generate_image_from_text(frame_prompt, pipe)
            if image:
                frames.append(np.array(image.convert("RGB")))
            progress.progress((i + 1) / num_frames)

        height, width = frames[0].shape[:2]
        output_path = os.path.join(st.session_state.output_path,
                                   f"generated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        return output_path
    except Exception as e:
        st.error(f"生成视频错误: {e}")
        return None


# 去除图像水印
def remove_watermark_image(image, mask, inpaint_radius=3):
    """使用 OpenCV inpainting 去除图像水印"""
    try:
        result = cv2.inpaint(image, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
        return result
    except Exception as e:
        st.error(f"图像去水印错误: {e}")
        return image


# 去除视频水印
def remove_watermark_video(video_path, mask, output_path, inpaint_radius=3, max_duration=None):
    """逐帧去除视频水印，支持指定时长"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_duration:
            max_frames = int(max_duration * fps)
            total_frames = min(total_frames, max_frames)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        progress = st.progress(0)
        frame_count = 0

        while cap.isOpened() and frame_count < total_frames:
            if st.session_state.cancel_task:
                cap.release()
                out.release()
                st.warning("视频处理已取消！")
                return None
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = remove_watermark_image(frame, mask, inpaint_radius)
            out.write(processed_frame)
            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()
        return output_path
    except Exception as e:
        st.error(f"视频去水印错误: {e}")
        return None


# 预计时间估算
def estimate_time(task, duration=None, frame_count=None):
    """估算任务时间（秒）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if task == "image":
        return 10 if device == "cuda" else 120
    elif task in ["video_generate", "video_from_image"]:
        image_time = 10 if device == "cuda" else 120
        return image_time * duration * 24
    elif task == "video_remove":
        frame_time = 0.05 if device == "cuda" else 0.1
        return frame_count * frame_time


# Streamlit 界面
st.title("🎥 高级个人 AI 图像与视频处理工具")
st.markdown("本地运行，带卡片式界面、动画按钮和交互式画布，支持图像+描述生成视频、自由时长和动态扩写！")
st.markdown(
    f"🌟 运行于 {'GPU' if torch.cuda.is_available() else 'CPU'} | 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 导航栏
st.sidebar.header("🚀 导航")
option = st.sidebar.radio(
    "选择功能",
    ["📷 描述生成图像", "🎬 描述生成视频", "🖼️ 图像+描述生成视频", "✍️ 智能扩写描述", "🧹 消除图像/视频水印"]
)

# 路径设置
with st.sidebar.expander("📁 文件路径设置"):
    upload_path = st.text_input("上传文件夹路径", value="input")
    output_path = st.text_input("保存文件夹路径", value="output")
    st.session_state.upload_path = upload_path
    st.session_state.output_path = output_path
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

# 历史文件
with st.sidebar.expander("📜 处理历史"):
    history_files = glob.glob(os.path.join(output_path, "*"))
    if history_files:
        st.write("最近生成/处理的文件：")
        for f in history_files[:5]:
            st.markdown(f'<p class="history-item">{os.path.basename(f)}</p>', unsafe_allow_html=True)
            st.download_button(
                label=f"⬇️ 下载 {os.path.basename(f)}",
                data=open(f, "rb").read(),
                file_name=os.path.basename(f),
                mime="image/png" if f.endswith(".png") else "video/mp4"
            )
    else:
        st.write("暂无历史文件")

# 双列布局
col1, col2 = st.columns([1, 1])

# 功能 1：描述生成图像
if option == "📷 描述生成图像":
    with col1:
        with st.container():
            st.markdown('<div class="card"><h3>📷 描述生成图像</h3>', unsafe_allow_html=True)
            prompt = st.text_area("输入描述（如 '夕阳下的湖'）", value="")
            uploaded_file = st.file_uploader("上传参考图像（可选）", type=["jpg", "png", "jpeg"])
            if st.checkbox("✨ 智能扩写描述"):
                prompt = expand_prompt(prompt, file=uploaded_file, file_type="图像")
                st.write("扩写后的描述：", prompt)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            estimated_time = estimate_time("image")
            st.write(f"预计生成时间：约 {estimated_time} 秒")
            if st.button("是否继续？"):
                st.session_state.continue_task = True
            if st.session_state.get("continue_task", False) and (st.markdown(
                    '<button class="custom-button">🎨 生成图像</button>',
                    unsafe_allow_html=True
            ) or st.button("生成图像（备用）")):
                st.session_state.cancel_task = False
                if st.markdown(
                        '<button class="custom-button cancel-button">❌ 取消生成</button>',
                        unsafe_allow_html=True
                ) or st.button("取消生成（备用）"):
                    st.session_state.cancel_task = True
                if not prompt:
                    st.error("请输入描述！")
                else:
                    with st.spinner("加载模型..."):
                        pipe = load_diffusion_model("generate")
                    if pipe:
                        with st.spinner("生成图像..."):
                            image = generate_image_from_text(prompt, pipe)
                        if image:
                            st.image(image, caption="生成的图像", use_column_width=True)
                            output_file = os.path.join(output_path,
                                                       f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                            image.save(output_file)
                            st.download_button(
                                label="⬇️ 下载生成的图像",
                                data=open(output_file, "rb").read(),
                                file_name=os.path.basename(output_file),
                                mime="image/png"
                            )
            st.markdown('</div>', unsafe_allow_html=True)

# 功能 2：描述生成视频
elif option == "🎬 描述生成视频":
    with col1:
        with st.container():
            st.markdown('<div class="card"><h3>🎬 描述生成视频</h3>', unsafe_allow_html=True)
            prompt = st.text_area("输入描述（如 '星空下的飞船'）", value="")
            duration_input = st.text_input("视频时长（秒，例：5.5）", value="5")
            try:
                duration = float(duration_input)
                if duration <= 0:
                    raise ValueError("时长必须为正数")
            except ValueError:
                st.error("请输入有效时长（正数）！")
                duration = None
            uploaded_file = st.file_uploader("上传参考图像/视频（可选）", type=["jpg", "png", "jpeg", "mp4", "avi"])
            if st.checkbox("✨ 智能扩写描述"):
                prompt = expand_prompt(prompt, duration, uploaded_file,
                                       "视频" if uploaded_file and uploaded_file.name.endswith(
                                           (".mp4", ".avi")) else "图像")
                st.write("扩写后的描述：", prompt)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if duration:
                estimated_time = estimate_time("video_generate", duration)
                st.write(f"预计生成时间：约 {estimated_time} 秒")
                if st.button("是否继续？"):
                    st.session_state.continue_task = True
                if st.session_state.get("continue_task", False) and (st.markdown(
                        '<button class="custom-button">🎬 生成视频</button>',
                        unsafe_allow_html=True
                ) or st.button("生成视频（备用）")):
                    st.session_state.cancel_task = False
                    if st.markdown(
                            '<button class="custom-button cancel-button">❌ 取消生成</button>',
                            unsafe_allow_html=True
                    ) or st.button("取消生成（备用）"):
                        st.session_state.cancel_task = True
                    if not prompt:
                        st.error("请输入描述！")
                    else:
                        with st.spinner("加载模型..."):
                            pipe = load_diffusion_model("generate")
                        if pipe:
                            with st.spinner("生成视频..."):
                                video_path = generate_video_from_text(prompt, pipe, duration)
                            if video_path:
                                st.video(video_path)
                                st.download_button(
                                    label="⬇️ 下载生成的视频",
                                    data=open(video_path, "rb").read(),
                                    file_name=os.path.basename(video_path),
                                    mime="video/mp4"
                                )
            st.markdown('</div>', unsafe_allow_html=True)

# 功能 3：图像+描述生成视频
elif option == "🖼️ 图像+描述生成视频":
    with col1:
        with st.container():
            st.markdown('<div class="card"><h3>🖼️ 图像+描述生成视频</h3>', unsafe_allow_html=True)
            prompt = st.text_area("输入描述（如 '飞船起飞'）", value="")
            uploaded_file = st.file_uploader("上传初始图像", type=["jpg", "png", "jpeg"])
            duration_input = st.text_input("视频时长（秒，例：5.5）", value="5")
            try:
                duration = float(duration_input)
                if duration <= 0:
                    raise ValueError("时长必须为正数")
            except ValueError:
                st.error("请输入有效时长（正数）！")
                duration = None
            if st.checkbox("✨ 智能扩写描述"):
                prompt = expand_prompt(prompt, duration, uploaded_file, "图像")
                st.write("扩写后的描述：", prompt)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if duration and uploaded_file:
                estimated_time = estimate_time("video_from_image", duration)
                st.write(f"预计生成时间：约 {estimated_time} 秒")
                if st.button("是否继续？"):
                    st.session_state.continue_task = True
                if st.session_state.get("continue_task", False) and (st.markdown(
                        '<button class="custom-button">🎬 生成视频</button>',
                        unsafe_allow_html=True
                ) or st.button("生成视频（备用）")):
                    st.session_state.cancel_task = False
                    if st.markdown(
                            '<button class="custom-button cancel-button">❌ 取消生成</button>',
                            unsafe_allow_html=True
                    ) or st.button("取消生成（备用）"):
                        st.session_state.cancel_task = True
                    if not prompt:
                        st.error("请输入描述！")
                    elif not uploaded_file:
                        st.error("请上传初始图像！")
                    else:
                        with st.spinner("加载模型..."):
                            pipe = load_diffusion_model("inpaint")
                        if pipe:
                            with st.spinner("生成视频..."):
                                image = Image.open(uploaded_file)
                                video_path = generate_video_from_image_and_text(image, prompt, pipe, duration)
                            if video_path:
                                st.video(video_path)
                                st.download_button(
                                    label="⬇️ 下载生成的视频",
                                    data=open(video_path, "rb").read(),
                                    file_name=os.path.basename(video_path),
                                    mime="video/mp4"
                                )
            st.markdown('</div>', unsafe_allow_html=True)

# 功能 4：智能扩写描述
elif option == "✍️ 智能扩写描述":
    with col1:
        with st.container():
            st.markdown('<div class="card"><h3>✍️ 智能扩写描述</h3>', unsafe_allow_html=True)
            prompt = st.text_area("输入简短描述（如 '猫咪'）", value="")
            duration_input = st.text_input("视频时长（秒，例：5.5，可选）", value="")
            try:
                duration = float(duration_input) if duration_input else None
                if duration is not None and duration <= 0:
                    raise ValueError("时长必须为正数")
            except ValueError:
                st.error("请输入有效时长（正数）或留空！")
                duration = None
            uploaded_file = st.file_uploader("上传参考图像/视频（可选）", type=["jpg", "png", "jpeg", "mp4", "avi"])
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if st.markdown(
                    '<button class="custom-button">✨ 扩写描述</button>',
                    unsafe_allow_html=True
            ) or st.button("扩写描述（备用）"):
                if prompt:
                    expanded_prompt = expand_prompt(
                        prompt,
                        duration,
                        uploaded_file,
                        "视频" if uploaded_file and uploaded_file.name.endswith((".mp4", ".avi")) else "图像"
                    )
                    st.write("扩写结果：", expanded_prompt)
                else:
                    st.error("请输入描述！")
            st.markdown('</div>', unsafe_allow_html=True)

# 功能 5：消除图像/视频水印
elif option == "🧹 消除图像/视频水印":
    with col1:
        with st.container():
            st.markdown('<div class="card"><h3>🧹 消除图像/视频水印</h3>', unsafe_allow_html=True)
            file_type = st.radio("文件类型", ["图像", "视频"])
            uploaded_file = st.file_uploader(
                f"上传{file_type}",
                type=["jpg", "png", "jpeg"] if file_type == "图像" else ["mp4", "avi"],
                accept_multiple_files=False
            )
            if file_type == "视频":
                duration_input = st.text_input("处理视频时长（秒，例：5.5）", value="10")
                try:
                    max_duration = float(duration_input)
                    if max_duration <= 0:
                        raise ValueError("时长必须为正数")
                except ValueError:
                    st.error("请输入有效时长（正数）！")
                    max_duration = None
            else:
                max_duration = None
            stroke_width = st.slider("🖌️ 画笔粗细", 1, 50, 10)
            stroke_color = st.color_picker("🎨 画笔颜色", "#FFFFFF")
            inpaint_radius = st.slider("🛠️ 补全半径", 1, 10, 3)
            if uploaded_file:
                if file_type == "图像":
                    image = Image.open(uploaded_file)
                    image_np = np.array(image)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    cap = cv2.VideoCapture(os.path.join(upload_path, uploaded_file.name))
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        st.error("无法读取视频帧！")
                        image_np = None
                    else:
                        image_np = frame
                if image_np is not None:
                    st.write("绘制水印区域（白色）或擦除（黑色橡皮）")
                    canvas_result = st_canvas(
                        fill_color=stroke_color,
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        background_image=Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)),
                        height=min(image_np.shape[0], 600),
                        width=min(image_np.shape[1], 800),
                        drawing_mode="freedraw",
                        key="canvas"
                    )
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if uploaded_file and max_duration is not None:
                estimated_time = estimate_time("image" if file_type == "图像" else "video_remove",
                                               frame_count=int(max_duration * 24) if file_type == "视频" else None)
                st.write(f"预计处理时间：约 {estimated_time} 秒")
                if st.button("是否继续？"):
                    st.session_state.continue_task = True
            if st.session_state.get("continue_task", False) and (st.markdown(
                    '<button class="custom-button">🧹 去除水印</button>',
                    unsafe_allow_html=True
            ) or st.button("去除水印（备用）")):
                st.session_state.cancel_task = False
                if st.markdown(
                        '<button class="custom-button cancel-button">❌ 取消处理</button>',
                        unsafe_allow_html=True
                ) or st.button("取消处理（备用）"):
                    st.session_state.cancel_task = True
                if uploaded_file and canvas_result and canvas_result.image_data is not None:
                    try:
                        mask = canvas_result.image_data[:, :, 3].astype(np.uint8)
                        mask[mask > 0] = 255

                        if file_type == "图像":
                            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                            with st.spinner("处理图像..."):
                                result = remove_watermark_image(image, mask, inpaint_radius)
                            st.image(result, caption="去水印后的图像", channels="BGR", use_column_width=True)
                            output_file = os.path.join(output_path,
                                                       f"processed_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                            cv2.imwrite(output_file, result)
                            st.download_button(
                                label="⬇️ 下载去水印图像",
                                data=open(output_file, "rb").read(),
                                file_name=os.path.basename(output_file),
                                mime="image/png"
                            )

                        elif file_type == "视频":
                            temp_path = os.path.join(upload_path, uploaded_file.name)
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.read())
                            output_file = os.path.join(output_path,
                                                       f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                            with st.spinner("处理视频..."):
                                result = remove_watermark_video(temp_path, mask, output_file, inpaint_radius,
                                                                max_duration)
                            if result:
                                st.video(result)
                                st.download_button(
                                    label="⬇️ 下载去水印视频",
                                    data=open(result, "rb").read(),
                                    file_name=os.path.basename(result),
                                    mime="video/mp4"
                                )

                    except Exception as e:
                        st.error(f"处理错误: {e}. 请检查文件格式或掩码是否正确。")
                else:
                    st.error("请上传文件并绘制水印区域！")
            st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown(f"🌟 运行于 {'GPU' if torch.cuda.is_available() else 'CPU'} | 上传: {upload_path} | 保存: {output_path}")
st.markdown("**提示**：输入任意时长（如 5.5 秒）；白色画笔标记水印，黑色擦除；动态水印需进一步优化。")