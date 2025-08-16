import streamlit as st
import torch
from diffusers import FluxPipeline, StableVideoDiffusionPipeline
from lama_cleaner.model import LaMa
from real_esrgan import RealESRGAN
from rife import RIFE
from transformers import pipeline
import pydub
import ffmpeg
import cv2
import pynvml
from huggingface_hub import hf_hub_download
import os
import shutil
import time
import logging
from streamlit_drawable_canvas import st_canvas

# 配置日志
logging.basicConfig(filename="logs/error.log", level=logging.ERROR)

# 界面设置
def setup_ui():
    st.markdown(
        """
        <style>
        .stApp { background: url('assets/background1.jpg'); background-size: cover; }
        .stButton>button { background: linear-gradient(45deg, #00f, #f0f); color: white; border-radius: 10px; }
        .stButton>button:hover { box-shadow: 0 0 15px #0ff; transform: rotateY(10deg); transition: 0.3s; }
        .stTabs>button { background: #0ff; color: white; border-radius: 5px; }
        .stTabs>button:hover { box-shadow: 0 0 10px #0ff; }
        .card { background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 10px; }
        .card:hover { transform: rotateY(180deg); transition: 0.5s; }
        </style>
        <script src='assets/particles.js'></script>
        """,
        unsafe_allow_html=True,
    )
    st.title("AI 图像与视频神器", anchor="title")

# 模型加载
@st.cache_resource
def load_models():
    try:
        flux = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16
        ).to("cuda")
        svd = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion", torch_dtype=torch.float16
        ).to("cuda")
        lama = LaMa()
        esrgan = RealESRGAN()
        rife = RIFE()
        return flux, svd, lama, esrgan, rife
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}")
        st.error("模型加载失败，请检查日志 logs/error.log")
        return None, None, None, None, None

# 缓存管理
def cache_manager():
    cache_dir = "cache/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    try:
        pynvml.smi.nvmlInit()
        mem_info = pynvml.smi.nvmlDeviceGetMemoryInfo(pynvml.smi.nvmlDeviceGetHandleByIndex(0))
        disk_usage = sum(
            os.path.getsize(f"{cache_dir}/{f}") for f in os.listdir(cache_dir) if os.path.isfile(f"{cache_dir}/{f}")
        )
        if mem_info.used > 7.5 * 1024**3 or disk_usage > 40 * 1024**3:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
            st.success("缓存已清理")
        for file in os.listdir(cache_dir):
            if time.time() - os.path.getmtime(f"{cache_dir}/{file}") > 12 * 3600:
                os.remove(f"{cache_dir}/{file}")
    except Exception as e:
        logging.error(f"缓存管理失败: {str(e)}")

# 音频处理
def audio_processor(video_file=None, audio_file=None, volume=1.0, fade_in=0, fade_out=0, start_time=0, end_time=None):
    try:
        if video_file:
            audio = pydub.AudioSegment.from_file(video_file, format="mp4")
            audio.export("cache/audio.mp3", format="mp3")
            audio.export("cache/audio.wav", format="wav")
            return audio
        if audio_file:
            audio = pydub.AudioSegment.from_file(audio_file)
            audio = audio.fade_in(fade_in * 1000).fade_out(fade_out * 1000)
            audio = audio + (volume * 10 - 10)
            if end_time:
                audio = audio[start_time * 1000 : end_time * 1000]
            audio.export("cache/processed_audio.mp3", format="mp3")
            return audio
    except Exception as e:
        logging.error(f"音频处理失败: {str(e)}")
        st.error("音频处理失败，请检查上传文件格式")

# 最近生成记录
def recent_results():
    cache_dir = "cache/"
    results = []
    for file in os.listdir(cache_dir):
        if file.endswith((".png", ".jpg", ".gif", ".mp4")):
            results.append({"file": file, "time": os.path.getmtime(f"{cache_dir}/{file}")})
    results.sort(key=lambda x: x["time"], reverse=True)
    for res in results:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if res["file"].endswith((".png", ".jpg", ".gif")):
                st.image(f"{cache_dir}/{res["file"]}", caption=res["file"])
            else:
                st.video(f"{cache_dir}/{res["file"]}")
            st.download_button("下载", open(f"{cache_dir}/{res["file"]}", "rb").read(), res["file"])
            st.markdown("</div>", unsafe_allow_html=True)

# 主函数
def main():
    setup_ui()
    flux, svd, lama, esrgan, rife = load_models()
    if not flux:
        return
    cache_manager()

    with st.sidebar:
        st.header("设置")
        bg = st.selectbox("背景", ["城市夜景", "未来科技", "霓虹街景"], key="bg")
        if bg != st.session_state.get("bg", ""):
            st.session_state.bg = bg
            st.markdown(f'<style>.stApp {{ background: url("assets/background{bg}.jpg"); }}</style>', unsafe_allow_html=True)
        if st.button("清理缓存"):
            shutil.rmtree("cache/")
            os.makedirs("cache/")
            st.success("缓存已清理")
        st.markdown("[帮助页面](#help)")

    tabs = st.tabs(["图像生成", "视频生成", "AI 消除", "画质增强", "补帧", "帮助"])

    with tabs[0]:  # 图像生成
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            prompt = st.text_area("描述", height=100, help="建议 100 字")
            expand = st.selectbox("扩充程度", ["简洁", "中等", "详细"])
            if st.button("扩充描述"):
                try:
                    expanded = pipeline("text-generation", model="Qwen/Qwen2-7B-Instruct")(prompt)
                    prompt = st.text_area("编辑描述", value=expanded[0]["generated_text"])
                except Exception as e:
                    logging.error(f"描述扩充失败: {str(e)}")
                    st.error("描述扩充失败，请检查网络或模型")
            image = st.file_uploader("辅助图片", ["png", "jpeg", "gif"])
            style = st.selectbox("风格", ["赛博朋克", "写实", "卡通"])
            if st.button("示例描述"):
                prompt = st.text_area("描述", value="赛博朋克城市夜景，霓虹灯光，未来科技感")
        with col2:
            if st.button("生成", key="gen_img"):
                with st.spinner("生成中..."):
                    try:
                        start_time = time.time()
                        img = flux(prompt, style=style, guidance_scale=7.5, num_inference_steps=50).images[0]
                        img.save(f"cache/image_{int(start_time)}.png")
                        st.image(img, caption="生成结果")
                        st.download_button("下载", open(f"cache/image_{int(start_time)}.png", "rb").read(), "result.png")
                        st.write(f"预计剩余时间: {int(10 - (time.time() - start_time))} 秒")
                    except Exception as e:
                        logging.error(f"图像生成失败: {str(e)}")
                        st.error("生成失败，请检查输入或日志")
            recent_results()

    with tabs[1]:  # 视频生成
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            prompt = st.text_area("描述", height=100)
            duration = st.slider("时长 (秒)", 1, 60, 10)
            fps = st.selectbox("帧率", [24, 30, 60])
            consistency = st.slider("一致性 (仅限 ≥15秒)", 0.0, 1.0, 0.8)
            image = st.file_uploader("辅助图片", ["png", "jpeg", "gif"])
            audio_file = st.file_uploader("音频", ["mp3", "wav"])
            video_file = st.file_uploader("提取音频", ["mp4"])
            if video_file:
                audio = audio_processor(video_file=video_file)
                st.audio("cache/audio.mp3")
                st.download_button("下载音频", open("cache/audio.mp3", "rb").read(), "audio.mp3")
            if audio_file:
                fade_in = st.slider("淡入 (秒)", 0, 5, 0)
                fade_out = st.slider("淡出 (秒)", 0, 5, 0)
                volume = st.slider("音量 (%)", 0, 200, 100)
                start_time = st.number_input("起始时间 (秒)", 0.0)
                end_time = st.number_input("结束时间 (秒)", value=None)
                audio = audio_processor(audio_file=audio_file, volume=volume / 100, fade_in=fade_in, fade_out=fade_out, start_time=start_time, end_time=end_time)
                st.audio("cache/processed_audio.mp3")
        with col2:
            if st.button("生成", key="gen_video"):
                with st.spinner("生成中..."):
                    try:
                        start_time = time.time()
                        if duration >= 15:
                            video = svd(prompt, duration=5, fps=fps, consistency=consistency, image=image).video
                            for i in range(1, duration // 5):
                                video += svd(prompt, duration=5, fps=fps, consistency=consistency, image=video[-1]).video
                        else:
                            video = svd(prompt, duration=duration, fps=fps, image=image).video
                        video_path = f"cache/video_{int(start_time)}.mp4"
                        video.save(video_path)
                        st.video(video_path)
                        st.download_button("下载", open(video_path, "rb").read(), "result.mp4")
                        st.write(f"预计剩余时间: {int(30 - (time.time() - start_time))} 秒")
                    except Exception as e:
                        logging.error(f"视频生成失败: {str(e)}")
                        st.error("生成失败，请检查输入或日志")
            recent_results()

    with tabs[2]:  # AI 消除
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            img_file = st.file_uploader("上传图像", ["png", "jpeg", "gif"])
            if img_file:
                img = cv2.imread(img_file.name)
                canvas = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.3)",
                    stroke_width=st.slider("画笔粗细", 1, 50, 10),
                    stroke_color="white",
                    background_image=img,
                    update_streamlit=True,
                    height=img.shape[0],
                    width=img.shape[1],
                    drawing_mode=st.selectbox("工具", ["freedraw", "eraser"]),
                    key="canvas",
                )
        with col2:
            if st.button("应用消除"):
                with st.spinner("处理中..."):
                    try:
                        mask = canvas.image_data[:, :, 3] > 0
                        result = lama(img, mask)
                        result_path = f"cache/remove_{int(time.time())}.png"
                        cv2.imwrite(result_path, result)
                        st.image(result_path, caption="消除结果")
                        st.download_button("下载", open(result_path, "rb").read(), "result.png")
                    except Exception as e:
                        logging.error(f"AI 消除失败: {str(e)}")
                        st.error("消除失败，请检查输入或日志")
            recent_results()

    with tabs[3]:  # 画质增强
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            files = st.file_uploader("上传图像/视频", ["png", "jpeg", "gif", "mp4"], accept_multiple_files=True)
            resolution = st.selectbox("目标分辨率", ["2K", "4K", "1080p", "720p"])
        with col2:
            if st.button("增强", key="enhance"):
                with st.spinner("处理中..."):
                    try:
                        start_time = time.time()
                        for file in files[:10]:
                            if file.name.endswith((".png", ".jpg", ".gif")):
                                img = cv2.imread(file.name)
                                result = esrgan(img, scale=2 if resolution in ["2K", "4K"] else 1)
                                result_path = f"cache/enhance_{int(start_time)}_{file.name}"
                                cv2.imwrite(result_path, result)
                                st.image(result_path, caption=file.name)
                                st.download_button("下载", open(result_path, "rb").read(), file.name)
                            else:
                                video = cv2.VideoCapture(file.name)
                                result = esrgan(video, scale=2 if resolution in ["2K", "4K"] else 1)
                                result_path = f"cache/enhance_{int(start_time)}_{file.name}"
                                result.save(result_path)
                                st.video(result_path)
                                st.download_button("下载", open(result_path, "rb").read(), file.name)
                            st.write(f"预计剩余时间: {int(15 - (time.time() - start_time))} 秒")
                    except Exception as e:
                        logging.error(f"画质增强失败: {str(e)}")
                        st.error("增强失败，请检查输入或日志")
            recent_results()

    with tabs[4]:  # 补帧
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            video_file = st.file_uploader("上传视频", ["mp4"])
            fps = st.selectbox("目标帧率", [60, 90, 120])
        with col2:
            if st.button("补帧", key="interpolate"):
                with st.spinner("处理中..."):
                    try:
                        start_time = time.time()
                        video = cv2.VideoCapture(video_file.name)
                        result = rife(video, target_fps=fps)
                        result_path = f"cache/interpolate_{int(start_time)}.mp4"
                        result.save(result_path)
                        st.video(result_path)
                        st.download_button("下载", open(result_path, "rb").read(), "result.mp4")
                        st.write(f"预计剩余时间: {int(20 - (time.time() - start_time))} 秒")
                    except Exception as e:
                        logging.error(f"补帧失败: {str(e)}")
                        st.error("补帧失败，请检查输入或日志")
            recent_results()

    with tabs[5]:  # 帮助
        st.header("帮助页面")
        st.write("### 图像生成\n输入描述，生成图像，支持批量处理...")
        st.write("### 常见问题\n- 显存不足: 降低批次或分辨率\n- 安装失败: 使用 --pre 安装预发布版")

if __name__ == "__main__":
    main()
