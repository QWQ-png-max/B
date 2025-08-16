@echo off
echo 正在配置 AI Toolkit 环境...

:: 检查 Python 是否安装
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 错误: 未找到 Python，请先安装 Python 3.13（下载：https://www.python.org/downloads/）
    pause
    exit /b
)

:: 创建并激活虚拟环境
if not exist "F:\1\.venv" (
    python -m venv F:\1\.venv
)
call F:\1\.venv\Scripts\activate

:: 升级 pip
python -m pip install --upgrade pip

:: 清理 pip 缓存
python -m pip cache purge

:: 安装依赖
echo 正在安装依赖...
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if %ERRORLEVEL% neq 0 (
    echo 依赖安装失败，请检查网络或 requirements.txt
    pause
    exit /b
)

:: 检查 FFmpeg
ffmpeg -version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 正在下载 FFmpeg...
    powershell -Command "Invoke-WebRequest -Uri https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip -OutFile ffmpeg.zip"
    powershell -Command "Expand-Archive -Path ffmpeg.zip -DestinationPath 'C:\Program Files\ffmpeg'"
    setx PATH "%PATH%;C:\Program Files\ffmpeg\ffmpeg-master-latest-win64-gpl\bin"
    del ffmpeg.zip
    echo FFmpeg 已安装并添加到 PATH
)

:: 创建必要文件夹
mkdir weights cache data logs
echo. > logs\error.log

:: 验证 CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
if %ERRORLEVEL% neq 0 (
    echo CUDA 配置可能有问题，请检查 NVIDIA 驱动和 CUDA 13.0
)

echo 安装完成！运行 run.bat 启动程序
pause
