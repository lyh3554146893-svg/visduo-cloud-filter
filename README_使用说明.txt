# VisDuo 云端筛选整合版（无 Docker 统一路径版）

这个包已经清理掉无用内容，并统一为一个目录：

```text
visduo_cloud_filter_clean\
```

云端统一路径：

```text
/root/autodl-tmp/visduo/
```

已删除 / 不再需要：

```text
.git/
__pycache__/
*.pyc
Dockerfile.*
Docker 版 bat
旧 fixed / nodocker 双版本 bat
多余 README
```

## 当前适配你的云环境

你的云端是类似 `autodl-container-...` 的容器环境，`systemctl` 和 `service docker` 都不能正常启动 Docker，所以这个包改成了**无 Docker 运行方式**。

也就是说：

```text
本地 Windows 运行 .bat
↓
自动上传数据到云端 /root/autodl-tmp/visduo/tasks/
↓
云端 Python venv 直接运行 visual_filter.py / prelabel_yolo.py
↓
自动下载结果回本地
```

## 文件说明

```text
cloud_config.txt              云服务器配置
cloud_deploy.bat              第一次部署环境
cloud_filter_upload.bat       上传本地数据到云端筛选
cloud_filter_server_path.bat  直接筛云端已有目录
cloud_upload_model.bat        上传自定义 best.pt
cloud_prelabel_upload.bat     云端预标注
cloud_prelabel_deploy.bat     提示文件：预标注已包含在 cloud_deploy.bat
run_local_filter.bat          本地直接筛选
visual_filter.py              数据筛选脚本
prelabel_yolo.py              YOLO 预标注脚本
requirements.txt              Python 依赖
快速命令.txt                  常用命令
```

## 先运行哪个

第一次只运行：

```bat
cloud_deploy.bat
```

部署成功后，先跑最稳的画质 + 版权 + OCR 初筛：

```bat
cloud_filter_upload.bat D:\data --disable_yolo --enable_copyright --enable_ocr --copy_mode all
```

输出会下载到本地：

```text
D:\visduo_cloud_filter_clean\cloud_output_时间戳\
```

## 再做预标注

用第一阶段输出目录做预标注：

```bat
cloud_prelabel_upload.bat D:\visduo_cloud_filter_clean\cloud_output_时间戳 --model yolov8n.pt --save_preview
```

如果有自定义模型：

```bat
cloud_upload_model.bat D:\models\best.pt
```

上传后云端模型路径是：

```text
/root/autodl-tmp/visduo/model/best.pt
```

预标注命令：

```bat
cloud_prelabel_upload.bat D:\visduo_cloud_filter_clean\cloud_output_时间戳 --model /root/autodl-tmp/visduo/model/best.pt --keep_names face phone smoke --save_preview
```

## 注意

Windows 命令里如果类别名带空格，比如 `cell phone`，建议用单引号：

```bat
cloud_filter_upload.bat D:\data --expected person 'cell phone' --enable_copyright --copy_mode all
```

第一步建议不要先加 YOLO / CLIP，先用：

```bat
cloud_filter_upload.bat D:\data --disable_yolo --enable_copyright --enable_ocr --copy_mode all
```

确认云端上传、运行、下载流程全部打通。
