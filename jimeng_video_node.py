"""
即梦AI视频节点
ComfyUI插件的文/图生视频合并为一个节点
"""

import os
import json
import logging
import torch
import time
import requests
import io
import datetime
import uuid
import random
import hashlib
import hmac
import binascii
import urllib.parse
import base64
from typing import Dict, Any, Tuple, Optional
from PIL import Image

# 导入核心模块
from .core.token_manager import TokenManager
from .core.api_client import ApiClient

logger = logging.getLogger(__name__)

class JimengVideoNode:
    """
    即梦AI视频节点
    通过 first_frame_image 是否为空自动判断是文生视频还是图生视频
    """
    def __init__(self):
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = self._load_config()
        self.token_manager = None
        self.api_client = None
        self._initialize_components()

    def _load_config(self) -> Dict[str, Any]:
        """
        加载插件的 config.json 配置文件。
        """
        try:
            config_path = os.path.join(self.plugin_dir, "config.json")
            if not os.path.exists(config_path):
                template_path = os.path.join(self.plugin_dir, "config.json.template")
                if os.path.exists(template_path):
                    import shutil
                    shutil.copy(template_path, config_path)
                    logger.info("[JimengVideoNode] 从模板创建了 config.json")
                else:
                    logger.error("[JimengVideoNode] 配置文件和模板文件都不存在！")
                    return {}
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("[JimengVideoNode] 配置文件加载成功")
            return config
        except Exception as e:
            logger.error(f"[JimengVideoNode] 配置文件加载失败: {e}")
            return {}

    def _is_configured(self) -> bool:
        """
        检查配置是否包含至少一个有效的sessionid。
        """
        accounts = self.config.get("accounts", [])
        if not isinstance(accounts, list) or not accounts:
            return False
        return any(acc.get("sessionid") for acc in accounts)

    def _initialize_components(self):
        """
        基于加载的配置初始化TokenManager和ApiClient。
        """
        if not self.config:
            logger.error("[JimengVideoNode] 因配置为空，核心组件初始化失败。")
            return
        try:
            self.token_manager = TokenManager(self.config)
            self.api_client = ApiClient(self.token_manager, self.config)
            logger.info("[JimengVideoNode] 核心组件初始化成功。")
        except Exception as e:
            logger.error(f"[JimengVideoNode] 核心组件初始化失败: {e}", exc_info=True)

    @classmethod
    def INPUT_TYPES(cls):
        """
        动态生成UI输入参数，包括视频模型、比例、时长等
        """
        # 加载配置
        try:
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(plugin_dir, "config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            config = {}
        # 视频模型
        video_models = config.get("video_models", {})
        model_options = [(k, v.get("name", k)) for k, v in video_models.items()]
        model_keys = [k for k, _ in model_options]
        model_names = [v for _, v in model_options]
        default_model = config.get("default_video_model", model_keys[0] if model_keys else "s2.0")
        # 比例
        video_ratios = config.get("video_ratios", {})
        ratio_options = list(video_ratios.keys())
        default_ratio = config.get("default_video_ratio", ratio_options[0] if ratio_options else "16:9")
        # 时长（前端以秒为单位，后端自动转毫秒）
        duration_options = [5, 10]
        default_duration = 5
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "描述你想要的视频"}),
                "video_model": (model_keys, {"default": default_model, "display_names": model_names, "tooltip": "选择视频生成模型"}),
                "video_ratio": (ratio_options, {"default": default_ratio, "tooltip": "选择视频比例"}),
                "duration": (duration_options, {"default": default_duration, "display_names": [f"{x}秒" for x in duration_options], "tooltip": "选择视频时长（秒）"}),
            },
            "optional": {
                "first_frame_image": ("IMAGE", {"tooltip": "可选，首帧图片，留空为文生视频"}),
                "end_frame_image": ("IMAGE", {"tooltip": "可选，首尾帧视频时需填写"})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_url", "generation_info")
    FUNCTION = "generate_video"
    CATEGORY = "即梦AI"

    def _save_input_image(self, image_tensor: torch.Tensor, frame_type: str = "input") -> Optional[str]:
        """
        将输入的图像张量保存为临时文件。
        
        Args:
            image_tensor: 图像张量
            frame_type: 帧类型标识，用于区分首帧和尾帧
        """
        try:
            temp_dir = os.path.join(self.plugin_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 使用更精确的时间戳和随机数确保文件名唯一
            import uuid
            timestamp = int(time.time() * 1000)  # 毫秒级时间戳
            random_id = str(uuid.uuid4())[:8]    # 8位随机ID
            temp_path = os.path.join(temp_dir, f"temp_{frame_type}_{timestamp}_{random_id}.png")
            
            if len(image_tensor.shape) == 4:
                image_tensor = image_tensor[0]
            image_tensor = torch.clamp(image_tensor, 0, 1)
            image_np = (image_tensor.cpu().numpy() * 255).astype('uint8')
            image_pil = Image.fromarray(image_np)
            image_pil.save(temp_path)
            
            # 计算图片MD5用于调试
            import hashlib
            with open(temp_path, "rb") as f:
                image_md5 = hashlib.md5(f.read()).hexdigest()[:8]
            
            logger.info(f"[JimengVideoNode] {frame_type}图像已保存到: {temp_path} (MD5: {image_md5})")
            return temp_path
        except Exception as e:
            logger.error(f"[JimengVideoNode] 保存{frame_type}图像失败: {e}")
            return None

    def _generate_info_text(self, prompt: str, model: str, ratio: str, duration: int) -> str:
        """
        生成generation_info文本，包含主要参数和积分
        """
        # 获取模型名称
        model_name = self.config.get("video_models", {}).get(model, {}).get("name", model)
        # 获取积分
        credit_info = self.token_manager.get_credit() if self.token_manager else None
        credit_text = f"\n当前账号剩余积分: {credit_info.get('total_credit', '未知')}" if credit_info else ""
        info_lines = [
            f"提示词: {prompt}",
            f"模型: {model_name}",
            f"比例: {ratio}",
            f"时长: {duration // 1000}秒"
        ]
        return "\n".join(info_lines) + credit_text

    def _log_video_request(self, account_number, credit, video_type, prompt, model, ratio, duration, web_id):
        logger.info(f"[JimengVideoNode] 使用账号{account_number}生成视频，当前积分：{credit}")
        logger.info(f"[JimengVideoNode] 视频请求参数：")
        logger.info(f"  - 类型：{video_type}")
        logger.info(f"  - 提示词：{prompt}")
        logger.info(f"  - 模型：{model}")
        logger.info(f"  - 比例：{ratio}")
        logger.info(f"  - 时长：{duration // 1000}秒")
        logger.info(f"  - 账号web_id：{web_id}")

    def generate_video(self, prompt: str, video_model: str, video_ratio: str, duration: int, first_frame_image: torch.Tensor = None, end_frame_image: torch.Tensor = None) -> Tuple[str, str]:
        try:
            if not self.token_manager or not self.api_client:
                return ("错误: 插件未正确初始化，请检查后台日志。", "")
            if not self._is_configured():
                return ("错误: 插件未配置，请在 config.json 中至少填入一个账号的 sessionid。", "")
            if not prompt or not prompt.strip():
                return ("错误: 提示词不能为空。", "")

            duration_ms = duration * 1000  # 前端为秒，后端转为毫秒
            account_number = self.token_manager.current_account_index + 1 if self.token_manager else 1
            credit_info = self.token_manager.get_credit() if self.token_manager else None
            credit = credit_info.get('total_credit', '未知') if credit_info else '未知'
            web_id = self.token_manager.get_current_account().get('web_id', '-') if self.token_manager else '-'

            # 首尾帧视频（仅支持v3.0）
            if first_frame_image is not None and end_frame_image is not None:
                if video_model != "v3.0":
                    return ("错误: 首尾帧视频目前仅支持v3.0模型。", "")
                video_type = "首尾帧视频"
                self._log_video_request(account_number, credit, video_type, prompt, video_model, video_ratio, duration_ms, web_id)
                # 保存图片
                first_path = self._save_input_image(first_frame_image, "first_frame")
                end_path = self._save_input_image(end_frame_image, "end_frame")
                if not first_path or not end_path:
                    return ("错误: 首/尾帧图片保存失败。", "")
                success, result = self.frames_to_video(
                    first_image_path=first_path,
                    end_image_path=end_path,
                    prompt=prompt,
                    duration_ms=duration_ms,
                    model=video_model
                )
                info = self._generate_info_text(prompt, video_model, video_ratio, duration_ms)
                if not success:
                    return (f"错误: {result}", info)
                logger.info(f"[JimengVideoNode] 首尾帧视频生成成功，下载链接: {result}")
                return (result, info)
            # 图生视频
            elif first_frame_image is not None:
                video_type = "图生视频"
                self._log_video_request(account_number, credit, video_type, prompt, video_model, video_ratio, duration_ms, web_id)
                image_path = self._save_input_image(first_frame_image, "first_frame")
                if not image_path:
                    return ("错误: 首帧图片保存失败。", "")
                success, result = self.upload_image_and_generate_video(
                    image_path=image_path,
                    prompt=prompt,
                    model=video_model,
                    duration_ms=duration_ms
                )
                info = self._generate_info_text(prompt, video_model, video_ratio, duration_ms)
                if not success:
                    return (f"错误: {result}", info)
                logger.info(f"[JimengVideoNode] 视频生成成功，下载链接: {result}")
                return (result, info)
            # 文生视频
            else:
                video_type = "文生视频"
                self._log_video_request(account_number, credit, video_type, prompt, video_model, video_ratio, duration_ms, web_id)
                success, result = self.text_to_video_generation(
                    prompt=prompt,
                    first_frame_info=None,
                    duration_ms=duration_ms,
                    model=video_model,
                    ratio=video_ratio
                )
                info = self._generate_info_text(prompt, video_model, video_ratio, duration_ms)
                if not success:
                    return (f"错误: {result}", info)
                logger.info(f"[JimengVideoNode] 视频生成成功，下载链接: {result}")
                return (result, info)
        except Exception as e:
            logger.error(f"[JimengVideoNode] 节点执行异常: {e}")
            return (f"错误: 节点执行异常: {e}", "")

    def upload_image_and_generate_video(self, image_path, prompt="人物表情慢慢变沮丧痛哭流涕", model="s2.0", duration_ms=5000):
        """上传图片并生成视频
        Args:
            image_path: 图片路径
            prompt: 提示词
            model: 视频模型，默认s2.0
            duration_ms: 视频时长（毫秒），默认5000ms（5秒）
        Returns:
            tuple: (success, result)
        """
        try:
            logger.info("[JimengVideoNode] 开始上传图片并生成视频")
            
            # 获取上传token
            token_info = self.token_manager.get_upload_token()
            if not token_info:
                return False, "获取上传token失败"
            
            # 获取文件大小
            file_size = os.path.getsize(image_path)
            
            # 申请上传
            upload_info = self.apply_image_upload(token_info, file_size)
            if not upload_info or "Result" not in upload_info:
                return False, f"申请上传失败: {upload_info}"
            
            logger.info("[JimengVideoNode] 图片上传申请成功，开始上传图片")
            logger.debug(f"[JimengVideoNode] 上传申请结果: {json.dumps(upload_info, ensure_ascii=False)}")
            
            # 上传图片
            file_response = self.upload_image_file(upload_info, image_path)
            if not file_response:
                return False, f"上传图片失败: {file_response}"
            
            logger.info("[JimengVideoNode] 图片上传成功，开始提交上传信息")
            
            # 提交上传
            commit_response = self.commit_image_upload(token_info, upload_info)
            if not commit_response or "Result" not in commit_response:
                return False, f"提交上传失败: {commit_response}"
            
            logger.info("[JimengVideoNode] 上传信息提交成功，开始生成视频")
            logger.debug(f"[JimengVideoNode] 图片信息: {json.dumps(commit_response, ensure_ascii=False)}")
            
            # 生成视频，传递 model 和 duration_ms 参数
            video_response = self._generate_video(commit_response, prompt=prompt, model=model, duration_ms=duration_ms)
            if not video_response or video_response.get("ret") != "0":
                return False, f"生成视频失败: {video_response}"
            
            # 获取task_id
            task_id = video_response.get("data", {}).get("aigc_data", {}).get("task", {}).get("task_id")
            if not task_id:
                return False, "未获取到任务ID"
            
            logger.info("[JimengVideoNode] 视频生成任务创建成功，等待生成完成")
            
            # 检查视频状态
            success, result = self.check_image_to_video_status(task_id)
            return success, result
            
        except Exception as e:
            logger.error(f"[JimengVideoNode] 视频生成过程出错: {str(e)}")
            return False, str(e)

    def _generate_video(self, commit_info, prompt="人物表情慢慢变沮丧痛哭流涕", model="s2.0", duration_ms=5000):
        """生成视频
        Args:
            commit_info: 提交上传后的信息
            prompt: 提示词
            model: 视频模型，默认s2.0
            duration_ms: 视频时长（毫秒），默认5000ms（5秒）
        Returns:
            dict: 视频生成任务的响应
        """
        try:
            # 记录调试信息
            logger.info(f"[JimengVideoNode] 开始生成视频，使用模型: {model}")
            
            # 获取token信息
            token_info = self.token_manager.get_token("/mweb/v1/generate_video")
            if not token_info:
                return None
                
            # 获取账号信息
            account = self.token_manager.get_current_account()
            if not account:
                return None

            # 从配置文件获取模型信息
            video_models = self.config.get("video_models", {})
            model_info = video_models.get(model.lower(), video_models.get("s2.0", {}))
            
            # 默认参数设置
            default_params = {
                "v3.0": {
                    "benefit_type": "basic_video_operation_vgfm_v_three",
                    "fps": 24,
                    "feature_entrance": "to_video",
                    "feature_entrance_detail": "to_image-text_to_video",
                    "video_mode": 2
                },
                "s2.0p": {
                    "benefit_type": "basic_video_operation_vgfm",
                    "fps": 24,
                    "feature_entrance": "to_video",
                    "feature_entrance_detail": "to_image-text_to_video",
                    "video_mode": 2
                },
                "s2.0": {
                    "benefit_type": "basic_video_operation_vgfm_lite",
                    "fps": 24,
                    "feature_entrance": "to_video",
                    "feature_entrance_detail": "to_image-text_to_video",
                    "video_mode": 2
                }
            }
            
            params_info = default_params.get(model.lower(), default_params["s2.0"])
            
            model_req_key = model_info.get("model_req_key", "dreamina_ic_generate_video_model_vgfm_lite")
            benefit_type = params_info["benefit_type"]
            fps = params_info["fps"]
            feature_entrance = params_info["feature_entrance"]
            feature_entrance_detail = params_info["feature_entrance_detail"]
            video_mode = params_info["video_mode"]

            plugin_result = commit_info['Result']['PluginResult'][0]
            
            # 构建请求参数
            babi_param = {
                "scenario": "image_video_generation",
                "feature_key": "image_to_video",
                "feature_entrance": feature_entrance,
                "feature_entrance_detail": feature_entrance_detail
            }
            
            params = {
                "aid": "513695",
                "babi_param": json.dumps(babi_param),
                "device_platform": "web",
                "region": "CN",
                "web_id": self.token_manager.get_web_id(),
                "msToken": token_info.get("msToken", ""),
                "a_bogus": token_info.get("a_bogus", "")
            }

            # 构建请求体
            data = {
                "submit_id": str(uuid.uuid4()),
                "task_extra": json.dumps({
                    "promptSource": "custom",
                    "originSubmitId": str(uuid.uuid4()),
                    "isDefaultSeed": 1,
                    "originTemplateId": "",
                    "imageNameMapping": {},
                    "isUseAiGenPrompt": False,
                    "batchNumber": 1
                }),
                "http_common_info": {
                    "aid": "513695"
                },
                "input": {
                    "seed": random.randint(1000000000, 9999999999),
                    "video_gen_inputs": [{
                        "prompt": prompt,
                        "first_frame_image": {
                            "width": plugin_result['ImageWidth'],
                            "height": plugin_result['ImageHeight'],
                            "image_uri": plugin_result['ImageUri']
                        },
                        "fps": fps,
                        "duration_ms": duration_ms,
                        "video_mode": video_mode,
                        "template_id": ""
                    }],
                    "priority": 0,
                    "model_req_key": model_req_key
                },
                "mode": "workbench",
                "history_option": {},
                "commerce_info": {
                    "resource_id": "generate_video",
                    "resource_id_type": "str",
                    "resource_sub_type": "aigc",
                    "benefit_type": benefit_type
                },
                "client_trace_data": {}
            }

            # 根据模型添加额外的请求头
            headers = {
                'accept': 'application/json, text/plain, */*',
                'accept-language': 'zh-CN,zh;q=0.9',
                'app-sdk-version': '48.0.0',
                'appid': '513695',
                'appvr': '5.8.0',
                'content-type': 'application/json',
                'cookie': token_info["cookie"],
                'device-time': token_info["device_time"],
                'lan': 'zh-Hans',
                'loc': 'cn',
                'origin': 'https://jimeng.jianying.com',
                'pf': '7',
                'priority': 'u=1, i',
                'referer': 'https://jimeng.jianying.com/ai-tool/video/generate',
                'sign': token_info["sign"],
                'sign-ver': '1',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin'
            }

            url = "https://jimeng.jianying.com/mweb/v1/generate_video"

            # 打印详细的请求信息
            logger.debug(f"[JimengVideoNode] 生成视频请求URL: {url}")
            logger.debug(f"[JimengVideoNode] 生成视频请求参数: {json.dumps(params, ensure_ascii=False)}")
            logger.debug(f"[JimengVideoNode] 生成视频请求头: {json.dumps(headers, ensure_ascii=False)}")
            logger.debug(f"[JimengVideoNode] 生成视频请求体: {json.dumps(data, ensure_ascii=False)}")

            response = requests.post(url, headers=headers, params=params, json=data)
            if response.status_code != 200:
                error_msg = f"视频生成请求失败: HTTP {response.status_code} - {response.text}"
                logger.error(f"[JimengVideoNode] {error_msg}")
                return None

            result = response.json()
            if result.get("ret") != "0":
                # 打印完整的错误信息
                error_msg = f"API错误: {result.get('ret')} - {result.get('errmsg')}"
                details = result.get("data", {})
                logger.error(f"[JimengVideoNode] {error_msg}")
                logger.error(f"[JimengVideoNode] 错误详情: {json.dumps(details, ensure_ascii=False)}")
                return None

            return result

        except Exception as e:
            logger.error(f"[JimengVideoNode] Error in _generate_video: {str(e)}")
            return None

    def apply_image_upload(self, token_info, file_size):
        """申请图片上传
        Args:
            token_info: 上传token信息
            file_size: 文件大小
        Returns:
            dict: 上传信息
        """
        try:
            # Get current timestamp
            t = datetime.datetime.utcnow()
            amz_date = t.strftime('%Y%m%dT%H%M%SZ')
            datestamp = t.strftime('%Y%m%d')
            
            # Request parameters - 保持固定顺序
            request_parameters = {
                'Action': 'ApplyImageUpload',
                'FileSize': str(file_size),
                'ServiceId': token_info['space_name'],
                'Version': '2018-08-01'
            }
            
            # 构建规范请求字符串
            canonical_querystring = '&'.join([f'{k}={urllib.parse.quote(str(v))}' for k, v in sorted(request_parameters.items())])
            
            # 构建规范请求
            canonical_uri = '/'
            canonical_headers = (
                f'host:imagex.bytedanceapi.com\n'
                f'x-amz-date:{amz_date}\n'
                f'x-amz-security-token:{token_info["session_token"]}\n'
            )
            signed_headers = 'host;x-amz-date;x-amz-security-token'
            
            # 计算请求体哈希
            payload_hash = hashlib.sha256(b'').hexdigest()
            
            # 构建规范请求
            canonical_request = '\n'.join([
                'GET',
                canonical_uri,
                canonical_querystring,
                canonical_headers,
                signed_headers,
                payload_hash
            ])
            
            # 获取授权头
            authorization = self.get_authorization(
                token_info['access_key_id'],
                token_info['secret_access_key'],
                'cn-north-1',
                'imagex',
                amz_date,
                token_info['session_token'],
                signed_headers,
                canonical_request
            )
            
            # 设置请求头
            headers = {
                'Authorization': authorization,
                'X-Amz-Date': amz_date,
                'X-Amz-Security-Token': token_info['session_token'],
                'Host': 'imagex.bytedanceapi.com'
            }
            
            url = f'https://imagex.bytedanceapi.com/?{canonical_querystring}'
            
            response = requests.get(url, headers=headers)
            return response.json()
            
        except Exception as e:
            logger.error(f"[JimengVideoNode] Error in apply_image_upload: {str(e)}")
            return None
        
    def upload_image_file(self, upload_info, image_path):
        """上传图片文件
        Args:
            upload_info: 上传信息
            image_path: 图片路径
        Returns:
            dict: 上传结果
        """
        try:
            store_info = upload_info['Result']['UploadAddress']['StoreInfos'][0]
            upload_host = upload_info['Result']['UploadAddress']['UploadHosts'][0]
            
            url = f"https://{upload_host}/upload/v1/{store_info['StoreUri']}"
            
            # 计算文件的CRC32
            with open(image_path, 'rb') as f:
                content = f.read()
                crc32 = format(binascii.crc32(content) & 0xFFFFFFFF, '08x')
            
            headers = {
                'accept': '*/*',
                'authorization': store_info['Auth'],
                'content-type': 'application/octet-stream',
                'content-disposition': 'attachment; filename="undefined"',
                'content-crc32': crc32,
                'origin': 'https://jimeng.jianying.com',
                'referer': 'https://jimeng.jianying.com/'
            }
            
            response = requests.post(url, headers=headers, data=content)
            return response.json()
            
        except Exception as e:
            logger.error(f"[JimengVideoNode] Error in upload_image_file: {str(e)}")
            return None

    def commit_image_upload(self, token_info, upload_info):
        """提交图片上传"""
        amz_date = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        session_key = upload_info['Result']['UploadAddress']['SessionKey']
        
        url = f"https://{token_info['upload_domain']}"
        params = {
            "Action": "CommitImageUpload",
            "Version": "2018-08-01",
            "ServiceId": token_info['space_name']
        }
        
        data = {"SessionKey": session_key}
        payload = json.dumps(data)
        content_sha256 = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        
        # 构建规范请求
        canonical_uri = "/"
        canonical_querystring = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        signed_headers = "x-amz-content-sha256;x-amz-date;x-amz-security-token"
        canonical_headers = f"x-amz-content-sha256:{content_sha256}\nx-amz-date:{amz_date}\nx-amz-security-token:{token_info['session_token']}\n"
        
        canonical_request = f"POST\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{content_sha256}"
        
        authorization = self.get_authorization(
            token_info['access_key_id'],
            token_info['secret_access_key'],
            'cn-north-1',
            'imagex',
            amz_date,
            token_info['session_token'],
            signed_headers,
            canonical_request
        )
        
        headers = {
            'accept': '*/*',
            'content-type': 'application/json',
            'authorization': authorization,
            'x-amz-content-sha256': content_sha256,
            'x-amz-date': amz_date,
            'x-amz-security-token': token_info['session_token'],
            'origin': 'https://jimeng.jianying.com',
            'referer': 'https://jimeng.jianying.com/'
        }
        
        response = requests.post(f"{url}?{canonical_querystring}", headers=headers, data=payload)
        return response.json()

    def check_image_to_video_status(self, task_id, max_attempts=200, check_interval=3):
        """检查图生视频生成状态
        Args:
            task_id: 任务ID
            max_attempts: 最大尝试次数
            check_interval: 检查间隔(秒)
        Returns:
            tuple: (success, result)
        """
        try:
            attempt = 0
            last_status = None
            status_unchanged_count = 0
            
            while attempt < max_attempts:
                attempt += 1
                try:
                    # 获取token信息
                    token_info = self.token_manager.get_token("/mweb/v1/mget_generate_task")
                    if not token_info:
                        return False, "获取token失败"
                        
                    # 获取账号信息
                    account = self.token_manager.get_current_account()
                    if not account:
                        return False, "获取账号信息失败"

                    headers = {
                        'accept': 'application/json, text/plain, */*',
                        'accept-language': 'zh-CN,zh;q=0.9',
                        'app-sdk-version': '48.0.0',
                        'appid': '513695',
                        'appvr': '5.8.0',
                        'content-type': 'application/json',
                        'cookie': token_info["cookie"],
                        'device-time': token_info["device_time"],
                        'sign': token_info["sign"],
                        'sign-ver': '1',
                        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    }

                    data = {
                        "task_id_list": [task_id],
                        "http_common_info": {
                            "aid": "513695",
                            "app_id": "513695",
                            "user_id": account.get("user_id", ""),
                            "device_id": account.get("web_id", "")
                        }
                    }

                    try:
                        response = requests.post(
                            f"https://jimeng.jianying.com/mweb/v1/mget_generate_task?aid=513695",
                            headers=headers,
                            json=data,
                            timeout=10
                        )
                    except requests.exceptions.Timeout:
                        logger.error(f"[JimengVideoNode] Request timeout when checking status")
                        time.sleep(check_interval)
                        continue
                    except requests.exceptions.RequestException as e:
                        logger.error(f"[JimengVideoNode] Request error when checking status: {str(e)}")
                        time.sleep(check_interval)
                        continue
                    
                    if response.status_code != 200:
                        logger.error(f"[JimengVideoNode] 检查视频状态失败: HTTP {response.status_code} - {response.text}")
                        time.sleep(check_interval)
                        continue
                        
                    result = response.json()
                    if result.get("ret") != "0":
                        logger.error(f"[JimengVideoNode] 检查视频状态API返回错误: {result}")
                        time.sleep(check_interval)
                        continue

                    task_map = result.get("data", {}).get("task_map", {})
                    task_info = task_map.get(task_id)
                    
                    if not task_info:
                        logger.error(f"[JimengVideoNode] 未找到视频任务信息，task_id: {task_id}, 可用任务: {list(task_map.keys())}")
                        time.sleep(check_interval)
                        continue
                    
                    status = task_info.get("status", 0)

                    # 只在状态变化时输出日志
                    if status != last_status:
                        status_msg = {
                            0: "初始化中",
                            10: "排队中",
                            20: "准备中",
                            30: "生成中",
                            40: "处理中",
                            50: "生成成功",
                            60: "生成失败"
                        }.get(status, f"未知状态({status})")
                        logger.info(f"[JimengVideoNode] 视频状态: {status_msg}")
                        
                        # 在状态30时记录更多信息
                        if status == 30:
                            logger.info(f"[JimengVideoNode] 进入生成中状态，任务信息: {json.dumps(task_info, ensure_ascii=False)}")
                        elif status == 60:
                            logger.error(f"[JimengVideoNode] 进入失败状态，任务信息: {json.dumps(task_info, ensure_ascii=False)}")
                    
                    # 检查状态是否连续多次未变化
                    if status == last_status:
                        status_unchanged_count += 1
                        
                        # 根据不同状态设置不同的报告间隔和超时时间
                        if status == 20:  # 准备中状态
                            # 每30秒报告一次（30秒 / 3秒间隔 = 10次检查）
                            if status_unchanged_count % 10 == 0:
                                logger.info(f"[JimengVideoNode] 状态 {status}(准备中) 已持续 {status_unchanged_count * check_interval} 秒未变化")
                            # 300秒超时（300秒 / 3秒间隔 = 100次检查）
                            if status_unchanged_count >= 100:
                                logger.warning(f"[JimengVideoNode] 状态 {status}(准备中) 超时，已等待 {status_unchanged_count * check_interval} 秒")
                                return False, "视频生成准备超时（5分钟）"
                        elif status == 30:  # 生成中状态
                            # 每30秒报告一次（30秒 / 3秒间隔 = 10次检查）
                            if status_unchanged_count % 10 == 0:
                                logger.info(f"[JimengVideoNode] 状态 {status}(生成中) 已持续 {status_unchanged_count * check_interval} 秒未变化")
                            # 300秒超时（300秒 / 3秒间隔 = 100次检查）
                            if status_unchanged_count >= 100:
                                logger.warning(f"[JimengVideoNode] 状态 {status}(生成中) 超时，已等待 {status_unchanged_count * check_interval} 秒")
                                return False, "视频生成超时（5分钟），积分已返还，请重试"
                        else:
                            # 其他状态每30秒报告一次（30秒 / 3秒间隔 = 10次检查）
                            if status_unchanged_count % 10 == 0:
                                logger.info(f"[JimengVideoNode] 状态 {status} 已持续 {status_unchanged_count * check_interval} 秒未变化")
                        
                        # 其他状态的超时检查（100次检查，约5分钟）
                        if status not in [20, 30] and status_unchanged_count >= 100:
                            logger.warning(f"[JimengVideoNode] 状态 {status} 超时，已等待 {status_unchanged_count * check_interval} 秒")
                            if status == 0:
                                return False, "视频生成初始化超时"
                            elif status == 10:
                                return False, "视频生成排队超时"
                            elif status == 40:
                                return False, "视频处理超时"
                    else:
                        if status_unchanged_count > 0:
                            logger.info(f"[JimengVideoNode] 状态从 {last_status} 变更为 {status}，重置计数器")
                        status_unchanged_count = 0
                        last_status = status
                    
                    # 状态码说明:
                    # 0: 初始化
                    # 10: 排队中
                    # 20: 准备中
                    # 30: 生成中
                    # 40: 处理中
                    # 50: 成功
                    # 60: 失败
                    
                    if status == 50:  # 成功
                        item_list = task_info.get("item_list", [])
                        if not item_list:
                            return False, "视频生成成功但未返回URL"
                        
                        video = item_list[0].get("video", {})
                        if not video:
                            return False, "视频数据为空"
                            
                        # 获取视频链接，优先使用原始清晰度
                        video_info = video.get("transcoded_video", {}).get("origin", {})
                        if not video_info:
                            return False, "未获取到视频信息"

                        video_url = video_info.get("video_url")
                        if not video_url:
                            return False, "未获取到视频URL"
                            
                        logger.info(f"[JimengVideoNode] 视频生成完成，清晰度: {video_info.get('definition', 'origin')}, 大小: {video_info.get('size', 0)} bytes")
                        return True, video_url
                        
                    elif status == 60:  # 失败
                        task_payload = task_info.get("task_payload", {})
                        fail_code = task_payload.get("fail_code", "")
                        fail_msg = task_info.get("fail_msg", "")
                        
                        # 记录详细的失败信息
                        logger.error(f"[JimengVideoNode] 视频生成失败详情:")
                        logger.error(f"[JimengVideoNode] - 失败代码: {fail_code}")
                        logger.error(f"[JimengVideoNode] - 失败消息: {fail_msg}")
                        logger.error(f"[JimengVideoNode] - 任务载荷: {task_payload}")
                        logger.error(f"[JimengVideoNode] - 完整任务信息: {task_info}")
                        
                        error_msg = f"视频生成失败"
                        if fail_code:
                            error_msg += f"，错误码：{fail_code}"
                        if fail_msg:
                            error_msg += f"，错误信息：{fail_msg}"
                            
                        return False, error_msg
                    
                    # 如果状态不在已知范围内，记录警告
                    if status not in [0, 10, 20, 30, 40, 50, 60]:
                        logger.warning(f"[JimengVideoNode] 发现未知状态码: {status}，任务信息: {json.dumps(task_info, ensure_ascii=False)}")
                    
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"[JimengVideoNode] 检查视频状态出错: {str(e)}")
                    time.sleep(check_interval)
                    
            return False, "视频生成超时，请稍后重试"
            
        except Exception as e:
            logger.error(f"[JimengVideoNode] Error in check_image_to_video_status: {str(e)}")
            return False, str(e)
        
    def get_authorization(self, access_key, secret_key, region, service, amz_date, security_token, signed_headers, canonical_request):
        """生成AWS授权头"""
        algorithm = 'AWS4-HMAC-SHA256'
        datestamp = amz_date[:8]
        credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
        
        # Create string to sign
        string_to_sign = '\n'.join([
            algorithm,
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
        ])
        
        # Calculate signature
        k_date = self.sign(('AWS4' + secret_key).encode('utf-8'), datestamp)
        k_region = self.sign(k_date, region)
        k_service = self.sign(k_region, service)
        k_signing = self.sign(k_service, 'aws4_request')
        signature = hmac.new(k_signing,
                           string_to_sign.encode('utf-8'),
                           hashlib.sha256).hexdigest()
        
        # Create authorization header
        authorization_header = (
            f"{algorithm} "
            f"Credential={access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )
        
        return authorization_header
        
    def sign(self, key, msg):
        """计算HMAC-SHA256签名"""
        return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

    def text_to_video_generation(self, prompt, first_frame_info=None, duration_ms=5000, model="s2.0", ratio="4:3"):
        """生成视频（文生视频方法）"""
        try:
            # 打印当前账号状态
            current_account_number = self.token_manager.current_account_index + 1
            logger.info(f"[JimengVideoNode] 开始生成视频任务，使用账号{current_account_number}")
            
            # 获取当前账号信息
            account = self.token_manager.get_current_account()
            if not account:
                logger.error("[JimengVideoNode] 获取当前账号信息失败")
                return False, "获取当前账号信息失败"
                
            # 检查当前账号积分
            credit_info = self.token_manager.get_credit()
            if not credit_info:
                logger.error("[JimengVideoNode] 获取积分信息失败")
                return False, "获取积分信息失败"
                
            required_credit = 20 if model == "p2.0p" else 5
            logger.info(f"[JimengVideoNode] 账号{current_account_number}当前积分: {credit_info['total_credit']}, 需要积分: {required_credit}")
            
            if credit_info['total_credit'] < required_credit:
                error_msg = f"账号{current_account_number}积分不足，当前积分: {credit_info['total_credit']}, 需要积分: {required_credit}"
                logger.error(f"[JimengVideoNode] {error_msg}")
                return False, error_msg
                
            logger.info(f"[JimengVideoNode] 使用账号{current_account_number}生成视频，当前积分：{credit_info['total_credit']}")
            
            # 生成唯一的submit_id
            submit_id = str(uuid.uuid4())
            
            # 生成当前时间戳
            current_timestamp = int(time.time())
            
            # 获取当前账号的web_id
            web_id = account.get('web_id', str(random.random() * 999999999999999999 + 7000000000000000000))
            
            # 生成msToken (基于账号信息和时间戳)
            msToken_base = f"{account['sessionid']}_{current_timestamp}"
            msToken = base64.b64encode(msToken_base.encode()).decode()
            
            # 生成sign (基于账号信息和时间戳)
            sign_base = f"{account['sessionid']}_{current_timestamp}_video"
            sign = hashlib.md5(sign_base.encode()).hexdigest()
            
            # 生成a_bogus (基于账号信息和时间戳)
            a_bogus_base = f"{account['sessionid']}_{current_timestamp}_bogus"
            a_bogus = base64.b64encode(a_bogus_base.encode()).decode()
            
            # 根据模型设置不同的参数
            if model == "p2.0p":
                model_params = {
                    "model_req_key": "dreamina_ailab_generate_video_model_v1.4",
                    "video_mode": 2,
                    "fps": 12,
                    "template_id": "",
                    "lens_motion_type": "0",
                    "motion_speed": "0",
                    "ending_control": "0",
                    "benefit_type": "basic_video_operation_lab_14",
                    "feature_entrance": "to_video",
                    "feature_entrance_detail": "to_video-text_to_video"
                }
            else:  # s2.0 或 s2.0p
                # 区分s2.0和s2.0p
                if model == "s2.0p":
                    model_req_key = "dreamina_ic_generate_video_model_vgfm1.0"
                    benefit_type = "basic_video_operation_vgfm"
                else:  # s2.0
                    model_req_key = "dreamina_ic_generate_video_model_vgfm_lite"
                    benefit_type = "basic_video_operation_vgfm_lite"
                    
                model_params = {
                    "model_req_key": model_req_key,
                    "video_mode": 2,  # 文生视频应该用2
                    "fps": 24,
                    "template_id": "",
                    "lens_motion_type": "",
                    "motion_speed": "",
                    "ending_control": "",
                    "benefit_type": benefit_type,
                    "feature_entrance": "to_video",
                    "feature_entrance_detail": "to_video-text_to_video"
                }
            
            # 构建 babi_param - 修改feature_key为text_to_video
            babi_param = {
                "scenario": "image_video_generation",
                "feature_key": "text_to_video",
                "feature_entrance": model_params["feature_entrance"],
                "feature_entrance_detail": model_params["feature_entrance_detail"]
            }
            
            # 构建URL参数
            url_params = {
                "aid": "513695",
                "babi_param": json.dumps(babi_param),
                "device_platform": "web",
                "region": "CN",
                "web_id": web_id,
                "msToken": msToken,
                "a_bogus": a_bogus
            }
            
            # 准备请求数据 - 调整文生视频的请求结构
            generate_video_payload = {
                "submit_id": submit_id,
                "task_extra": json.dumps({
                    "promptSource": "custom",
                    "originSubmitId": str(uuid.uuid4()),
                    "isDefaultSeed": 1,
                    "originTemplateId": "",
                    "imageNameMapping": {},
                    "isUseAiGenPrompt": False,
                    "batchNumber": 1
                }),
                "http_common_info": {"aid": "513695"},
                "input": {
                    "video_aspect_ratio": ratio,
                    "seed": random.randint(1000000000, 9999999999),
                    "video_gen_inputs": [
                        {
                            "prompt": prompt,
                            "fps": model_params["fps"],
                            "duration_ms": duration_ms,
                            "video_mode": model_params["video_mode"],
                            "template_id": model_params["template_id"]
                        }
                    ],
                    "priority": 0,
                    "model_req_key": model_params["model_req_key"]
                },
                "mode": "workbench",
                "history_option": {},
                "commerce_info": {
                    "resource_id": "generate_video",
                    "resource_id_type": "str",
                    "resource_sub_type": "aigc",
                    "benefit_type": model_params["benefit_type"]
                },
                "client_trace_data": {}
            }

            # 使用当前账号的sessionid构建cookie
            cookie = f"sessionid={account['sessionid']}; sessionid_ss={account['sessionid']}; _tea_web_id={web_id}; web_id={web_id}; _v2_spipe_web_id={web_id}"
            
            # 使用最新的token发送请求
            headers = {
                "Authorization": f"Bearer {self.token_manager.get_token()}",
                "Content-Type": "application/json",
                "accept": "application/json, text/plain, */*",
                "accept-language": "zh-CN,zh;q=0.9",
                "app-sdk-version": "48.0.0",
                "appid": "513695",
                "appvr": "5.8.0",
                "device-time": str(current_timestamp),
                "lan": "zh-Hans",
                "loc": "cn",
                "origin": "https://jimeng.jianying.com",
                "pf": "7",
                "priority": "u=1, i",
                "referer": "https://jimeng.jianying.com/ai-tool/video/generate",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
                "sec-ch-ua": "\"Not A(Brand\";v=\"8\", \"Chromium\";v=\"132\", \"Android WebView\";v=\"132\"",
                "sec-ch-ua-mobile": "?1",
                "sec-ch-ua-platform": "\"Android\"",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "x-requested-with": "mark.via",
                "sign": sign,
                "sign-ver": "1",
                "Cookie": cookie
            }
            
            # 打印完整的请求信息
            logger.info("[JimengVideoNode] 生成视频请求信息:")
            logger.info(f"当前使用账号: {current_account_number}")
            logger.info(f"URL: https://jimeng.jianying.com/mweb/v1/generate_video")
            logger.info(f"URL Params: {json.dumps(url_params, ensure_ascii=False, indent=2)}")
            logger.info(f"Query Params: {json.dumps(generate_video_payload, ensure_ascii=False, indent=2)}")
            logger.info(f"Headers: {json.dumps(headers, ensure_ascii=False, indent=2)}")
            
            response = requests.post(
                "https://jimeng.jianying.com/mweb/v1/generate_video",
                params=url_params,
                headers=headers,
                json=generate_video_payload,
                timeout=10
            )
            
            # 打印响应信息
            logger.info(f"[JimengVideoNode] Response Status: {response.status_code}")
            logger.info(f"[JimengVideoNode] Response Headers: {json.dumps(dict(response.headers), ensure_ascii=False, indent=2)}")
            logger.info(f"[JimengVideoNode] Response Body: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
            
            if response.status_code != 200:
                logger.error(f"[JimengVideoNode] Video generation request failed: HTTP {response.status_code} - {response.text}")
                return False, f"视频生成请求失败，状态码：{response.status_code}"
                
            result = response.json()
            if result.get("ret") != "0":
                logger.error(f"[JimengVideoNode] API error: {result}")
                return False, f"API错误：{result.get('msg', '未知错误')}"
                
            task_id = result.get("data", {}).get("aigc_data", {}).get("task", {}).get("task_id")
            if not task_id:
                logger.error(f"[JimengVideoNode] Task ID not found in response: {result}")
                return False, "未获取到任务ID"
                
            # 检查视频生成状态
            success, result = self.check_text_to_video_status(task_id)
            if not success:
                return False, result
                
            return True, result
            
        except Exception as e:
            logger.error(f"[JimengVideoNode] Error generating video: {str(e)}")
            return False, f"视频生成失败: {str(e)}"

    def check_text_to_video_status(self, task_id, max_retries=None):
        """检查文生视频生成状态"""
        try:
            # 获取当前账号信息
            current_account_number = self.token_manager.current_account_index + 1
            account = self.token_manager.get_current_account()
            if not account:
                return False, "获取当前账号信息失败"
                
            logger.info(f"[JimengVideoNode] 使用账号{current_account_number}检查视频状态")
            
            # 准备请求参数
            url = "https://jimeng.jianying.com/mweb/v1/mget_generate_task"
            params = {
                "aid": "513695",
                "device_platform": "web",
                "region": "CN",
                "web_id": account.get('web_id', '')
            }
            
            # 准备请求体
            data = {
                "task_id_list": [task_id],
                "http_common_info": {"aid": "513695"}
            }
            
            # 获取token信息
            token_info = self.token_manager.get_token()
            
            # 从配置文件读取超时参数
            if max_retries is None:
                timeout_config = self.config.get("timeout", {})
                max_wait_time = timeout_config.get("max_wait_time", 300)  # 默认5分钟
                check_interval = timeout_config.get("check_interval", 5)  # 默认5秒间隔
                max_retries = max_wait_time // check_interval
            else:
                check_interval = 5  # 默认5秒间隔
                
            # 轮询检查视频状态 - 配置化超时机制
            logger.info(f"[JimengVideoNode] 开始轮询视频生成状态，最大等待时间: {max_retries * check_interval}秒")
            for i in range(max_retries):
                try:
                    # 准备请求头
                    headers = {
                        'accept': 'application/json, text/plain, */*',
                        'accept-language': 'zh-CN,zh;q=0.9',
                        'app-sdk-version': '48.0.0',
                        'appid': '513695',
                        'appvr': '5.8.0',
                        'content-type': 'application/json',
                        'cookie': token_info["cookie"],
                        'device-time': token_info["device_time"],
                        'sign': token_info["sign"],
                        'sign-ver': '1',
                        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
                    }
                    
                    response = requests.post(
                        url,
                        params=params,
                        headers=headers,
                        json=data,
                        timeout=10
                    )
                    
                    # 打印响应信息用于调试
                    logger.debug(f"[JimengVideoNode] 状态检查响应: {response.text}")
                    
                    if response.status_code != 200:
                        logger.error(f"[JimengVideoNode] 状态检查失败: HTTP {response.status_code} - {response.text}")
                        time.sleep(3)
                        continue
                        
                    result = response.json()
                    if result.get("ret") != "0":
                        logger.error(f"[JimengVideoNode] API错误: {result}")
                        time.sleep(3)
                        continue
                    
                    # 从task_map获取视频任务信息
                    task_map = result.get("data", {}).get("task_map", {})
                    task_info = task_map.get(task_id)
                    
                    if not task_info:
                        logger.error(f"[JimengVideoNode] 未找到任务信息: {task_id}")
                        time.sleep(3)
                        continue
                    
                    # 获取任务状态
                    status = task_info.get("status")
                    logger.info(f"[JimengVideoNode] 视频状态: {status}")
                    
                    if status == 50:  # 生成成功
                        item_list = task_info.get("item_list", [])
                        if not item_list:
                            logger.error("[JimengVideoNode] 没有找到视频内容")
                            time.sleep(3)
                            continue
                            
                        # 从第一个item中获取视频信息
                        video = item_list[0].get("video", {})
                        if not video:
                            logger.error("[JimengVideoNode] 没有找到视频对象")
                            time.sleep(3)
                            continue
                            
                        # 从transcoded_video中获取origin视频URL
                        transcoded_video = video.get("transcoded_video", {})
                        origin_video = transcoded_video.get("origin", {})
                        
                        video_url = origin_video.get("video_url")
                        if not video_url:
                            logger.error("[JimengVideoNode] 没有找到视频URL")
                            time.sleep(3)
                            continue
                            
                        return True, video_url
                        
                    elif status == 60:  # 生成失败
                        fail_msg = task_info.get("fail_msg", "未知错误")
                        logger.error(f"[JimengVideoNode] 视频生成失败: {fail_msg}")
                        return False, f"视频生成失败: {fail_msg}"
                        
                    # 继续等待，每30秒显示一次进度
                    if (i + 1) % 10 == 0:  # 每10次循环（30秒）显示一次进度
                        elapsed_time = (i + 1) * check_interval
                        total_time = max_retries * check_interval
                        logger.info(f"[JimengVideoNode] 视频生成中... 已等待 {elapsed_time}秒/{total_time}秒")
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"[JimengVideoNode] 检查状态出错: {str(e)}")
                    time.sleep(3)
                    
            logger.error(f"[JimengVideoNode] 视频生成超时，已等待 {max_retries * check_interval}秒")
            return False, f"视频生成超时，已等待 {max_retries * check_interval}秒，请稍后查看结果"
            
        except Exception as e:
            logger.error(f"[JimengVideoNode] 检查视频状态异常: {str(e)}")
            return False, f"检查视频状态失败: {str(e)}"

    def frames_to_video(self, first_image_path, end_image_path, prompt="镜头匀速推进", duration_ms=5000, lens_motion_type="ZoomIn", motion_speed="Moderate", camera_strength="1", model="v3.0"):
        """首尾帧生成视频 - 使用最新的3.0模型
        Args:
            first_image_path: 首帧图片路径
            end_image_path: 尾帧图片路径
            prompt: 提示词，默认"镜头匀速推进"
            duration_ms: 视频时长（毫秒），支持5000/10000
            lens_motion_type: 镜头运动类型（3.0版本已移除此参数）
            motion_speed: 运动速度（3.0版本已移除此参数）
            camera_strength: 镜头强度（3.0版本已移除此参数）
            model: 视频模型，默认"v3.0"，仅支持"v3.0"
        Returns:
            tuple: (success, result)
        """
        try:
            logger.info("[Jimeng] 开始首尾帧视频3.0生成")
            
            # 获取上传token
            token_info = self.token_manager.get_upload_token()
            if not token_info:
                return False, "获取上传token失败"
            
            # 上传首帧图片
            first_frame_uri = self._upload_frame_image(first_image_path, token_info)
            if not first_frame_uri:
                return False, "上传首帧图片失败"
            
            # 上传尾帧图片
            end_frame_uri = self._upload_frame_image(end_image_path, token_info)
            if not end_frame_uri:
                return False, "上传尾帧图片失败"
            
            logger.info(f"[Jimeng] 图片上传成功，首帧URI: {first_frame_uri}, 尾帧URI: {end_frame_uri}")
            
            # 获取图片尺寸
            from PIL import Image
            with Image.open(first_image_path) as first_img:
                first_width, first_height = first_img.size
            with Image.open(end_image_path) as end_img:
                end_width, end_height = end_img.size
            
            logger.info(f"[Jimeng] 首帧尺寸: {first_width}x{first_height}, 尾帧尺寸: {end_width}x{end_height}")
            logger.info(f"[Jimeng] 生成参数: prompt='{prompt}', duration_ms={duration_ms}, model='{model}'")
            
            # 生成视频
            success, result = self._generate_frames_video(
                first_frame_uri, end_frame_uri, first_width, first_height, end_width, end_height,
                prompt, duration_ms, model
            )
            
            return success, result
            
        except Exception as e:
            logger.error(f"[Jimeng] 首尾帧视频生成过程出错: {str(e)}")
            return False, str(e)
    
    def _upload_frame_image(self, image_path, token_info):
        """上传帧图片，复用现有的上传逻辑
        Args:
            image_path: 图片路径
            token_info: 上传token信息
        Returns:
            str: 图片URI
        """
        try:
            # 获取文件大小
            file_size = os.path.getsize(image_path)
            
            # 申请上传
            upload_info = self.apply_image_upload(token_info, file_size)
            if not upload_info or "Result" not in upload_info:
                return None
            
            # 上传图片
            file_response = self.upload_image_file(upload_info, image_path)
            if not file_response:
                return None
            
            # 提交上传
            commit_response = self.commit_image_upload(token_info, upload_info)
            if not commit_response or "Result" not in commit_response:
                return None
            
            # 获取图片URI
            plugin_result = commit_response['Result']['PluginResult'][0]
            return plugin_result['ImageUri']
            
        except Exception as e:
            logger.error(f"[Jimeng] Error uploading frame image: {str(e)}")
            return None
    
    def _generate_frames_video(self, first_frame_uri, end_frame_uri, first_width, first_height, end_width, end_height, prompt, duration_ms, model):
        """生成首尾帧视频 - 使用3.0版本API
        Args:
            first_frame_uri: 首帧图片URI
            end_frame_uri: 尾帧图片URI
            first_width: 首帧图片宽度
            first_height: 首帧图片高度
            end_width: 尾帧图片宽度
            end_height: 尾帧图片高度
            prompt: 提示词
            duration_ms: 视频时长（毫秒）
            model: 视频模型
        Returns:
            tuple: (success, result)
        """
        try:
            # 获取token信息
            token_info = self.token_manager.get_token("/mweb/v1/aigc_draft/generate")
            if not token_info:
                return False, "获取token失败"
                
            # 获取账号信息
            account = self.token_manager.get_current_account()
            if not account:
                return False, "获取账号信息失败"

            # 3.0版本固定参数
            model_req_key = "dreamina_ic_generate_video_model_vgfm_3.0"
            benefit_type = "basic_video_operation_vgfm_v_three"

            # 构建请求参数
            babi_param = {
                "scenario": "image_video_generation",
                "feature_key": "image_to_video",
                "feature_entrance": "to_video",
                "feature_entrance_detail": "to_video-image_to_video"
            }
            
            params = {
                "aid": "513695",
                "babi_param": json.dumps(babi_param),
                "device_platform": "web",
                "region": "CN",
                "web_id": self.token_manager.get_web_id(),
                "da_version": "3.2.5",
                "aigc_features": "app_lip_sync",
                "msToken": token_info.get("msToken", ""),
                "a_bogus": token_info.get("a_bogus", "")
            }

            # 生成UUID
            submit_id = str(uuid.uuid4())
            draft_id = str(uuid.uuid4())
            main_component_id = str(uuid.uuid4())
            metadata_id = str(uuid.uuid4())
            abilities_id = str(uuid.uuid4())
            gen_video_id = str(uuid.uuid4())
            text_to_video_params_id = str(uuid.uuid4())
            video_gen_input_id = str(uuid.uuid4())
            first_frame_image_id = str(uuid.uuid4())
            end_frame_image_id = str(uuid.uuid4())
            
            # 计算视频比例
            video_aspect_ratio = "9:16"  # 默认比例
            if first_width and first_height:
                ratio = first_width / first_height
                if ratio > 1.7:
                    video_aspect_ratio = "16:9"
                elif ratio > 1.2:
                    video_aspect_ratio = "4:3"
                elif ratio > 0.9:
                    video_aspect_ratio = "1:1"
                elif ratio > 0.7:
                    video_aspect_ratio = "3:4"
                else:
                    video_aspect_ratio = "9:16"

            # 构建复杂的draft_content结构
            draft_content = {
                "type": "draft",
                "id": draft_id,
                "min_version": "3.0.5",
                "min_features": [],
                "is_from_tsn": True,
                "version": "3.2.5",
                "main_component_id": main_component_id,
                "component_list": [
                    {
                        "type": "video_base_component",
                        "id": main_component_id,
                        "min_version": "1.0.0",
                        "metadata": {
                            "type": "",
                            "id": metadata_id,
                            "created_platform": 3,
                            "created_platform_version": "",
                            "created_time_in_ms": str(int(time.time() * 1000)),
                            "created_did": ""
                        },
                        "generate_type": "gen_video",
                        "aigc_mode": "workbench",
                        "abilities": {
                            "type": "",
                            "id": abilities_id,
                            "gen_video": {
                                "type": "",
                                "id": gen_video_id,
                                "text_to_video_params": {
                                    "type": "",
                                    "id": text_to_video_params_id,
                                    "video_gen_inputs": [
                                        {
                                            "type": "",
                                            "id": video_gen_input_id,
                                            "min_version": "3.0.5",
                                            "prompt": prompt,
                                            "first_frame_image": {
                                                "type": "image",
                                                "id": first_frame_image_id,
                                                "source_from": "upload",
                                                "platform_type": 1,
                                                "name": "",
                                                "image_uri": first_frame_uri,
                                                "width": first_width,
                                                "height": first_height,
                                                "format": "",
                                                "uri": first_frame_uri
                                            },
                                            "end_frame_image": {
                                                "type": "image",
                                                "id": end_frame_image_id,
                                                "source_from": "upload",
                                                "platform_type": 1,
                                                "name": "",
                                                "image_uri": end_frame_uri,
                                                "width": end_width,
                                                "height": end_height,
                                                "format": "",
                                                "uri": end_frame_uri
                                            },
                                            "ending_control": "1.0",
                                            "video_mode": 2,
                                            "fps": 24,
                                            "duration_ms": duration_ms
                                        }
                                    ],
                                    "video_aspect_ratio": video_aspect_ratio,
                                    "seed": random.randint(1000000000, 9999999999),
                                    "model_req_key": model_req_key
                                },
                                "video_task_extra": json.dumps({
                                    "promptSource": "custom",
                                    "originSubmitId": str(uuid.uuid4()),
                                    "isDefaultSeed": 1,
                                    "imageNameMapping": {}
                                })
                            }
                        },
                        "process_type": 1
                    }
                ]
            }

            # 构建请求体
            data = {
                "extend": {
                    "m_video_commerce_info": {
                        "resource_id": "generate_video",
                        "resource_id_type": "str",
                        "resource_sub_type": "aigc",
                        "benefit_type": benefit_type
                    },
                    "root_model": model_req_key,
                    "history_option": {}
                },
                "submit_id": submit_id,
                "metrics_extra": json.dumps({
                    "promptSource": "custom",
                    "originSubmitId": str(uuid.uuid4()),
                    "isDefaultSeed": 1,
                    "imageNameMapping": {}
                }),
                "draft_content": json.dumps(draft_content)
            }

            # 构建请求头
            headers = {
                'accept': 'application/json, text/plain, */*',
                'accept-language': 'zh-CN,zh;q=0.9',
                'app-sdk-version': '48.0.0',
                'appid': '513695',
                'appvr': '5.8.0',
                'content-type': 'application/json',
                'cookie': token_info["cookie"],
                'device-time': token_info["device_time"],
                'lan': 'zh-Hans',
                'loc': 'cn',
                'origin': 'https://jimeng.jianying.com',
                'pf': '7',
                'priority': 'u=1, i',
                'referer': 'https://jimeng.jianying.com/ai-tool/video/generate',
                'sign': token_info["sign"],
                'sign-ver': '1',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Microsoft Edge";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin'
            }

            url = "https://jimeng.jianying.com/mweb/v1/aigc_draft/generate"

            # 打印详细的请求信息
            logger.info(f"[Jimeng] 首尾帧视频3.0请求URL: {url}")
            logger.info(f"[Jimeng] 首尾帧视频3.0请求参数: {json.dumps(params, ensure_ascii=False)}")
            logger.debug(f"[Jimeng] 首尾帧视频3.0请求头: {json.dumps(headers, ensure_ascii=False)}")
            logger.debug(f"[Jimeng] 首尾帧视频3.0请求体: {json.dumps(data, ensure_ascii=False)}")

            response = requests.post(url, headers=headers, params=params, json=data)
            
            if response.status_code != 200:
                error_msg = f"首尾帧视频3.0生成请求失败: HTTP {response.status_code} - {response.text}"
                logger.error(f"[Jimeng] {error_msg}")
                return False, error_msg

            result = response.json()
            logger.debug(f"[Jimeng] 首尾帧视频3.0响应内容: {json.dumps(result, ensure_ascii=False)}")
            
            if result.get("ret") != "0":
                error_msg = f"API错误: {result.get('ret')} - {result.get('errmsg')}"
                logger.error(f"[Jimeng] {error_msg}")
                return False, error_msg

            # 获取history_record_id用于查询结果
            history_record_id = result.get("data", {}).get("aigc_data", {}).get("history_record_id")
            if not history_record_id:
                return False, "未获取到历史记录ID"
            
            logger.info(f"[Jimeng] 首尾帧视频3.0生成任务创建成功，history_record_id: {history_record_id}")
            
            # 使用新的查询方法检查视频状态
            success, result = self._check_frames_video_status(history_record_id)
            return success, result

        except Exception as e:
            logger.error(f"[Jimeng] Error in _generate_frames_video: {str(e)}")
            return False, str(e)
    
    def _check_frames_video_status(self, history_record_id, max_retries=30):
        """检查首尾帧视频3.0生成状态
        Args:
            history_record_id: 历史记录ID
            max_retries: 最大重试次数
        Returns:
            tuple: (success, result)
        """
        try:
            for i in range(max_retries):
                try:
                    # 获取token信息
                    token_info = self.token_manager.get_token("/mweb/v1/get_history_by_ids")
                    if not token_info:
                        time.sleep(3)
                        continue
                    
                    # 构建请求参数
                    params = {
                        "aid": "513695",
                        "device_platform": "web",
                        "region": "CN",
                        "web_id": self.token_manager.get_web_id(),
                        "da_version": "3.2.5",
                        "aigc_features": "app_lip_sync",
                        "msToken": token_info.get("msToken", ""),
                        "a_bogus": token_info.get("a_bogus", "")
                    }
                    
                    # 构建请求体
                    data = {
                        "history_ids": [history_record_id]
                    }
                    
                    # 构建请求头
                    headers = {
                        'accept': 'application/json, text/plain, */*',
                        'accept-language': 'zh-CN,zh;q=0.9',
                        'app-sdk-version': '48.0.0',
                        'appid': '513695',
                        'appvr': '5.8.0',
                        'content-type': 'application/json',
                        'cookie': token_info["cookie"],
                        'device-time': token_info["device_time"],
                        'lan': 'zh-Hans',
                        'loc': 'cn',
                        'origin': 'https://jimeng.jianying.com',
                        'pf': '7',
                        'priority': 'u=1, i',
                        'referer': 'https://jimeng.jianying.com/ai-tool/video/generate',
                        'sign': token_info["sign"],
                        'sign-ver': '1',
                        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
                        'sec-ch-ua': '"Microsoft Edge";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
                        'sec-ch-ua-mobile': '?0',
                        'sec-ch-ua-platform': '"Windows"',
                        'sec-fetch-dest': 'empty',
                        'sec-fetch-mode': 'cors',
                        'sec-fetch-site': 'same-origin'
                    }
                    
                    url = "https://jimeng.jianying.com/mweb/v1/get_history_by_ids"
                    
                    response = requests.post(url, headers=headers, params=params, json=data, timeout=10)
                    
                    if response.status_code != 200:
                        logger.debug(f"[Jimeng] 状态检查失败: HTTP {response.status_code}")
                        time.sleep(3)
                        continue
                        
                    result = response.json()
                    if result.get("ret") != "0":
                        logger.debug(f"[Jimeng] 状态检查API错误: {result}")
                        time.sleep(3)
                        continue
                    
                    # 解析响应
                    data_result = result.get("data", {})
                    history_data = data_result.get(history_record_id)
                    if not history_data:
                        logger.debug(f"[Jimeng] 未找到历史数据: {history_record_id}")
                        time.sleep(3)
                        continue
                    
                    # 检查状态
                    status = history_data.get("status", 0)
                    logger.info(f"[Jimeng] 首尾帧视频3.0状态: {status}")
                    
                    if status == 50:  # 生成成功
                        # 获取视频URL
                        item_list = history_data.get("item_list", [])
                        if item_list:
                            item = item_list[0]
                            video_info = item.get("video", {})
                            transcoded_video = video_info.get("transcoded_video", {})
                            origin_video = transcoded_video.get("origin", {})
                            video_url = origin_video.get("video_url")
                            
                            if video_url:
                                logger.info(f"[Jimeng] 首尾帧视频3.0生成成功: {video_url}")
                                return True, video_url
                        
                        return False, "未找到视频URL"
                        
                    elif status == 60:  # 生成失败
                        fail_msg = history_data.get("fail_msg", "生成失败")
                        logger.error(f"[Jimeng] 首尾帧视频3.0生成失败: {fail_msg}")
                        return False, f"首尾帧视频生成失败: {fail_msg}"
                    
                    # 其他状态，继续等待
                    time.sleep(3)
                    
                except Exception as e:
                    logger.debug(f"[Jimeng] 检查状态出错: {str(e)}")
                    time.sleep(3)
                    
            return False, "首尾帧视频生成超时"
            
        except Exception as e:
            logger.error(f"[Jimeng] 检查首尾帧视频3.0状态异常: {str(e)}")
            return False, f"检查视频状态失败: {str(e)}"

# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "Jimeng_Video": JimengVideoNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Jimeng_Video": "即梦AI视频"
}
