import requests
import json
import logging
import os
import time
import uuid
import random
import hashlib
import hmac
import binascii
import datetime
import urllib.parse
import torch
import numpy as np
from PIL import Image
import io
from typing import Dict, Optional, Any, Tuple, List

# 确保从同级目录导入
from .token_manager import TokenManager

logger = logging.getLogger(__name__)

class ApiClient:
    def __init__(self, token_manager, config):
        self.token_manager = token_manager
        self.config = config
        self.temp_files = []
        self.base_url = "https://jimeng.jianying.com"  # 改回正确的域名
        self.aid = "513695"
        self.app_version = "5.8.0"

    def _get_headers(self, uri="/"):
        """获取请求头"""
        token_info = self.token_manager.get_token(uri)
        # 统一使用与成功请求一致的最小必要头，并在生成接口时移除 msToken/a-bogus（改用URL参数）
        headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'zh-CN,zh;q=0.9',
            'app-sdk-version': '48.0.0',
            'appid': self.aid,
            'appvr': self.app_version,
            'content-type': 'application/json',
            'cookie': token_info.get("cookie", ""),
            'device-time': token_info.get("device_time", ""),
            'lan': 'zh-Hans',
            'loc': 'cn',
            'origin': 'https://jimeng.jianying.com',
            'pf': '7',
            'priority': 'u=1, i',
            # 参照微信机器人版与线上成功请求
            'referer': 'https://jimeng.jianying.com/ai-tool/generate',
            'sec-ch-ua': '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'sign': token_info.get("sign", ""),
            'sign-ver': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        }
        # 仅非生成接口保留 header 中的 msToken/a-bogus
        if uri != '/mweb/v1/aigc_draft/generate':
            if token_info.get("msToken"):
                headers['msToken'] = token_info["msToken"]
            if token_info.get("a_bogus"):
                headers['a-bogus'] = token_info["a_bogus"]
        return headers

    def _send_request(self, method, url, **kwargs):
        """发送HTTP请求"""
        try:
            # 获取URI
            uri = url.split(self.base_url)[-1].split('?')[0]
            
            # 获取headers
            headers = self._get_headers(uri)
            
            # 如果kwargs中有headers，合并它们
            if 'headers' in kwargs:
                headers.update(kwargs.pop('headers'))
            
            kwargs['headers'] = headers
            
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            
            # 记录请求和响应信息
            logger.debug(f"[Jimeng] Request URL: {url}")
            logger.debug(f"[Jimeng] Request headers: {headers}")
            if 'params' in kwargs:
                logger.debug(f"[Jimeng] Request params: {kwargs['params']}")
            if 'json' in kwargs:
                logger.debug(f"[Jimeng] Request data: {kwargs['json']}")
            logger.debug(f"[Jimeng] Response: {response.text}")
            return response.json()
        except Exception as e:
            logger.error(f"[Jimeng] Request failed: {e}")
            return None

    def generate_t2i(self, prompt: str, model: str, ratio: str, seed: int = -1):
        """处理文生图请求
        Args:
            prompt: 提示词
            model: 模型名称
            ratio: 图片比例
        Returns:
            dict: 包含生成的图片URL列表
        """
        try:
            # 获取实际的模型key
            model = self._get_model_key(model)
            
            # 获取图片尺寸
            width, height = self._get_ratio_dimensions(ratio)
            
            # 生成随机种子
            seed = random.randint(1, 999999999)
            
            # 准备请求数据
            url = f"{self.base_url}/mweb/v1/aigc_draft/generate"
            
            # 获取模型配置
            models = self.config.get("params", {}).get("models", {})
            model_info = models.get(model, {})
            model_req_key = model_info.get("model_req_key", f"high_aes_general_v20:general_{model}")
            
            # 准备babi_param - 简化版本，与curl文件保持一致
            babi_param = {
                "scenario": "image_video_generation",
                "feature_key": "aigc_to_image",
                "feature_entrance": "to_image",
                "feature_entrance_detail": "to_image"
            }
            
            # 生成唯一的submit_id
            submit_id = str(uuid.uuid4())
            draft_id = str(uuid.uuid4())
            component_id = str(uuid.uuid4())
            
            # 准备metrics_extra
            metrics_extra = {
                "templateId": "",
                "generateCount": 1,
                "promptSource": "custom",
                "templateSource": "",
                "lastRequestId": "",
                "originRequestId": "",
                "originSubmitId": "",
                "isDefaultSeed": 1,
                "originTemplateId": "",
                "imageNameMapping": {},
                "isUseAiGenPrompt": False,
                "batchNumber": 1
            }
            
            data = {
                "extend": {
                    "root_model": model_req_key,
                    "template_id": ""
                },
                "submit_id": submit_id,
                "metrics_extra": json.dumps(metrics_extra),
                "draft_content": json.dumps({
                    "type": "draft",
                    "id": draft_id,
                    "min_version": "3.0.2",
                    "min_features": [],
                    "is_from_tsn": True,
                    "version": "3.2.8",  # 更新版本号
                    "main_component_id": component_id,
                    "component_list": [{
                        "type": "image_base_component",
                        "id": component_id,
                        "min_version": "3.0.2",
                        "aigc_mode": "workbench",
                        "metadata": {  # 添加metadata字段
                            "type": "",
                            "id": str(uuid.uuid4()),
                            "created_platform": 3,
                            "created_platform_version": "",
                            "created_time_in_ms": str(int(time.time() * 1000)),
                            "created_did": ""
                        },
                        "generate_type": "generate",
                        "abilities": {
                            "type": "",
                            "id": str(uuid.uuid4()),
                            "generate": {
                                "type": "",
                                "id": str(uuid.uuid4()),
                                "core_param": {
                                    "type": "",
                                    "id": str(uuid.uuid4()),
                                    "model": model_req_key,
                                    "prompt": prompt,
                                    "negative_prompt": "",
                                    "seed": seed,
                                    "sample_strength": 0.5,
                                    "image_ratio": 3 if ratio == "9:16" else self._get_ratio_value(ratio),
                                    "large_image_info": {
                                        "type": "",
                                        "id": str(uuid.uuid4()),
                                        "height": height,
                                        "width": width,
                                        "resolution_type": "1k"  # 添加分辨率类型
                                    }
                                },
                                "history_option": {
                                    "type": "",
                                    "id": str(uuid.uuid4())
                                }
                            }
                        }
                    }]
                }),
                "http_common_info": {"aid": self.aid}
            }
            
            # 将 msToken 与 a_bogus 放入 URL 参数（对齐线上成功请求）
            token_info = self.token_manager.get_token('/mweb/v1/aigc_draft/generate')
            params = {
                "babi_param": json.dumps(babi_param),
                "aid": self.aid,
                "device_platform": "web",
                "region": "CN",
                "web_id": self.token_manager.get_web_id(),
            }
            # 仅当可用时附加
            if token_info.get("msToken"):
                params["msToken"] = token_info["msToken"]
            if token_info.get("a_bogus"):
                params["a_bogus"] = token_info["a_bogus"]

            # 发送请求
            logger.debug(f"[Jimeng] Generating image with prompt: {prompt}, model: {model}, ratio: {ratio}")
            response = self._send_request("POST", url, params=params, json=data)

            # 透传非零 ret 的错误详情，便于节点层展示真实原因
            if not response or str(response.get('ret')) != '0':
                logger.error(f"[Jimeng] Failed to generate image: {response}")
                return {"error": True, "response": response}
                
            # 获取history_id
            history_id = response.get('data', {}).get('aigc_data', {}).get('history_record_id')
            if not history_id:
                logger.error("[Jimeng] No history_id in response")
                return {"error": True, "response": response}
                
            # 从配置文件读取超时参数
            timeout_config = self.config.get("timeout", {})
            max_wait_time = timeout_config.get("max_wait_time", 300)  # 默认5分钟
            check_interval = timeout_config.get("check_interval", 5)  # 默认5秒间隔
            max_retries = max_wait_time // check_interval
            
            logger.info(f"[Jimeng] 开始轮询图片生成状态，最大等待时间: {max_wait_time}秒")
            
            # 立即获取一次状态，检查排队信息
            first_check_result = self._get_generated_images(history_id)
            queue_info = self._get_queue_info_from_response(history_id)
            
            # 如果有排队信息且图片未生成完成，立即返回排队信息
            if queue_info and not first_check_result:
                queue_msg = self._format_queue_message(queue_info)
                logger.info(f"[Jimeng] {queue_msg}")
                # 立即返回排队信息，让用户知道需要等待多久
                return {
                    "is_queued": True,
                    "queue_message": queue_msg,
                    "history_id": history_id
                }
            
            if first_check_result:
                logger.info("[Jimeng] 图片生成成功，无需等待")
                return {"urls": first_check_result, "history_record_id": history_id}
            
            for attempt in range(max_retries):
                time.sleep(check_interval)
                image_urls = self._get_generated_images(history_id)
                if image_urls:
                    elapsed_time = (attempt + 1) * check_interval
                    logger.info(f"[Jimeng] 图片生成成功，总耗时: {elapsed_time}秒")
                    return {"urls": image_urls, "history_record_id": history_id}
                    
                # 每30秒输出一次进度日志
                if (attempt + 1) % 10 == 0:
                    elapsed_time = (attempt + 1) * check_interval
                    logger.info(f"[Jimeng] 图片生成中... 已等待 {elapsed_time}秒/{max_wait_time}秒")
                    
            logger.error(f"[Jimeng] 图片生成超时，已等待 {max_wait_time}秒")
            return None
            
        except Exception as e:
            logger.error(f"[Jimeng] Error generating image: {e}")
            return None

    def generate_i2i(self, image: torch.Tensor = None, images: Optional[List[torch.Tensor]] = None, prompt: str = "", model: str = "3.0", ratio: str = "1:1", seed: int = -1, num_images: int = 4) -> Tuple[torch.Tensor, str, str]:
        """处理图生图请求"""
        try:
            if not self.token_manager:
                return self._create_error_result("插件未正确初始化，请检查后台日志。")
            
            if not self._is_configured():
                return self._create_error_result("插件未配置，请在 config.json 中至少填入一个账号的 sessionid。")
            
            if not prompt or not prompt.strip():
                return self._create_error_result("提示词不能为空。")

            # 积分检查 - 图生图需要2积分
            if not self.token_manager.find_account_with_sufficient_credit(2):
                 return self._create_error_result("所有账号积分均不足2点，无法生成。")

            # 保存输入图像（支持多图）
            input_tensors = images if images and len(images) > 0 else ([image] if image is not None else [])
            if not input_tensors:
                return self._create_error_result("未提供参考图，请至少选择一张参考图。")

            input_image_paths = []
            for t in input_tensors:
                p = self._save_input_image(t)
                if not p:
                    return self._create_error_result("保存输入图像失败。")
                input_image_paths.append(p)

            logger.debug(f"[Jimeng] 开始图生图: {prompt[:50]}...，参考图数量: {len(input_image_paths)}")
            result = self.upload_image_and_generate_with_reference(
                image_paths=input_image_paths,
                prompt=prompt,
                model=model,
                ratio=ratio
            )
            
            if not result:
                return self._create_error_result("API 调用失败，返回为空。请检查网络、防火墙或账号配置。")
            
            # 检查是否是排队模式
            if result.get("is_queued"):
                history_id = result.get("history_id")
                queue_msg = result.get("queue_message", "任务已进入队列，请等待...")
                logger.debug(f"[Jimeng] {queue_msg}")
                
                # 开始轮询等待
                timeout_config = self.config.get("timeout", {})
                max_wait_time = timeout_config.get("max_wait_time", 300)
                check_interval = timeout_config.get("check_interval", 5)
                max_retries = max_wait_time // check_interval
                
                for attempt in range(max_retries):
                    time.sleep(check_interval)
                    # 优先按 submit_id 轮询（官网记录页按提交ID归档）
                    submit_id = result.get("submit_id")
                    image_urls = self._get_generated_images_by_submit_id(submit_id) if submit_id else None
                    if not image_urls:
                        # 回落到按 history_id 轮询
                        image_urls = self._get_generated_images_by_history_id(history_id)
                    if image_urls:
                        urls_to_download = image_urls[:num_images]
                        # 统一尺寸到当前比例对应的宽高，确保 torch.cat 不会因尺寸不一致报错
                        width, height = self._get_ratio_dimensions(ratio)
                        images = self._download_images(urls_to_download, target_size=(width, height))
                        if not images:
                            return self._create_error_result("下载图片失败，可能链接已失效。")
                        
                        image_batch = torch.cat(images, dim=0)
                        generation_info = self._generate_info_text(prompt, model, ratio, len(images))
                        # 为返回的URL追加history_id，提升可见性与可追踪性
                        urls_with_history = [(u + f"&history_id={history_id}") if "?" in u else (u + f"?history_id={history_id}") for u in urls_to_download]
                        image_urls_text = "\n\n".join(urls_with_history)
                        
                        # 清理临时文件
                        try:
                            for _p in input_image_paths:
                                os.remove(_p)
                        except Exception as e:
                            logger.warning(f"[Jimeng] 清理临时文件失败: {e}")
                            
                        return (image_batch, generation_info, image_urls_text)
                        
                    # 每30秒输出一次进度日志
                    if (attempt + 1) % 6 == 0:
                        elapsed_time = (attempt + 1) * check_interval
                        logger.debug(f"[Jimeng] 图片生成中... 已等待 {elapsed_time}秒/{max_wait_time}秒")
                
                return self._create_error_result(f"等待图片生成超时，已等待 {max_wait_time}秒")
            
            # 非排队模式，直接获取URLs
            urls = result.get("urls", [])
            if not urls:
                return self._create_error_result("API未返回图片URL。")
            
            urls_to_download = urls[:num_images]
            # 统一尺寸到当前比例对应的宽高
            width, height = self._get_ratio_dimensions(ratio)
            images = self._download_images(urls_to_download, target_size=(width, height))
            if not images:
                return self._create_error_result("下载图片失败，可能链接已失效。")
            
            image_batch = torch.cat(images, dim=0)
            generation_info = self._generate_info_text(prompt, model, ratio, len(images))
            # 为返回的URL追加history_id，提升可见性与可追踪性
            history_id_local = result.get("history_record_id", "")
            urls_with_history = [(u + f"&history_id={history_id_local}") if "?" in u else (u + f"?history_id={history_id_local}") for u in urls_to_download] if history_id_local else urls_to_download
            image_urls = "\n\n".join(urls_with_history)

            # 清理临时文件
            try:
                for _p in input_image_paths:
                    os.remove(_p)
            except Exception as e:
                logger.warning(f"[Jimeng] 清理临时文件失败: {e}")

            logger.debug(f"[Jimeng] 成功生成 {len(images)} 张图片。")
            return (image_batch, generation_info, image_urls)
            
        except Exception as e:
            logger.exception(f"[Jimeng] 生成图片时发生意外错误")
            return self._create_error_result(f"发生未知错误: {e}")

    def _get_ratio_value(self, ratio: str) -> int:
        """将比例字符串转换为数值
        Args:
            ratio: 比例字符串，如 "4:3"
        Returns:
            int: 比例对应的数值
        """
        ratio_map = {
            "4:3": 4,
            "3:4": 3,
            "1:1": 1,
            "16:9": 16,
            "9:16": 9
        }
        return ratio_map.get(ratio, 1)

    def _get_ratio_dimensions(self, ratio):
        """获取指定比例的图片尺寸
        Args:
            ratio: 图片比例，如 "1:1", "16:9", "9:16" 等
        Returns:
            tuple: (width, height)
        """
        ratios = self.config.get("params", {}).get("ratios", {})
        ratio_config = ratios.get(ratio)
        
        if not ratio_config:
            # 默认使用 1:1
            return (1024, 1024)
            
        return (ratio_config.get("width", 1024), ratio_config.get("height", 1024))

    def _get_model_key(self, model):
        """获取模型的实际key
        Args:
            model: 模型名称或简写
        Returns:
            str: 模型的实际key
        """
        # 处理简写
        model_map = {
            "20": "2.0",
            "21": "2.1",
            "20p": "2.0p",
            "xlpro": "xl",
            "xl": "xl"
        }
        
        # 如果是简写，转换为完整名称
        if model.lower() in model_map:
            model = model_map[model.lower()]
            
        # 获取模型配置
        models = self.config.get("params", {}).get("models", {})
        if model not in models:
            # 如果模型不存在，使用默认模型
            return self.config.get("params", {}).get("default_model", "3.0")
            
        return model

    def _get_upload_token(self):
        """获取上传token"""
        try:
            url = f"{self.base_url}/mweb/v1/get_upload_token"
            params = {
                "aid": self.aid,
                "device_platform": "web",
                "region": "CN"
            }
            
            # 准备POST请求体
            data = {
                "scene": 2
            }
            
            # 发送POST请求而非GET请求
            response = self._send_request("POST", url, params=params, json=data)
            if not response or response.get("ret") != "0":
                logger.error(f"[Jimeng] Failed to get upload token: {response}")
                return None
                
            data = response.get("data", {})
            if not data:
                logger.error("[Jimeng] No data in get_upload_token response")
                return None
                
            return data
        except Exception as e:
            logger.error(f"[Jimeng] Error getting upload token: {e}")
            return None

    def _upload_image(self, image_path, upload_token):
        """上传图片到服务器，使用与视频上传相同的AWS签名方式
        Args:
            image_path: 图片路径
            upload_token: 上传token信息
        Returns:
            str: 上传成功后的图片URI
        """
        try:
            # 获取文件大小
            file_size = os.path.getsize(image_path)
            
            # 第一步：申请图片上传，获取上传地址
            t = datetime.datetime.utcnow()
            amz_date = t.strftime('%Y%m%dT%H%M%SZ')
            
            # 请求参数 - 保持固定顺序
            request_parameters = {
                'Action': 'ApplyImageUpload',
                'FileSize': str(file_size),
                'ServiceId': upload_token.get('space_name', 'tb4s082cfz'),
                'Version': '2018-08-01'
            }
            
            # 构建规范请求字符串
            canonical_querystring = '&'.join([f'{k}={urllib.parse.quote(str(v))}' for k, v in sorted(request_parameters.items())])
            
            # 构建规范请求
            canonical_uri = '/'
            canonical_headers = (
                f'host:imagex.bytedanceapi.com\n'
                f'x-amz-date:{amz_date}\n'
                f'x-amz-security-token:{upload_token.get("session_token", "")}\n'
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
                upload_token.get('access_key_id', ''),
                upload_token.get('secret_access_key', ''),
                'cn-north-1',
                'imagex',
                amz_date,
                upload_token.get('session_token', ''),
                signed_headers,
                canonical_request
            )
            
            # 设置请求头
            headers = {
                'Authorization': authorization,
                'X-Amz-Date': amz_date,
                'X-Amz-Security-Token': upload_token.get('session_token', ''),
                'Host': 'imagex.bytedanceapi.com'
            }
            
            url = f'https://imagex.bytedanceapi.com/?{canonical_querystring}'
            
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                logger.error(f"[Jimeng] Failed to get upload authorization: {response.text}")
                return None
                
            upload_info = response.json()
            if not upload_info or "Result" not in upload_info:
                logger.error(f"[Jimeng] No Result in ApplyImageUpload response: {upload_info}")
                return None
            
            # 第二步：上传图片文件
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
            if response.status_code != 200:
                logger.error(f"[Jimeng] Failed to upload image: {response.text}")
                return None
                
            upload_result = response.json()
            if upload_result.get("code") != 2000:
                logger.error(f"[Jimeng] Upload image error: {upload_result}")
                return None
            
            # 第三步：提交上传，确认图片
            session_key = upload_info['Result']['UploadAddress']['SessionKey']
            store_uri = store_info.get("StoreUri", "")
            
            amz_date = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            
            params = {
                "Action": "CommitImageUpload",
                "Version": "2018-08-01",
                "ServiceId": upload_token.get('space_name', 'tb4s082cfz')
            }
            
            data = {
                "SessionKey": session_key,
                "UploadHosts": upload_info['Result']['UploadAddress']['UploadHosts'],
                "StoreKeys": [store_uri]
            }
            
            payload = json.dumps(data)
            content_sha256 = hashlib.sha256(payload.encode('utf-8')).hexdigest()
            
            # 构建规范请求
            canonical_uri = "/"
            canonical_querystring = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            signed_headers = "x-amz-content-sha256;x-amz-date;x-amz-security-token"
            canonical_headers = f"x-amz-content-sha256:{content_sha256}\nx-amz-date:{amz_date}\nx-amz-security-token:{upload_token.get('session_token', '')}\n"
            
            canonical_request = f"POST\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{content_sha256}"
            
            authorization = self.get_authorization(
                upload_token.get('access_key_id', ''),
                upload_token.get('secret_access_key', ''),
                'cn-north-1',
                'imagex',
                amz_date,
                upload_token.get('session_token', ''),
                signed_headers,
                canonical_request
            )
            
            headers = {
                'accept': '*/*',
                'content-type': 'application/json',
                'authorization': authorization,
                'x-amz-content-sha256': content_sha256,
                'x-amz-date': amz_date,
                'x-amz-security-token': upload_token.get('session_token', ''),
                'origin': 'https://jimeng.jianying.com',
                'referer': 'https://jimeng.jianying.com/'
            }
            
            commit_url = f"https://{upload_token.get('upload_domain', 'imagex.bytedanceapi.com')}"
            response = requests.post(f"{commit_url}?{canonical_querystring}", headers=headers, data=payload)
            if response.status_code != 200:
                logger.error(f"[Jimeng] Failed to commit upload: {response.text}")
                return None
                
            commit_result = response.json()
            if not commit_result or "Result" not in commit_result:
                logger.error(f"[Jimeng] No Result in CommitImageUpload response: {commit_result}")
                return None
                
            # 返回图片URI
            return store_uri
            
        except Exception as e:
            logger.error(f"[Jimeng] Error uploading image: {e}")
            return None

    def _verify_uploaded_image(self, image_uri):
        """验证上传的图片"""
        try:
            url = f"{self.base_url}/mweb/v1/algo_proxy"
            params = {
                "babi_param": json.dumps({
                    "scenario": "image_video_generation",
                    "feature_key": "aigc_to_image",
                    "feature_entrance": "to_image",
                    "feature_entrance_detail": "to_image-algo_proxy"
                }),
                "needCache": "true",
                "cacheErrorCodes[]": "2203",
                "aid": self.aid,
                "device_platform": "web",
                "region": "CN",
                "web_id": self.token_manager.get_web_id(),
                "da_version": "3.1.5"
            }
            
            data = {
                "scene": "image_face_ip",
                "options": {"ip_check": True},
                "req_key": "benchmark_test_user_upload_image_input",
                "file_list": [{"file_uri": image_uri}],
                "req_params": {}
            }
            
            response = self._send_request("POST", url, params=params, json=data)
            return response and response.get("ret") == "0"
            
        except Exception as e:
            logger.error(f"[Jimeng] Error verifying uploaded image: {e}")
            return False

    def _get_image_description(self, image_uri):
        """获取图片描述"""
        try:
            url = f"{self.base_url}/mweb/v1/get_image_description"
            params = {
                "babi_param": json.dumps({
                    "scenario": "image_video_generation",
                    "feature_key": "aigc_to_image",
                    "feature_entrance": "to_image",
                    "feature_entrance_detail": "to_image-get_image_description"
                }),
                "needCache": "false",
                "aid": self.aid,
                "device_platform": "web",
                "region": "CN",
                "web_id": self.token_manager.get_web_id(),
                "da_version": "3.1.5"
            }
            
            data = {
                "file_uri": image_uri
            }
            
            response = self._send_request("POST", url, params=params, json=data)
            if response and response.get("ret") == "0":
                return response.get("data", {}).get("description", "")
            
            return ""
            
        except Exception as e:
            logger.error(f"[Jimeng] Error getting image description: {e}")
            return ""

    def upload_image_and_generate_with_reference(self, image_path=None, image_paths=None, prompt="", model="3.0", ratio="1:1"):
        """上传参考图并生成新图片
        Args:
            image_path: 参考图片路径
            prompt: 提示词
            model: 模型名称
            ratio: 图片比例
        Returns:
            dict: 包含生成的图片URL列表
        """
        try:
            # 获取图片尺寸
            width, height = self._get_ratio_dimensions(ratio)
            
            # 获取上传token
            upload_token = self._get_upload_token()
            if not upload_token:
                logger.error("[Jimeng] Failed to get upload token")
                return None
                
            # 整理待上传路径
            paths = image_paths if image_paths else ([image_path] if image_path else [])
            if not paths:
                logger.error("[Jimeng] 未提供参考图路径")
                return None

            logger.debug(f"[Jimeng] 待上传参考图路径: {paths}")
            # 上传图片（多张）
            image_uris = []
            image_meta = {}
            for p in paths:
                uri = self._upload_image(p, upload_token)
                if not uri:
                    logger.error("[Jimeng] Failed to upload image")
                    return None
                image_uris.append(uri)
                # 可选：逐张验证
                self._verify_uploaded_image(uri)
                # 记录参考图的元数据（宽、高、格式），供网页端展示
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                        fmt = (im.format or "").lower()
                except Exception:
                    w, h, fmt = 0, 0, ""
                image_meta[uri] = {"width": w, "height": h, "format": fmt}

            logger.debug(f"[Jimeng] 图片上传成功, 数量: {len(image_uris)}")
            
            # 获取模型配置
            models = self.config.get("params", {}).get("models", {})
            model_info = models.get(model, {})
            
            # 默认使用4.0模型
            model_req_key = "high_aes_general_v40"
            if model == "4.0":
                model_req_key = model_info.get("model_req_key", "high_aes_general_v40")
            
            # 准备请求参数
            submit_id = str(uuid.uuid4())
            draft_id = "adffa4e0-fced-fc0c-b972-5c5a0f51cb2f"  # 固定draft_id与示例一致
            component_id = "c440938a-d652-fc79-1a48-c516d848094c"  # 固定component_id与示例一致
            # 补充分辨率类型，用于 large_image_info.resolution_type
            resolution_type = self.config.get("params", {}).get("resolution_type", "2k")
            
            # 准备babi_param
            babi_param = {
                "scenario": "image_video_generation",
                "feature_key": "aigc_to_image",
                "feature_entrance": "to_image",
                "feature_entrance_detail": "to_image"
            }
            
            # 给提示词增加前缀，如示例中的"##​"
            formatted_prompt = f"##​{prompt}" if not prompt.startswith("##") else prompt
            
            # 准备draft_content, 严格按照成功请求的格式
            draft_content = {
                "type": "draft",
                "id": draft_id,
                "min_version": "3.0.2",
                "min_features": [],
                "is_from_tsn": True,
                "version": "3.3.2",
                "main_component_id": component_id,
                "component_list": [{
                    "type": "image_base_component",
                    "id": component_id,
                    "min_version": "3.0.2",
                    "gen_type": 12,
                    "generate_type": "blend",
                    "aigc_mode": "workbench",
                    "metadata": {
                        "type": "",
                        "id": str(uuid.uuid4()),
                        "created_platform": 3,
                        "created_platform_version": "",
                        "created_time_in_ms": str(int(time.time() * 1000)),
                        "created_did": ""
                    },
                    "abilities": {
                        "type": "",
                        "id": "df594b0f-9e1f-08ff-c031-54fdb2fff8b3",
                        "blend": {
                            "type": "",
                            "id": "82cebd72-65a1-cb61-e1bc-c4b79003d1d5",
                            "min_features": [],
                            "core_param": {
                                "type": "",
                                "id": "5388a3a6-d1e8-fe78-51e8-8b6e4490f21c",
                                "model": model_req_key,
                                "prompt": formatted_prompt,
                                "sample_strength": 0.5,
                                "image_ratio": self._get_ratio_value(ratio),
                                "large_image_info": {
                                    "type": "",
                                    "id": "364336a8-c4c7-fbaa-c876-4402656ab195",
                                    "height": height,
                                    "width": width,
                                    "resolution_type": resolution_type
                                },
                                "intelligent_ratio": False
                            },
                            "ability_list": [{
                                "type": "",
                                "id": str(uuid.uuid4()),
                                "name": "byte_edit",
                                "image_uri_list": [uri],
                                "image_list": [{
                                    "type": "image",
                                    "id": str(uuid.uuid4()),
                                    "source_from": "upload",
                                    "platform_type": 1,
                                    "name": "",
                                    "image_uri": uri,
                                    "width": image_meta.get(uri, {}).get("width", 0),
                                    "height": image_meta.get(uri, {}).get("height", 0),
                                    "format": image_meta.get(uri, {}).get("format", ""),
                                    "uri": uri
                                }],
                                "strength": 0.5
                            } for uri in image_uris],
                            "history_option": {
                                "type": "",
                                "id": "6ec2e2cd-99af-0033-99a1-a325a27aad88"
                            },
                            "prompt_placeholder_info_list": [{
                                "type": "",
                                "id": str(uuid.uuid4()),
                                "ability_index": idx
                            } for idx in range(len(image_uris))],
                            "postedit_param": {
                                "type": "",
                                "id": "02518f0d-5d8f-c7d3-a8f9-d36aa3600d24",
                                "generate_type": 0
                            }
                        }
                    }
                }]
            }
            
            # 准备请求数据
            url = f"{self.base_url}/mweb/v1/aigc_draft/generate"
            # 对齐网页埋点
            metrics_extra = {
                "promptSource": "custom",
                "generateCount": 1,
                "enterFrom": "reprompt",
                "templateId": "0",
                "generateId": submit_id,
                "isRegenerate": False
            }
            data = {
                "extend": {
                    "root_model": model_req_key,
                    "template_id": ""
                },
                "submit_id": submit_id,
                "metrics_extra": json.dumps(metrics_extra),
                "draft_content": json.dumps(draft_content),
                "http_common_info": {"aid": self.aid}
            }
            
            params = {
                "babi_param": json.dumps(babi_param),
                "aid": self.aid,
                "device_platform": "web",
                "region": "cn",
                "webId": self.token_manager.get_web_id(),
                # 网页端期望的埋点/版本参数
                "da_version": "3.3.2",
                "web_component_open_flag": "1",
                "web_version": "7.5.0",
                "aigc_features": "app_lip_sync"
            }
            # 将 msToken 与 a_bogus 放入 URL 参数（与官网一致）
            token_info = self.token_manager.get_token('/mweb/v1/aigc_draft/generate')
            if token_info:
                if token_info.get("msToken"):
                    params["msToken"] = token_info["msToken"]
                if token_info.get("a_bogus"):
                    params["a_bogus"] = token_info["a_bogus"]
            
            # 发送生成请求
            response = self._send_request("POST", url, params=params, json=data)
            
            if not response or str(response.get("ret")) != "0":
                logger.error(f"[Jimeng] Failed to generate image with reference: {response}")
                return {"error": True, "response": response}
                
            # 获取aigc_data信息
            aigc_data = response.get("data", {}).get("aigc_data", {})
            
            # 获取history_id 和 history_group_key_md5
            history_id = aigc_data.get("history_record_id")
            history_group_key_md5 = aigc_data.get("history_group_key_md5")
            
            if not history_id:
                logger.error("[Jimeng] No history_id in response")
                return None
                
            logger.debug(f"[Jimeng] 请求成功，history_id: {history_id}")
            
            # 从配置文件读取超时参数（参考图生成）
            timeout_config = self.config.get("timeout", {})
            max_wait_time = timeout_config.get("max_wait_time", 300)  # 默认5分钟
            check_interval = timeout_config.get("check_interval", 5)  # 默认5秒间隔
            
            # 立即获取一次状态，检查排队信息
            first_check_result = self._get_generated_images_by_history_id(history_id)
            queue_info = self._get_queue_info_from_response(history_id)
            
            # 如果有排队信息且图片未生成完成，立即返回排队信息
            if queue_info and not first_check_result:
                queue_msg = self._format_queue_message(queue_info)
                # 立即返回排队信息，让用户知道需要等待多久
                return {
                    "is_queued": True,
                    "queue_message": queue_msg,
                    "history_id": history_id,
                    "submit_id": submit_id
                }
            
            if first_check_result:
                logger.debug("[Jimeng] 参考图生成成功，无需等待")
                return {"urls": first_check_result, "history_record_id": history_id, "submit_id": submit_id}
            
            return {"urls": [], "history_record_id": history_id, "submit_id": submit_id}
            
        except Exception as e:
            logger.error(f"[Jimeng] Error generating image with reference: {e}")
            return None

    def _get_generated_images(self, history_id):
        """通过历史ID获取生成的图片(文生图)，增加备用解析逻辑"""
        try:
            url = f"{self.base_url}/mweb/v1/get_history_by_ids"
            
            params = {
                "aid": self.aid,
                "device_platform": "web",
                "region": "cn",
                "webId": self.token_manager.get_web_id(),
                "da_version": "3.3.2",
                "web_component_open_flag": "1",
                "web_version": "7.5.0",
                "aigc_features": "app_lip_sync"
            }
            # 将 msToken 与 a_bogus 放入 URL 参数，保持与提交接口一致
            token_info = self.token_manager.get_token('/mweb/v1/get_history_by_ids')
            if token_info:
                if token_info.get("msToken"):
                    params["msToken"] = token_info["msToken"]
                if token_info.get("a_bogus"):
                    params["a_bogus"] = token_info["a_bogus"]
            
            data = {
                "history_ids": [history_id],
                "image_info": {
                    "width": 2048,
                    "height": 2048,
                    "format": "webp",
                    "image_scene_list": [
                        {"scene": "normal", "width": 2400, "height": 2400, "uniq_key": "2400", "format": "webp"},
                        {"scene": "loss", "width": 1080, "height": 1080, "uniq_key": "1080", "format": "webp"},
                    ]
                }
            }
            
            result = self._send_request("POST", url, params=params, json=data)
            
            if not result or result.get("ret") != "0":
                logger.error(f"[Jimeng] 获取生成状态失败 (T2I): {result}")
                return None
                
            history_data = result.get("data", {}).get(history_id, {})
            if not history_data:
                return None
                
            status = history_data.get("status")
            
            if status == 50:  # 任务已完成
                image_urls = []
                
                # 方案一：优先从 'resources' 字段提取
                resources = history_data.get("resources", [])
                if resources:
                    for resource in resources:
                        if resource.get("type") == "image":
                            image_info = resource.get("image_info", {})
                            image_url = image_info.get("image_url")
                            if image_url:
                                image_urls.append(image_url)
                
                # 方案二：如果 'resources' 中没有，则从 'item_list' 提取（关键补充）
                if not image_urls:
                    item_list = history_data.get("item_list", [])
                    if item_list:
                        logger.debug("[Jimeng] 'resources'为空，尝试从'item_list'提取URL...")
                        for item in item_list:
                            image = item.get("image", {})
                            if image and "large_images" in image:
                                for large_image in image["large_images"]:
                                    if large_image.get("image_url"):
                                        image_urls.append(large_image["image_url"])
                            elif image and image.get("image_url"):
                                image_urls.append(image["image_url"])

                if image_urls:
                    logger.info(f"[Jimeng] 轮询成功，获取到 {len(image_urls)} 个图片URL。")
                    return image_urls
                else:
                    logger.error("[Jimeng] 轮询失败: 在 'resources' 和 'item_list' 中均未找到图片URL。")
                    return None
            
            return None  # 状态不是50，说明还在处理中
                
        except Exception as e:
            logger.error(f"[Jimeng] 检查生成状态时发生意外错误 (T2I): {e}", exc_info=True)
            return None

    def _get_generated_images_by_history_id(self, history_id):
        """通过历史ID获取生成的图片
        Args:
            history_id: 历史ID
        Returns:
            list: 图片URL列表
        """
        try:
            url = f"{self.base_url}/mweb/v1/get_history_by_ids"
            
            params = {
                "aid": self.aid,
                "device_platform": "web",
                "region": "cn",
                "webId": self.token_manager.get_web_id(),
                "da_version": "3.3.2",
                "web_component_open_flag": "1",
                "web_version": "7.5.0",
                "aigc_features": "app_lip_sync"
            }
            token_info = self.token_manager.get_token('/mweb/v1/get_history_by_ids')
            if token_info:
                if token_info.get("msToken"):
                    params["msToken"] = token_info["msToken"]
                if token_info.get("a_bogus"):
                    params["a_bogus"] = token_info["a_bogus"]
            
            data = {
                "history_ids": [history_id],
                "image_info": {
                    "width": 2048,
                    "height": 2048,
                    "format": "webp",
                    "image_scene_list": [
                        {"scene": "normal", "width": 2400, "height": 2400, "uniq_key": "2400", "format": "webp"},
                        {"scene": "loss", "width": 1080, "height": 1080, "uniq_key": "1080", "format": "webp"},
                        {"scene": "loss", "width": 720, "height": 720, "uniq_key": "720", "format": "webp"},
                        {"scene": "loss", "width": 480, "height": 480, "uniq_key": "480", "format": "webp"},
                        {"scene": "loss", "width": 360, "height": 360, "uniq_key": "360", "format": "webp"}
                    ]
                }
            }
            
            result = self._send_request("POST", url, params=params, json=data)
            
            if not result or result.get("ret") != "0":
                logger.error(f"[Jimeng] 获取生成状态失败: {result}")
                return None
                
            # 获取历史记录数据
            history_data = result.get("data", {}).get(history_id, {})
            if not history_data:
                return None
                
            status = history_data.get("status")
            
            # 只有当状态为50(完成)时才处理
            if status == 50:
                resources = history_data.get("resources", [])
                draft_content = history_data.get("draft_content", "")
                
                if not resources:
                    logger.error("[Jimeng] 未找到资源数据")
                    return None
                
                # 解析draft_content以获取所有原始上传图片的URI集合
                upload_image_uris = set()
                try:
                    draft_content_dict = json.loads(draft_content)
                    component_list = draft_content_dict.get("component_list", [])
                    for component in component_list:
                        abilities = component.get("abilities", {})
                        blend_data = abilities.get("blend", {})
                        ability_list = blend_data.get("ability_list", [])
                        for ability in ability_list:
                            if ability.get("name") == "byte_edit":
                                for _uri in ability.get("image_uri_list", []):
                                    upload_image_uris.add(_uri)
                except Exception as e:
                    logger.error(f"[Jimeng] 解析draft_content失败: {e}")
                    
                # 从resources中提取图片URL，排除原始上传图片
                image_urls = []
                for resource in resources:
                    if resource.get("type") == "image":
                        image_info = resource.get("image_info", {})
                        resource_uri = resource.get("key")  # 资源的URI
                        image_url = image_info.get("image_url")
                        
                        # 如果这个资源不是上传的原图，添加到结果中
                        if (not upload_image_uris or resource_uri not in upload_image_uris) and image_url:
                            image_urls.append(image_url)
                
                # 如果从resources中找不到生成的图片，尝试从item_list中获取
                if not image_urls:
                    item_list = history_data.get("item_list", [])
                    for item in item_list:
                        image = item.get("image", {})
                        if image and "large_images" in image:
                            for large_image in image["large_images"]:
                                image_url = large_image.get("image_url")
                                if image_url:
                                    image_urls.append(image_url)
                
                if image_urls:
                    return image_urls
                else:
                    logger.error("[Jimeng] 未找到生成的图片URL")
                    return None
                
            # 其他状态表示正在处理中
            return None
                
        except Exception as e:
            logger.error(f"[Jimeng] 检查生成状态时发生错误: {e}")
            return None

    def _get_generated_images_by_submit_id(self, submit_id: str):
        """通过 submit_id 获取生成的图片（图生图/通用），与官网查询示例对齐
        Args:
            submit_id: 提交ID
        Returns:
            list: 图片URL列表
        """
        try:
            url = f"{self.base_url}/mweb/v1/get_history_by_ids"
            params = {
                "aid": self.aid,
                "device_platform": "web",
                "region": "cn",
                "webId": self.token_manager.get_web_id(),
                "da_version": "3.3.2",
                "web_component_open_flag": "1",
                "web_version": "7.5.0",
                "aigc_features": "app_lip_sync"
            }
            token_info = self.token_manager.get_token('/mweb/v1/get_history_by_ids')
            if token_info:
                if token_info.get("msToken"):
                    params["msToken"] = token_info["msToken"]
                if token_info.get("a_bogus"):
                    params["a_bogus"] = token_info["a_bogus"]

            data = {
                "submit_ids": [submit_id],
                "image_info": {
                    "width": 2048,
                    "height": 2048,
                    "format": "webp",
                    "image_scene_list": [
                        {"scene": "normal", "width": 2400, "height": 2400, "uniq_key": "2400", "format": "webp"},
                        {"scene": "loss", "width": 1080, "height": 1080, "uniq_key": "1080", "format": "webp"},
                        {"scene": "loss", "width": 720, "height": 720, "uniq_key": "720", "format": "webp"},
                        {"scene": "loss", "width": 480, "height": 480, "uniq_key": "480", "format": "webp"},
                        {"scene": "loss", "width": 360, "height": 360, "uniq_key": "360", "format": "webp"}
                    ]
                }
            }

            result = self._send_request("POST", url, params=params, json=data)
            if not result or str(result.get("ret")) != "0":
                logger.error(f"[Jimeng] 按 submit_id 获取生成状态失败: {result}")
                return None

            history_data = result.get("data", {}).get(submit_id, {})
            if not history_data:
                return None

            status = history_data.get("status")
            if status != 50:
                return None

            # 先从 resources 提取图片
            image_urls = []
            resources = history_data.get("resources", [])
            if resources:
                for resource in resources:
                    if resource.get("type") == "image":
                        image_info = resource.get("image_info", {})
                        image_url = image_info.get("image_url")
                        if image_url:
                            image_urls.append(image_url)

            # 如果 resources 没有，再从 item_list 提取
            if not image_urls:
                item_list = history_data.get("item_list", [])
                for item in item_list:
                    image = item.get("image", {})
                    if image and "large_images" in image:
                        for large_image in image["large_images"]:
                            u = large_image.get("image_url")
                            if u:
                                image_urls.append(u)
                    elif image and image.get("image_url"):
                        image_urls.append(image["image_url"])

            return image_urls if image_urls else None

        except Exception as e:
            logger.error(f"[Jimeng] 检查生成状态（按 submit_id）时发生错误: {e}")
            return None

    def _get_queue_info_from_response(self, history_id):
        """从API响应中获取排队信息"""
        try:
            url = f"{self.base_url}/mweb/v1/get_history_by_ids"
            
            params = {
                "aid": self.aid,
                "device_platform": "web",
                "region": "cn",
                "webId": self.token_manager.get_web_id(),
                "da_version": "3.3.2",
                "web_component_open_flag": "1",
                "web_version": "7.5.0",
                "aigc_features": "app_lip_sync"
            }
            token_info = self.token_manager.get_token('/mweb/v1/get_history_by_ids')
            if token_info:
                if token_info.get("msToken"):
                    params["msToken"] = token_info["msToken"]
                if token_info.get("a_bogus"):
                    params["a_bogus"] = token_info["a_bogus"]
            
            data = {
                "history_ids": [history_id],
                "image_info": {
                    "width": 2048,
                    "height": 2048,
                    "format": "webp",
                    "image_scene_list": [
                        {"scene": "normal", "width": 2400, "height": 2400, "uniq_key": "2400", "format": "webp"},
                        {"scene": "normal", "width": 1080, "height": 1080, "uniq_key": "1080", "format": "webp"}
                    ]
                },
                "http_common_info": {"aid": self.aid}
            }
            
            result = self._send_request("POST", url, params=params, json=data)
            
            if result and result.get('ret') == '0':
                history_data = result.get('data', {}).get(history_id, {})
                queue_info = history_data.get('queue_info', {})
                if queue_info:
                    return queue_info
                return None
                
        except Exception as e:
            logger.error(f"[Jimeng] Error getting queue info: {e}")
            return None

    def _format_queue_message(self, queue_info):
        """格式化排队信息为用户友好的消息"""
        try:
            queue_idx = queue_info.get('queue_idx', 0)
            queue_length = queue_info.get('queue_length', 0)
            queue_status = queue_info.get('queue_status', 0)
            
            # 获取真正的等待时间阈值
            priority_queue_display_threshold = queue_info.get('priority_queue_display_threshold', {})
            waiting_time_threshold = priority_queue_display_threshold.get('waiting_time_threshold', 0)
            
            # 将waiting_time_threshold从秒转换为分钟
            wait_minutes = waiting_time_threshold // 60
            wait_seconds = waiting_time_threshold % 60
            
            if wait_minutes > 0:
                time_str = f"{wait_minutes}分{wait_seconds}秒" if wait_seconds > 0 else f"{wait_minutes}分钟"
            else:
                time_str = f"{wait_seconds}秒"
            
            if queue_status == 1:  # 正在排队
                if queue_idx > 0 and queue_length > 0:
                    return f"📊 总队列长度：{queue_length}人\n🔄 您的位置：第{queue_idx}位\n⏰ 预计等待时间：{time_str}\n\n图片正在排队生成中，请耐心等待..."
                else:
                    return f"🔄 图片生成任务已提交，预计等待时间：{time_str}"
            else:
                return "🚀 当前无需排队，正在使用快速生成模式，请等待片刻..."
                
        except Exception as e:
            logger.error(f"[Jimeng] Error formatting queue message: {e}")
            return "🔄 图片生成任务正在排队处理中，请稍候..." 

    def get_authorization(self, access_key, secret_key, region, service, amz_date, security_token, signed_headers, canonical_request):
        """获取AWS V4签名授权头
        Args:
            access_key: 访问密钥ID
            secret_key: 密钥
            region: 地区
            service: 服务名
            amz_date: 日期时间
            security_token: 安全令牌
            signed_headers: 已签名的头部
            canonical_request: 规范请求
        Returns:
            str: 授权头
        """
        try:
            datestamp = amz_date[:8]
            
            # 计算规范请求的哈希值
            canonical_request_hash = hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
            
            # 构建待签名字符串
            credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
            string_to_sign = f"AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{canonical_request_hash}"
            
            # 计算签名密钥
            k_date = hmac.new(f"AWS4{secret_key}".encode('utf-8'), datestamp.encode('utf-8'), hashlib.sha256).digest()
            k_region = hmac.new(k_date, region.encode('utf-8'), hashlib.sha256).digest()
            k_service = hmac.new(k_region, service.encode('utf-8'), hashlib.sha256).digest()
            k_signing = hmac.new(k_service, b'aws4_request', hashlib.sha256).digest()
            
            # 计算签名
            signature = hmac.new(k_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
            
            # 构建授权头
            authorization = (
                f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
                f"SignedHeaders={signed_headers}, Signature={signature}"
            )
            
            return authorization
        except Exception as e:
            logger.error(f"[Jimeng] Error generating authorization: {str(e)}")
            return ""

    def _create_error_result(self, error_msg: str) -> Tuple[torch.Tensor, str, str]:
        """创建错误结果
        Args:
            error_msg: 错误信息
        Returns:
            Tuple[torch.Tensor, str, str]: (错误图像, 错误信息, 空URL列表)
        """
        logger.error(f"[Jimeng] {error_msg}")
        error_image = torch.ones(1, 256, 256, 3) * torch.tensor([1.0, 0.0, 0.0])
        return (error_image, f"错误: {error_msg}", "")

    def _download_images(self, urls: List[str], target_size: Optional[Tuple[int, int]] = None) -> List[torch.Tensor]:
        """下载图片并转换为统一尺寸的张量
        Args:
            urls: 图片URL列表
            target_size: 期望的 (width, height)，用于统一所有图片尺寸；为 None 时不缩放
        Returns:
            List[torch.Tensor]: 图片张量列表 (形状: [1, H, W, 3])
        """
        images = []
        for url in urls:
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                img_data = response.content

                pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                w, h = pil_image.size
                if w < 2 or h < 2:
                    logger.warning(f"[Jimeng] 跳过异常尺寸图片 {url}: {w}x{h}")
                    continue

                # 若提供统一目标尺寸，则按配置的比例尺寸统一缩放
                if target_size and all(isinstance(v, int) and v > 0 for v in target_size):
                    tw, th = target_size
                    try:
                        pil_image = pil_image.resize((tw, th), resample=Image.LANCZOS)
                    except Exception as re:
                        logger.warning(f"[Jimeng] 图片缩放失败，保留原尺寸 {w}x{h}: {re}")

                np_image = np.array(pil_image, dtype=np.float32) / 255.0
                # 期望形状为 [1, H, W, 3]
                tensor_image = torch.from_numpy(np_image).unsqueeze(0)
                images.append(tensor_image)
            except Exception as e:
                logger.error(f"[Jimeng] 下载或处理图片失败 {url}: {e}")
                continue
        return images

    def _save_input_image(self, image_tensor: torch.Tensor) -> Optional[str]:
        """将输入的图像张量保存为临时文件
        Args:
            image_tensor: 输入图像张量
        Returns:
            str: 临时文件路径，如果保存失败则返回None
        """
        try:
            # 确保临时目录存在
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 生成临时文件路径
            unique_name = f"temp_input_{time.time_ns()}_{uuid.uuid4().hex}.png"
            temp_path = os.path.join(temp_dir, unique_name)
            
            # 将张量转换为PIL图像并保存
            if len(image_tensor.shape) == 4:  # batch, height, width, channels
                image_tensor = image_tensor[0]  # 取第一张图片
            
            # 确保值在0-1范围内
            image_tensor = torch.clamp(image_tensor, 0, 1)
            
            # 转换为PIL图像
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            
            # 保存图像
            image_pil.save(temp_path)
            logger.info(f"[Jimeng] 输入图像已保存到: {temp_path}")
            
            return temp_path
        except Exception as e:
            logger.error(f"[Jimeng] 保存输入图像失败: {e}")
            return None

    def _generate_info_text(self, prompt: str, model: str, ratio: str, num_images: int) -> str:
        """生成图片信息文本
        Args:
            prompt: 提示词
            model: 模型名称
            ratio: 图片比例
            num_images: 图片数量
        Returns:
            str: 信息文本
        """
        models_config = self.config.get("params", {}).get("models", {})
        model_display_name = models_config.get(model, {}).get("name", model)
        
        info_lines = [f"提示词: {prompt}", f"模型: {model_display_name}", f"比例: {ratio}", f"数量: {num_images}"]
        return "\n".join(info_lines)

    def _is_configured(self) -> bool:
        """检查配置是否包含至少一个有效的sessionid。"""
        accounts = self.config.get("accounts", [])
        if not isinstance(accounts, list) or not accounts:
            return False
        return any(acc.get("sessionid") for acc in accounts)
