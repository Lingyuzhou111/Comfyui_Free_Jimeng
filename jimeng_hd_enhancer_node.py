"""
即梦AI图片高清化节点
ComfyUI插件的高清化功能节点，可以对接上游即梦AI生图节点
"""

import os
import json
import logging
import torch
import numpy as np
import time
import requests
import io
import uuid
import base64
import hashlib
import random
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List

# 导入核心模块
from .core.token_manager import TokenManager
from .core.api_client import ApiClient

logger = logging.getLogger(__name__)

def _load_config_for_class() -> Dict[str, Any]:
    """
    辅助函数：用于在节点类实例化前加载配置，
    以便为UI输入选项提供动态数据。
    """
    try:
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(plugin_dir, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[JimengHDNode] 无法为UI加载配置文件: {e}。将使用默认值。")
        return {"params": {}, "timeout": {"max_wait_time": 300, "check_interval": 5}}

class JimengHDEnhancerNode:
    """
    即梦AI图片高清化节点
    接收上游即梦AI生图节点的image_urls输出，对指定序号的图片进行高清化处理
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
                    logger.info("[JimengHDNode] 从模板创建了 config.json")
                else:
                    logger.error("[JimengHDNode] 配置文件和模板文件都不存在！")
                    return {}
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("[JimengHDNode] 配置文件加载成功")
            return config
        except Exception as e:
            logger.error(f"[JimengHDNode] 配置文件加载失败: {e}")
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
            logger.error("[JimengHDNode] 因配置为空，核心组件初始化失败。")
            return
        try:
            self.token_manager = TokenManager(self.config)
            self.api_client = ApiClient(self.token_manager, self.config)
            logger.info("[JimengHDNode] 核心组件初始化成功。")
        except Exception as e:
            logger.error(f"[JimengHDNode] 核心组件初始化失败: {e}", exc_info=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "history_id": ("STRING", {"default": "", "tooltip": "从上游即梦AI生图节点连接history_id输出"}),
                "image_index": (["1", "2", "3", "4"], {"default": "1", "tooltip": "选择要高清化的图片序号"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_image", "enhancement_info", "enhanced_image_url")
    FUNCTION = "enhance_image"
    CATEGORY = "即梦AI"
    

    
    def _get_item_id_for_hd_generation(self, history_id: str, index: int) -> Optional[str]:
        """
        获取指定序号图片的item_id，用于超清生成。
        
        Args:
            history_id: 历史记录ID
            index: 图片序号(1-4)
            
        Returns:
            str: item_id，如果获取失败则返回None
        """
        try:
            logger.info(f"[JimengHDNode] 开始获取item_id，history_id: {history_id}, index: {index}")
            
            # 获取当前账号信息
            account = self.token_manager.get_current_account()
            if not account:
                logger.error("[JimengHDNode] 无法获取当前账号信息")
                return None
            
            # 构建请求参数
            current_timestamp = int(time.time())
            web_id = account.get('web_id', str(random.random() * 999999999999999999 + 7000000000000000000))
            
            # 生成msToken
            msToken_base = f"{account['sessionid']}_{current_timestamp}"
            msToken = base64.b64encode(msToken_base.encode()).decode()
            
            # 生成a_bogus
            a_bogus_base = f"{account['sessionid']}_{current_timestamp}_item"
            a_bogus = base64.b64encode(a_bogus_base.encode()).decode()
            
            # 构建URL参数
            url_params = {
                "aid": "513695",
                "device_platform": "web", 
                "region": "CN",
                "web_id": web_id,
                "msToken": msToken,
                "a_bogus": a_bogus
            }
            
            # 构建请求数据，包含必要的image_info用于超清版本请求
            request_data = {
                "history_ids": [history_id],
                "image_info": {
                    "width": 2048,
                    "height": 2048,
                    "format": "webp",
                    "image_scene_list": [
                        {"scene": "normal", "width": 2400, "height": 2400, "uniq_key": "2400", "format": "webp"}
                    ]
                }
            }
            
            # 使用当前账号的sessionid构建cookie
            cookie = f"sessionid={account['sessionid']}; sessionid_ss={account['sessionid']}; _tea_web_id={web_id}; web_id={web_id}; _v2_spipe_web_id={web_id}"
            
            # 生成sign
            sign_base = f"{account['sessionid']}_{current_timestamp}_item"
            sign = hashlib.md5(sign_base.encode()).hexdigest()
            
            # 构建请求头
            headers = {
                'accept': 'application/json, text/plain, */*',
                'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
                'appvr': '5.8.0',
                'content-type': 'application/json',
                'device-time': str(current_timestamp),
                'origin': 'https://jimeng.jianying.com',
                'pf': '7',
                'priority': 'u=1, i',
                'referer': 'https://jimeng.jianying.com/ai-tool/image/generate',
                'sec-ch-ua': '"Microsoft Edge";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'sign': sign,
                'sign-ver': '1',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0',
                'Cookie': cookie
            }
            
            logger.info(f"[JimengHDNode] 发送get_history_by_ids请求，URL参数: {url_params}")
            
            # 发送请求
            response = requests.post(
                "https://jimeng.jianying.com/mweb/v1/get_history_by_ids",
                params=url_params,
                headers=headers,
                json=request_data,
                timeout=10
            )
            
            logger.info(f"[JimengHDNode] get_history_by_ids响应状态: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"[JimengHDNode] Get item_id request failed: HTTP {response.status_code} - {response.text}")
                return None
                
            result = response.json()
            
            if result.get("ret") != "0":
                logger.error(f"[JimengHDNode] Get item_id API error: {result}")
                return None
                
            # 解析响应获取item_id
            data = result.get("data", {})
            history_data = data.get(str(history_id))  # 确保使用字符串形式的history_id作为key
            
            if not history_data:
                # 尝试使用整数形式的history_id
                history_data = data.get(int(history_id))
                
            if not history_data:
                logger.error(f"[JimengHDNode] 未找到history_data，可用的keys: {list(data.keys())}")
                return None
                
            logger.info(f"[JimengHDNode] 找到history_data，包含的keys: {list(history_data.keys())}")
            
            # 检查生成类型
            generate_type = history_data.get("generate_type")
            logger.info(f"[JimengHDNode] generate_type: {generate_type}")
            
            if generate_type == 13:
                # 这是超清生成记录，需要获取原始图片的history_id
                origin_history_id = history_data.get("origin_history_record_id")
                if origin_history_id:
                    logger.info(f"[JimengHDNode] 检测到超清记录，重新获取原始记录: {origin_history_id}")
                    return self._get_item_id_for_hd_generation(origin_history_id, index)
            
            # 获取item_list
            item_list = history_data.get("item_list", [])
            logger.info(f"[JimengHDNode] item_list长度: {len(item_list)}")
            
            if not item_list:
                logger.error("[JimengHDNode] item_list为空")
                return None
            
            # 根据generate_type使用不同的逻辑
            if generate_type == 1:
                # 普通文字生成图片：每个item对应一张生成的图片
                # 直接根据index选择对应的item
                if len(item_list) < index:
                    logger.error(f"[JimengHDNode] 普通文字生成的item_list不足，长度: {len(item_list)}, 需要index: {index}")
                    return None
                
                # 获取指定序号的item（index从1开始，所以要减1）
                target_item = item_list[index - 1]
                logger.info(f"[JimengHDNode] 选择第{index}个item，keys: {list(target_item.keys()) if target_item else 'None'}")
                
                if "common_attr" in target_item:
                    item_id = target_item.get("common_attr", {}).get("id")
                else:
                    item_id = target_item.get("id")
                    
            elif generate_type == 12:
                # 参考图生成：每个item对应一张生成的图片
                # 可以直接根据index选择对应的item
                if len(item_list) < index:
                    logger.error(f"[JimengHDNode] 参考图生成的item_list不足，长度: {len(item_list)}, 需要index: {index}")
                    return None
                
                # 获取指定序号的item（index从1开始，所以要减1）
                target_item = item_list[index - 1]
                logger.info(f"[JimengHDNode] 选择第{index}个item，keys: {list(target_item.keys()) if target_item else 'None'}")
                
                if "common_attr" in target_item:
                    item_id = target_item.get("common_attr", {}).get("id")
                else:
                    item_id = target_item.get("id")
                    
            else:
                # 其他类型的生成，尝试使用第一个item
                logger.warning(f"[JimengHDNode] 未知的generate_type: {generate_type}，尝试使用第一个item")
                first_item = item_list[0]
                if "common_attr" in first_item:
                    item_id = first_item.get("common_attr", {}).get("id")
                else:
                    item_id = first_item.get("id")
                
            logger.info(f"[JimengHDNode] 找到item_id: {item_id}，将用于生成第{index}张图片的超清版本")
            
            return item_id
            
        except Exception as e:
            logger.error(f"[JimengHDNode] Error getting item_id: {str(e)}")
            return None
    
    def _generate_hd_image(self, item_id: str, origin_history_id: str) -> Tuple[bool, str]:
        """
        调用即梦API生成超清图片。
        
        Args:
            item_id: 图片项目ID
            origin_history_id: 原始历史记录ID
            
        Returns:
            Tuple[bool, str]: (是否成功, 结果URL或错误信息)
        """
        try:
            # 检查必要参数
            if not item_id:
                return False, "item_id为空，无法生成超清图片"
            if not origin_history_id:
                return False, "origin_history_id为空，无法生成超清图片"
                
            logger.info(f"[JimengHDNode] 开始生成超清图片，item_id: {item_id}, origin_history_id: {origin_history_id}")
            
            # 获取当前账号信息
            account = self.token_manager.get_current_account()
            if not account:
                return False, "获取当前账号信息失败"
            
            # 构建请求参数
            current_timestamp = int(time.time())
            web_id = account.get('web_id', str(random.random() * 999999999999999999 + 7000000000000000000))
            
            # 生成唯一的submit_id
            submit_id = str(uuid.uuid4())
            
            # 生成msToken
            msToken_base = f"{account['sessionid']}_{current_timestamp}"
            msToken = base64.b64encode(msToken_base.encode()).decode()
            
            # 生成a_bogus
            a_bogus_base = f"{account['sessionid']}_{current_timestamp}_gen_hd"
            a_bogus = base64.b64encode(a_bogus_base.encode()).decode()
            
            # 生成sign
            sign_base = f"{account['sessionid']}_{current_timestamp}_gen_hd"
            sign = hashlib.md5(sign_base.encode()).hexdigest()
            
            # 构建babi_param
            babi_param = {
                "scenario": "image_video_generation",
                "feature_key": "to_image_enhancement",
                "feature_entrance": "to_image",
                "feature_entrance_detail": "to_image-enhancement"
            }
            
            # 构建URL参数
            url_params = {
                "babi_param": json.dumps(babi_param),
                "aid": "513695",
                "device_platform": "web",
                "region": "CN",
                "web_id": web_id,
                "msToken": msToken,
                "a_bogus": a_bogus
            }
            
            # 构建复杂的draft_content
            # 创建主组件ID
            main_component_id = str(uuid.uuid4())
            parent_component_id = str(uuid.uuid4())
            
            draft_content = {
                "type": "draft",
                "id": str(uuid.uuid4()),
                "min_version": "3.0.2",
                "min_features": [],
                "is_from_tsn": True,
                "version": "3.2.5",
                "main_component_id": main_component_id,
                "component_list": [
                    # 父组件 - 原始生成的组件
                    {
                        "type": "image_base_component",
                        "id": parent_component_id,
                        "min_version": "3.0.2",
                        "gen_type": 1,
                        "generate_type": "generate",
                        "aigc_mode": "workbench",
                        "abilities": {
                            "type": "",
                            "id": str(uuid.uuid4()),
                            "generate": {
                                "type": "",
                                "id": str(uuid.uuid4()),
                                "core_param": {
                                    "type": "",
                                    "id": str(uuid.uuid4()),
                                    "model": "high_aes_general_v30l:general_v3.0_18b",
                                    "prompt": "原始提示词",  # 这里应该用实际的提示词，但目前先用占位符
                                    "negative_prompt": "",
                                    "seed": 768018461,
                                    "sample_strength": 0.5,
                                    "image_ratio": 3,
                                    "large_image_info": {
                                        "type": "",
                                        "id": str(uuid.uuid4()),
                                        "height": 1024,
                                        "width": 576
                                    }
                                },
                                "history_option": {
                                    "type": "",
                                    "id": str(uuid.uuid4())
                                }
                            }
                        }
                    },
                    # 子组件 - 超清生成组件
                    {
                        "type": "image_base_component",
                        "id": main_component_id,
                        "min_version": "3.0.2",
                        "parent_id": parent_component_id,
                        "metadata": {
                            "type": "",
                            "id": str(uuid.uuid4()),
                            "created_platform": 3,
                            "created_platform_version": "",
                            "created_time_in_ms": str(current_timestamp * 1000),
                            "created_did": ""
                        },
                        "generate_type": "normal_hd",
                        "aigc_mode": "workbench",
                        "abilities": {
                            "type": "",
                            "id": str(uuid.uuid4()),
                            "normal_hd": {
                                "type": "",
                                "id": str(uuid.uuid4()),
                                "postedit_param": {
                                    "type": "",
                                    "id": str(uuid.uuid4()),
                                    "generate_type": 13,
                                    "item_id": int(item_id),
                                    "origin_history_id": int(origin_history_id),
                                    "history_option": {
                                        "type": "",
                                        "id": str(uuid.uuid4())
                                    }
                                }
                            }
                        }
                    }
                ]
            }
            
            # 构建请求数据
            request_data = {
                "extend": {
                    "root_model": "high_aes_general_v30l:general_v3.0_18b",
                    "template_id": ""
                },
                "submit_id": submit_id,
                "draft_content": json.dumps(draft_content)
            }
            
            # 使用当前账号的sessionid构建cookie
            cookie = f"sessionid={account['sessionid']}; sessionid_ss={account['sessionid']}; _tea_web_id={web_id}; web_id={web_id}; _v2_spipe_web_id={web_id}"
            
            # 构建请求头
            headers = {
                'accept': 'application/json, text/plain, */*',
                'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
                'appvr': '5.8.0',
                'content-type': 'application/json',
                'device-time': str(current_timestamp),
                'origin': 'https://jimeng.jianying.com',
                'pf': '7',
                'priority': 'u=1, i',
                'referer': 'https://jimeng.jianying.com/ai-tool/image/generate',
                'sec-ch-ua': '"Microsoft Edge";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'sign': sign,
                'sign-ver': '1',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0',
                'Cookie': cookie
            }
            
            logger.info(f"[JimengHDNode] 发送超清生成请求，item_id: {item_id}, origin_history_id: {origin_history_id}")
            
            # 发送生成请求
            response = requests.post(
                "https://jimeng.jianying.com/mweb/v1/aigc_draft/generate",
                params=url_params,
                headers=headers,
                json=request_data,
                timeout=10
            )
            
            logger.info(f"[JimengHDNode] HD generation request status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"[JimengHDNode] HD generation request failed: HTTP {response.status_code} - {response.text}")
                return False, f"超清生成请求失败，状态码：{response.status_code}"
                
            result = response.json()
            if result.get("ret") != "0":
                logger.error(f"[JimengHDNode] HD generation API error: {result}")
                return False, f"超清生成API错误：{result.get('errmsg', '未知错误')}"
                
            # 获取生成任务的history_record_id
            aigc_data = result.get("data", {}).get("aigc_data", {})
            hd_history_id = aigc_data.get("history_record_id")
            
            if not hd_history_id:
                logger.error(f"[JimengHDNode] No HD history_record_id in response: {result}")
                return False, "未获取到超清生成任务ID"
                
            logger.info(f"[JimengHDNode] 超清生成任务已提交，history_record_id: {hd_history_id}")
            
            # 等待生成完成并获取结果
            success, result = self._wait_for_hd_generation(hd_history_id)
            return success, result
            
        except Exception as e:
            logger.error(f"[JimengHDNode] Error generating HD image: {str(e)}")
            return False, f"生成超清图片失败: {str(e)}"
    
    def _wait_for_hd_generation(self, hd_history_id: str, max_retries: Optional[int] = None) -> Tuple[bool, str]:
        """
        等待超清图片生成完成。
        
        Args:
            hd_history_id: 超清生成任务的历史记录ID
            max_retries: 最大重试次数，如果为None则使用配置文件中的设置
            
        Returns:
            Tuple[bool, str]: (是否成功, 结果URL或错误信息)
        """
        try:
            # 从配置文件读取超时参数
            if max_retries is None:
                timeout_config = self.config.get("timeout", {})
                max_wait_time = timeout_config.get("max_wait_time", 300)  # 默认5分钟
                check_interval = timeout_config.get("check_interval", 5)  # 改为5秒间隔，与859bot版一致
                max_retries = max_wait_time // check_interval
            else:
                check_interval = 5  # 改为5秒间隔
                
            logger.info(f"[JimengHDNode] 开始轮询超清图片生成状态，最大等待时间: {max_retries * check_interval}秒")
            
            for i in range(max_retries):
                # 检查生成状态
                success, result = self._check_hd_generation_status(hd_history_id)
                if success and result:
                    # 生成完成，返回超清图片URL
                    elapsed_time = (i + 1) * check_interval
                    logger.info(f"[JimengHDNode] 超清图片生成完成，总耗时: {elapsed_time}秒")
                    
                    # 如果是WebP格式，下载并转换为PNG
                    if result.endswith('.webp') or '.webp' in result:
                        converted_path = self._convert_webp_to_png(result)
                        return True, converted_path
                    
                    return True, result
                elif success is False:
                    # 生成失败
                    return False, result
                
                # 继续等待，每30秒显示一次进度
                if (i + 1) % 6 == 0:  # 每6次循环（30秒）显示一次进度
                    elapsed_time = (i + 1) * check_interval
                    total_time = max_retries * check_interval
                    logger.info(f"[JimengHDNode] 超清图片生成中... 已等待 {elapsed_time}秒/{total_time}秒")
                time.sleep(check_interval)
                
            logger.error(f"[JimengHDNode] 超清图片生成超时，已等待 {max_retries * check_interval}秒")
            return False, f"超清图片生成超时，已等待 {max_retries * check_interval}秒"
            
        except Exception as e:
            logger.error(f"[JimengHDNode] Error waiting for HD generation: {str(e)}")
            return False, f"等待超清生成失败: {str(e)}"
    
    def _check_hd_generation_status(self, hd_history_id: str) -> Tuple[Optional[bool], Optional[str]]:
        """
        检查超清图片生成状态。
        
        Args:
            hd_history_id: 超清生成任务的历史记录ID
            
        Returns:
            Tuple[Optional[bool], Optional[str]]: (是否成功, 结果URL或错误信息)
            None表示继续等待
        """
        try:
            # 获取当前账号信息
            account = self.token_manager.get_current_account()
            if not account:
                return None, "获取当前账号信息失败"
            
            # 构建请求参数
            current_timestamp = int(time.time())
            web_id = account.get('web_id', str(random.random() * 999999999999999999 + 7000000000000000000))
            
            # 生成msToken
            msToken_base = f"{account['sessionid']}_{current_timestamp}"
            msToken = base64.b64encode(msToken_base.encode()).decode()
            
            # 生成a_bogus
            a_bogus_base = f"{account['sessionid']}_{current_timestamp}_check"
            a_bogus = base64.b64encode(a_bogus_base.encode()).decode()
            
            # 构建URL参数
            url_params = {
                "aid": "513695",
                "device_platform": "web", 
                "region": "CN",
                "web_id": web_id,
                "msToken": msToken,
                "a_bogus": a_bogus
            }
            
            # 构建请求数据
            request_data = {
                "history_ids": [hd_history_id],
                "image_info": {
                    "width": 2048,
                    "height": 2048,
                    "format": "webp",
                    "image_scene_list": [
                        {"scene": "normal", "width": 2400, "height": 2400, "uniq_key": "2400", "format": "webp"}
                    ]
                }
            }
            
            # 使用当前账号的sessionid构建cookie
            cookie = f"sessionid={account['sessionid']}; sessionid_ss={account['sessionid']}; _tea_web_id={web_id}; web_id={web_id}; _v2_spipe_web_id={web_id}"
            
            # 生成sign
            sign_base = f"{account['sessionid']}_{current_timestamp}_check"
            sign = hashlib.md5(sign_base.encode()).hexdigest()
            
            # 构建请求头
            headers = {
                'accept': 'application/json, text/plain, */*',
                'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
                'appvr': '5.8.0',
                'content-type': 'application/json',
                'device-time': str(current_timestamp),
                'origin': 'https://jimeng.jianying.com',
                'pf': '7',
                'priority': 'u=1, i',
                'referer': 'https://jimeng.jianying.com/ai-tool/image/generate',
                'sec-ch-ua': '"Microsoft Edge";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'sign': sign,
                'sign-ver': '1',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0',
                'Cookie': cookie
            }
            
            # 发送请求
            response = requests.post(
                "https://jimeng.jianying.com/mweb/v1/get_history_by_ids",
                params=url_params,
                headers=headers,
                json=request_data,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.debug(f"[JimengHDNode] HD status check failed: HTTP {response.status_code}")
                return None, None  # 继续等待
                
            result = response.json()
            if result.get("ret") != "0":
                logger.debug(f"[JimengHDNode] HD status check API error: {result}")
                return None, None  # 继续等待
                
            # 解析响应
            data = result.get("data", {})
            history_data = data.get(hd_history_id)
            if not history_data:
                return None, None  # 继续等待

            # 检查任务状态 - 按照859bot版的逻辑
            task = history_data.get("task", {})
            status = task.get("status", 0)

            if status == 50:  # 生成成功
                # 获取超清图片URL
                item_list = history_data.get("item_list", [])
                if item_list:
                    item = item_list[0]
                    # 优先从large_images获取
                    image_info = item.get("image", {})
                    large_images = image_info.get("large_images", [])
                    if large_images:
                        large_image = large_images[0]
                        hd_url = large_image.get("image_url")
                        if hd_url:
                            return True, hd_url
                    # 回退到cover_url_map
                    cover_url_map = item.get("common_attr", {}).get("cover_url_map", {})
                    hd_url = cover_url_map.get("2400")
                    if hd_url:
                        return True, hd_url
                return False, "未找到超清图片URL"
            elif status == 60:  # 生成失败
                fail_msg = task.get("task_payload", {}).get("fail_msg", "生成失败")
                return False, f"超清图片生成失败: {fail_msg}"
            # 其他状态，继续等待
            return None, None
            
        except Exception as e:
            logger.debug(f"[JimengHDNode] Error checking HD status: {str(e)}")
            return None, None  # 继续等待
    
    def _convert_webp_to_png(self, webp_url: str) -> str:
        """
        将WebP图片转换为PNG格式。
        
        Args:
            webp_url: WebP图片的URL
            
        Returns:
            str: 转换后的PNG文件路径或原始URL
        """
        try:
            logger.info(f"[JimengHDNode] 下载并转换WebP图片: {webp_url}")
            
            # 下载WebP图片
            download_response = requests.get(webp_url, timeout=30)
            if download_response.status_code != 200:
                logger.error(f"[JimengHDNode] 下载WebP图片失败: HTTP {download_response.status_code}")
                return webp_url  # 返回原始URL作为后备
            
            # 使用PIL转换WebP到PNG
            webp_image = Image.open(io.BytesIO(download_response.content))
            
            # PNG支持透明通道，保持原始模式
            if webp_image.mode not in ('RGBA', 'RGB', 'LA', 'L'):
                webp_image = webp_image.convert('RGBA')
            
            # 保存为PNG到临时文件
            temp_dir = os.path.join(self.plugin_dir, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            temp_filename = f"hd_image_{int(time.time())}.png"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            webp_image.save(temp_path, 'PNG', optimize=True)
            logger.info(f"[JimengHDNode] WebP已转换为PNG: {temp_path}")
            
            return temp_path
            
        except Exception as e:
            logger.error(f"[JimengHDNode] WebP转换失败: {str(e)}")
            return webp_url  # 返回原始URL作为后备
    
    def _download_image(self, url: str) -> Optional[torch.Tensor]:
        """
        下载图片并转换为torch张量。
        
        Args:
            url: 图片URL
            
        Returns:
            torch.Tensor: 图片张量，如果下载失败则返回None
        """
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            img_data = response.content
            pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
            np_image = np.array(pil_image, dtype=np.float32) / 255.0
            tensor_image = torch.from_numpy(np_image).unsqueeze(0)
            return tensor_image
        except Exception as e:
            logger.error(f"[JimengHDNode] 下载或处理图片失败 {url}: {e}")
            return None

    def enhance_image(self, history_id: str, image_index: int) -> Tuple[torch.Tensor, str, str]:
        """
        主执行函数：对指定序号的图片进行高清化处理。
        
        Args:
            history_id: 历史记录ID字符串
            image_index: 要高清化的图片序号(1-4)
            
        Returns:
            Tuple[torch.Tensor, str, str]: (高清化后的图片, 处理信息, 高清化后的图片URL)
        """
        try:
            # --- 1. 通用检查 ---
            if not self.token_manager or not self.api_client:
                return self._create_error_result("插件未正确初始化，请检查后台日志。")
            if not self._is_configured():
                return self._create_error_result("插件未配置，请在 config.json 中至少填入一个账号的 sessionid。")
            if not history_id or not history_id.strip():
                return self._create_error_result("历史记录ID不能为空。")
            # 将字符串转换为整数
            try:
                image_index_int = int(image_index)
            except ValueError:
                return self._create_error_result("图片序号格式错误。")
                
            if image_index_int < 1 or image_index_int > 4:
                return self._create_error_result("图片序号应在1-4之间。")

            # --- 2. 获取历史记录ID ---
            target_history_id = history_id.strip()
            logger.info(f"[JimengHDNode] 使用history_id: {target_history_id}")
            logger.info(f"[JimengHDNode] 选择第{image_index_int}张图片进行高清化")
            
            # --- 4. 检查积分 ---
            if not self.token_manager.find_account_with_sufficient_credit(2):
                return self._create_error_result("所有账号积分均不足2点，无法进行高清化处理。")

            # --- 5. 获取item_id ---
            item_id = self._get_item_id_for_hd_generation(target_history_id, image_index_int)
            if not item_id:
                return self._create_error_result("无法获取图片项目ID，无法进行高清化处理。")

            # --- 6. 生成高清化图片 ---
            logger.info("=" * 50)
            logger.info(f"[JimengHDNode] 开始高清化处理第{image_index_int}张图片")
            logger.info(f"[JimengHDNode] 原始历史记录ID: {target_history_id}")
            logger.info(f"[JimengHDNode] item_id: {item_id}")
            logger.info("-" * 50)

            success, result = self._generate_hd_image(item_id, target_history_id)
            if not success:
                return self._create_error_result(f"高清化处理失败: {result}")

            # --- 7. 下载高清化后的图片 ---
            enhanced_image_tensor = self._download_image(result)
            if enhanced_image_tensor is None:
                return self._create_error_result("下载高清化后的图片失败。")

            # --- 8. 准备输出信息 ---
            credit_info = self.token_manager.get_credit()
            credit_text = f"\n当前账号剩余积分: {credit_info.get('total_credit', '未知')}" if credit_info else ""
            enhancement_info = f"高清化处理完成\n原始图片序号: {image_index_int}\n原始历史记录ID: {target_history_id}\n高清化图片URL: {result}" + credit_text

            logger.info(f"[JimengHDNode] ✅ 高清化处理完成，成功生成高清图片。")
            logger.info(f"[JimengHDNode] {credit_text.strip()}")
            logger.info(f"[JimengHDNode] 高清化图片URL: {result}")
            logger.info("=" * 50)

            return (enhanced_image_tensor, enhancement_info, result)

        except Exception as e:
            logger.exception(f"[JimengHDNode] 节点执行时发生意外错误")
            return self._create_error_result(f"节点执行异常: {e}")

    def _create_error_result(self, error_msg: str) -> Tuple[torch.Tensor, str, str]:
        """
        创建错误结果。
        
        Args:
            error_msg: 错误信息
            
        Returns:
            Tuple[torch.Tensor, str, str]: 错误结果
        """
        logger.error(f"[JimengHDNode] {error_msg}")
        error_image = torch.ones(1, 256, 256, 3) * torch.tensor([1.0, 0.0, 0.0])
        return (error_image, f"错误: {error_msg}", "")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "Jimeng_HD_Enhancer": JimengHDEnhancerNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Jimeng_HD_Enhancer": "即梦AI图片高清化"
} 