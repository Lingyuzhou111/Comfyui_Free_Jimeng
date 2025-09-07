"""
即梦AI文/图生图合并节点
ComfyUI插件的文生图和图生图功能合并为一个节点
"""

import os
import json
import logging
import torch
import numpy as np
import time
import requests
import io
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List

# 导入核心模块
from .core.token_manager import TokenManager
from .core.api_client import ApiClient

logger = logging.getLogger(__name__)

def _load_config_for_class() -> Dict[str, Any]:
    """
    辅助函数：用于在节点类实例化前加载配置，
    以便为UI输入选项提供动态数据（如模型列表、账号列表）。
    """
    try:
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(plugin_dir, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[JimengNode] 无法为UI加载配置文件: {e}。将使用默认值。")
        return {"params": {"models": {}, "ratios": {}, "default_model": "", "default_ratio": ""}, "accounts": []}

class JimengImageNode:
    """
    即梦AI文/图生图合并节点
    通过是否提供 ref_image_1…ref_image_6 中任意一张参考图自动判断是文生图还是图生图
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
                    logger.info("[JimengNode] 从模板创建了 config.json")
                else:
                    logger.error("[JimengNode] 配置文件和模板文件都不存在！")
                    return {}
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("[JimengNode] 配置文件加载成功")
            return config
        except Exception as e:
            logger.error(f"[JimengNode] 配置文件加载失败: {e}")
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
            logger.error("[JimengNode] 因配置为空，核心组件初始化失败。")
            return
        try:
            self.token_manager = TokenManager(self.config)
            self.api_client = ApiClient(self.token_manager, self.config)
            logger.info("[JimengNode] 核心组件初始化成功。")
        except Exception as e:
            logger.error(f"[JimengNode] 核心组件初始化失败: {e}", exc_info=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        config = _load_config_for_class()
        params = config.get("params", {})
        models = params.get("models", {})
        ratios = params.get("ratios", {})
        accounts = config.get("accounts", [])
        
        defaults = {"model": params.get("default_model", "3.0"), "ratio": params.get("default_ratio", "1:1")}
        model_options = list(models.keys())
        ratio_options = list(ratios.keys())
        if not model_options: model_options = ["-"]
        if not ratio_options: ratio_options = ["-"]
        
        # 生成账号选择选项
        account_options = []
        if accounts:
            for i, account in enumerate(accounts):
                description = account.get("description", f"账号{i+1}")
                account_options.append(description)
        else:
            account_options = ["无可用账号"]

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "一只可爱的小猫咪"}),
                "account": (account_options, {"default": account_options[0] if account_options else "无可用账号"}),
                "model": (model_options, {"default": defaults["model"]}),
                "ratio": (ratio_options, {"default": defaults["ratio"]}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "num_images": ("INT", {"default": 4, "min": 1, "max": 4}),
                "ref_image_1": ("IMAGE", {"tooltip": "参考图1，留空则不使用"}),
                "ref_image_2": ("IMAGE", {"tooltip": "参考图2，留空则不使用"}),
                "ref_image_3": ("IMAGE", {"tooltip": "参考图3，留空则不使用"}),
                "ref_image_4": ("IMAGE", {"tooltip": "参考图4，留空则不使用"}),
                "ref_image_5": ("IMAGE", {"tooltip": "参考图5，留空则不使用"}),
                "ref_image_6": ("IMAGE", {"tooltip": "参考图6，留空则不使用"})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "generation_info", "image_urls", "history_id")
    FUNCTION = "generate_images"
    CATEGORY = "即梦AI"
    
    def _wait_for_generation(self, history_id: str, is_image2image: bool = False) -> Optional[List[str]]:
        """
        轮询等待任务完成，支持文生图和图生图两种API。
        """
        timeout_config = self.config.get("timeout", {})
        max_wait_time = timeout_config.get("max_wait_time", 300)
        check_interval = timeout_config.get("check_interval", 5)
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            time.sleep(check_interval)
            logger.info(f"[JimengNode] 轮询任务状态: {history_id}")
            if is_image2image:
                image_urls = self.api_client._get_generated_images_by_history_id(history_id)
            else:
                image_urls = self.api_client._get_generated_images(history_id)
            if image_urls:
                logger.info(f"[JimengNode] 任务 {history_id} 生成成功")
                # 为每个URL添加history_id参数，以便下游高清化节点使用
                enhanced_urls = self._add_history_id_to_urls(image_urls, history_id)
                return enhanced_urls
        logger.error(f"[JimengNode] 等待任务 {history_id} 超时")
        return None

    def _add_history_id_to_urls(self, urls: List[str], history_id: str) -> List[str]:
        """
        为图片URL添加history_id参数，以便下游高清化节点使用。
        
        Args:
            urls: 原始图片URL列表
            history_id: 历史记录ID
            
        Returns:
            List[str]: 包含history_id参数的URL列表
        """
        enhanced_urls = []
        for url in urls:
            if '?' in url:
                # URL已经有参数，添加history_id
                enhanced_url = f"{url}&history_id={history_id}"
            else:
                # URL没有参数，直接添加history_id
                enhanced_url = f"{url}?history_id={history_id}"
            enhanced_urls.append(enhanced_url)
        return enhanced_urls

    def _download_images(self, urls: List[str]) -> List[torch.Tensor]:
        """
        下载图片并转换为torch张量。
        """
        images = []
        for url in urls:
            try:
                # 移除history_id参数，因为下载图片时不需要这个参数
                clean_url = self._remove_history_id_from_url(url)
                response = requests.get(clean_url, timeout=60)
                response.raise_for_status()
                img_data = response.content
                pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                np_image = np.array(pil_image, dtype=np.float32) / 255.0
                tensor_image = torch.from_numpy(np_image).unsqueeze(0)
                images.append(tensor_image)
            except Exception as e:
                logger.error(f"[JimengNode] 下载或处理图片失败 {url}: {e}")
                continue
        return images

    def _remove_history_id_from_url(self, url: str) -> str:
        """
        从URL中移除history_id参数，用于图片下载。
        
        Args:
            url: 包含history_id参数的URL
            
        Returns:
            str: 清理后的URL
        """
        if 'history_id=' in url:
            # 移除history_id参数
            if '&history_id=' in url:
                # history_id在中间或末尾
                url = url.replace('&history_id=', '&').replace('&&', '&')
                if url.endswith('&'):
                    url = url[:-1]
            elif '?history_id=' in url:
                # history_id是第一个参数
                if '&' in url:
                    # 还有其他参数
                    url = url.replace('?history_id=', '?').replace('&&', '&')
                else:
                    # 只有history_id参数
                    url = url.split('?history_id=')[0]
        return url

    def _extract_history_id_from_urls(self, urls) -> str:
        """
        从URL列表或字符串中提取history_id。
        
        Args:
            urls: URL列表或字符串
            
        Returns:
            str: history_id，如果无法提取则返回空字符串
        """
        try:
            if isinstance(urls, str):
                # 如果是字符串，按行分割
                url_list = [url.strip() for url in urls.split('\n') if url.strip()]
            elif isinstance(urls, list):
                url_list = urls
            else:
                return ""
            
            # 从第一个URL中提取history_id
            if url_list:
                first_url = url_list[0]
                if 'history_id=' in first_url:
                    history_id = first_url.split('history_id=')[1].split('&')[0]
                    return history_id
            
            return ""
        except Exception as e:
            logger.error(f"[JimengNode] 提取history_id失败: {e}")
            return ""

    def _save_input_image(self, image_tensor: torch.Tensor) -> str:
        """
        将输入的图像张量保存为临时文件。
        """
        try:
            temp_dir = os.path.join(self.plugin_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"temp_input_{int(time.time())}.png")
            if len(image_tensor.shape) == 4:
                image_tensor = image_tensor[0]
            image_tensor = torch.clamp(image_tensor, 0, 1)
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            image_pil.save(temp_path)
            logger.info(f"[JimengNode] 输入图像已保存到: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"[JimengNode] 保存输入图像失败: {e}")
            return None

    def _get_account_index_by_description(self, account_description: str) -> Optional[int]:
        """
        根据账号描述找到对应的账号索引
        
        Args:
            account_description: 账号描述（如"账号1"、"账号2"等）
            
        Returns:
            int: 账号索引，如果未找到则返回None
        """
        try:
            accounts = self.config.get("accounts", [])
            for i, account in enumerate(accounts):
                description = account.get("description", f"账号{i+1}")
                if description == account_description:
                    return i
            
            # 如果没有找到精确匹配，尝试处理"无可用账号"的情况
            if account_description == "无可用账号":
                return None
                
            logger.warning(f"[JimengNode] 未找到账号描述: {account_description}")
            return None
        except Exception as e:
            logger.error(f"[JimengNode] 查找账号索引时出错: {e}")
            return None

    def generate_images(self, prompt: str, model: str, ratio: str, account: str, seed: int, num_images: int = 4,
                        ref_image_1: torch.Tensor = None, ref_image_2: torch.Tensor = None, ref_image_3: torch.Tensor = None,
                        ref_image_4: torch.Tensor = None, ref_image_5: torch.Tensor = None, ref_image_6: torch.Tensor = None) -> Tuple[torch.Tensor, str, str, str]:
        """
        主执行函数：根据reference_image是否为空，自动判断文生图或图生图。
        
        Args:
            prompt: 文本提示词
            model: 使用的模型版本
            ratio: 图片比例
            account: 选择的账号描述
            seed: 随机种子
            num_images: 生成图片数量
            reference_image: 可选的参考图像，如果提供则进行图生图
            
        Returns:
            Tuple[torch.Tensor, str, str, str]: (图片张量, 生成信息, 图片URLs, 历史ID)
        """
        try:
            # --- 1. 通用检查 ---
            if not self.token_manager or not self.api_client:
                return self._create_error_result("插件未正确初始化，请检查后台日志。")
            if not self._is_configured():
                return self._create_error_result("插件未配置，请在 config.json 中至少填入一个账号的 sessionid。")
            if not prompt or not prompt.strip():
                return self._create_error_result("提示词不能为空。")

            # --- 2. 账号切换逻辑 ---
            account_index = self._get_account_index_by_description(account)
            if account_index is None:
                return self._create_error_result(f"未找到指定账号: {account}")
            
            # 切换到指定账号
            success_account = self.token_manager.switch_to_account(account_index)
            if not success_account:
                return self._create_error_result(f"切换到账号 {account} 失败")
            
            logger.info(f"[JimengNode] 已切换到账号: {account} (索引: {account_index})")
            
            # 获取账号积分信息以便后续显示
            current_credit = self.token_manager.get_credit()
            credit_display = f"当前使用账号: {account}"
            if current_credit:
                credit_display += f" (剩余积分: {current_credit.get('total_credit', '未知')})"

            # --- 3. 判断是文生图还是图生图 ---
            ref_images = [ri for ri in [ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5, ref_image_6] if ri is not None]
            if len(ref_images) > 0:
                # --- 图生图逻辑 (Image-to-Image) ---
                logger.info("=" * 50)
                logger.info(f"[JimengNode] 检测到 {len(ref_images)} 张参考图，进入图生图模式")
                logger.info(f"[JimengNode] 提示词: {prompt}")
                logger.info(f"[JimengNode] 模型: {model}, 比例: {ratio}, 数量: {num_images}")
                logger.info("-" * 50)

                # 检查当前账号积分
                current_credit_info = self.token_manager.get_credit()
                if not current_credit_info or current_credit_info.get('total_credit', 0) < 2:
                    return self._create_error_result(f"账号 {account} 积分不足2点，无法进行图生图。当前积分: {current_credit_info.get('total_credit', 0) if current_credit_info else '未知'}")
                
                # 直接调用图生图API（已扩展支持多图）
                result = self.api_client.generate_i2i(
                    images=ref_images,
                    prompt=prompt,
                    model=model,
                    ratio=ratio,
                    seed=seed,
                    num_images=num_images
                )

                # 对成功的结果补充积分信息
                if isinstance(result, tuple) and len(result) == 3 and not result[1].startswith("错误:"):
                    images, info, urls = result
                    credit_info = self.token_manager.get_credit()
                    credit_text = f"\n当前账号剩余积分: {credit_info.get('total_credit', '未知')}" if credit_info else ""
                    
                    # 补充图生图的详细日志
                    logger.info(f"[JimengNode] ✅ 图生图任务完成，成功生成 {images.shape[0]} 张图片。")
                    logger.info(f"[JimengNode] {credit_text.strip()}")
                    if urls:
                        logger.info("[JimengNode] 图片链接:")
                        # urls在这里是单个字符串，需要分割
                        for url in urls.split('\n'):
                            if url: logger.info(f"[JimengNode] - {url}")
                    logger.info("=" * 50)
                    
                    # 从urls中提取history_id（如果存在）
                    history_id_str = self._extract_history_id_from_urls(urls)
                    return (images, info + credit_text, urls, history_id_str)
                
                # 如果API调用失败或返回错误，直接返回结果
                if isinstance(result, tuple) and len(result) == 3:
                    # 兼容旧的三元组返回格式
                    images, info, urls = result
                    history_id_str = self._extract_history_id_from_urls(urls)
                    return (images, info, urls, history_id_str)
                else:
                    return self._create_error_result("图生图API调用失败")
            
            else:
                # --- 文生图逻辑 (Text-to-Image) ---
                logger.info("=" * 50)
                logger.info("[JimengNode] 未检测到参考图，进入文生图模式")
                logger.info(f"[JimengNode] 提示词: {prompt[:50]}...")
                logger.info(f"[JimengNode] 模型: {model}, 比例: {ratio}, 数量: {num_images}")
                logger.info("-" * 50)

                # 检查当前账号积分
                current_credit_info = self.token_manager.get_credit()
                if not current_credit_info or current_credit_info.get('total_credit', 0) < 1:
                    return self._create_error_result(f"账号 {account} 积分不足1点，无法进行文生图。当前积分: {current_credit_info.get('total_credit', 0) if current_credit_info else '未知'}")
                
                # 调用文生图API
                result = self.api_client.generate_t2i(prompt=prompt, model=model, ratio=ratio, seed=seed)
                
                if not result:
                    return self._create_error_result("API 调用失败，未收到任何返回。")

                # 处理API返回结果，轮询并获取URL
                if result.get("is_queued"):
                    history_id = result.get("history_id")
                    if not history_id:
                        return self._create_error_result(f"API进入排队模式但未返回任务ID。")
                    
                    logger.info(f"[JimengNode] 任务已入队(ID: {history_id})，开始轮询状态...")
                    urls = self._wait_for_generation(history_id, is_image2image=False)
                else:
                    urls = result.get("urls")
                
                if not urls:
                    return self._create_error_result("等待图片生成超时或API未返回有效URL。")

                # 下载图片
                urls_to_download = urls[:num_images]
                images = self._download_images(urls_to_download)
                if not images:
                    return self._create_error_result("图片下载失败，可能链接已失效。")
                
                image_batch = torch.cat(images, dim=0)
                
                # 准备最终输出
                credit_info = self.token_manager.get_credit()
                credit_text = f"\n当前账号剩余积分: {credit_info.get('total_credit', '未知')}" if credit_info else ""
                generation_info = self._generate_info_text(prompt, model, ratio, len(images), account) + credit_text
                image_urls_str = "\n".join(urls_to_download)

                logger.info(f"[JimengNode] ✅ 文生图任务完成，成功生成 {len(images)} 张图片。")
                logger.info(f"[JimengNode] {credit_text.strip()}")
                if urls: # urls在这里是列表
                    logger.info("[JimengNode] 图片链接:")
                    for url in urls_to_download:
                        logger.info(f"[JimengNode] - {url}")
                logger.info("=" * 50)
                
                # 准备history_id输出
                history_id_str = history_id if history_id else ""
                
                return (image_batch, generation_info, image_urls_str, history_id_str)

        except Exception as e:
            logger.exception(f"[JimengNode] 节点执行时发生意外错误")
            return self._create_error_result(f"节点执行异常: {e}")

    def _create_error_result(self, error_msg: str) -> Tuple[torch.Tensor, str, str, str]:
        logger.error(f"[JimengNode] {error_msg}")
        error_image = torch.ones(1, 256, 256, 3) * torch.tensor([1.0, 0.0, 0.0])
        return (error_image, f"错误: {error_msg}", "", "")

    def _generate_info_text(self, prompt: str, model: str, ratio: str, num_images: int, account: str = None) -> str:
        models_config = self.config.get("params", {}).get("models", {})
        model_display_name = models_config.get(model, {}).get("name", model)
        info_lines = [f"提示词: {prompt}", f"模型: {model_display_name}", f"比例: {ratio}", f"数量: {num_images}"]
        if account:
            info_lines.append(f"使用账号: {account}")
        return "\n".join(info_lines)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "Jimeng_Image": JimengImageNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Jimeng_Image": "即梦AI图片生成"
} 
