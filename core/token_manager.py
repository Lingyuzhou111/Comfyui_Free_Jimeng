import requests
import json
import os
import random
import hashlib
import time
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class TokenManager:
    def __init__(self, config):
        self.config = config
        self.accounts = config.get("accounts", [])
        self.current_account_index = 0
        self.version_code = "5.8.0"
        self.platform_code = "7"
        self.device_id = str(random.random() * 999999999999999999 + 7000000000000000000)
        self.web_id = str(random.random() * 999999999999999999 + 7000000000000000000)
        self.user_id = str(random.random() * 999999999999999999 + 7000000000000000000) # Changed to generate a random user_id
        
        logger.info(f"[Jimeng] Initialized with {len(self.accounts)} accounts")
        
        # 初始化时为每个账号生成一个web_id
        for account in self.accounts:
            if not hasattr(account, 'web_id'):
                account['web_id'] = self._generate_web_id()
        
        self._extract_web_id_from_cookie()
        
        # 如果没有从cookie中提取到web_id，则生成一个新的
        if not self.web_id:
            self.web_id = self._generate_web_id()
        
    def _extract_web_id_from_cookie(self):
        """从cookie中提取web_id"""
        try:
            account = self.get_current_account()
            if not account:
                return
                
            # 如果账号已有web_id，直接使用
            if account.get('web_id'):
                self.web_id = account['web_id']
                return
                
            # 否则生成新的web_id
            account['web_id'] = self._generate_web_id()
            self.web_id = account['web_id']
            
        except Exception as e:
            logger.error(f"[Jimeng] Failed to extract web_id from cookie: {e}")
            # 出错时生成新的web_id
            self.web_id = self._generate_web_id()
        
    def _generate_web_id(self):
        """生成新的web_id"""
        # 生成一个19位的随机数字字符串
        web_id = ''.join([str(random.randint(0, 9)) for _ in range(19)])
        return web_id
        
    def get_web_id(self):
        """获取web_id"""
        if not self.web_id:
            self.web_id = self._generate_web_id()
        return self.web_id
        
    def get_current_account(self):
        """获取当前账号"""
        if not self.accounts:
            return None
        return self.accounts[self.current_account_index]
        
    def switch_to_account(self, account_index):
        """切换到指定账号"""
        if not self.accounts:
            raise Exception("No accounts configured")
        if account_index < 0 or account_index >= len(self.accounts):
            logger.error(f"[Jimeng] Invalid account index: {account_index}, total accounts: {len(self.accounts)}")
            return None
        self.current_account_index = account_index
        logger.info(f"[Jimeng] Switched to account {account_index + 1}")
        return self.get_current_account()
        
    def get_account_count(self):
        """获取账号总数"""
        return len(self.accounts)

    def find_account_with_sufficient_credit(self, required_credit):
        """查找有足够积分的账号"""
        original_index = self.current_account_index
        
        # 检查所有账号
        for i in range(len(self.accounts)):
            credit_info = self.get_credit()
            if credit_info and credit_info["total_credit"] >= required_credit:
                logger.info(f"[Jimeng] Found account with sufficient credit: {credit_info['total_credit']}")
                return self.get_current_account()
            
            # 切换到下一个账号
            next_index = (self.current_account_index + 1) % len(self.accounts)
            self.switch_to_account(next_index)
        
        # 如果没有找到合适的账号，恢复原始账号
        self.switch_to_account(original_index)
        return None

    def get_token(self, api_path="/"):
        """获取token信息
        Args:
            api_path: API路径，用于生成不同的签名
        Returns:
            dict: token信息
        """
        try:
            account = self.get_current_account()
            if not account:
                return None
                
            # 获取当前时间戳
            timestamp = str(int(time.time()))
            
            # 生成新的msToken
            msToken = self._generate_ms_token()
            
            # 生成新的sign
            sign = self._generate_sign(api_path, timestamp)
            
            # 生成新的a_bogus
            a_bogus = self._generate_a_bogus(api_path, timestamp)
            
            # 生成新的cookie
            cookie = self._generate_cookie(account)
            
            return {
                "cookie": cookie,
                "msToken": msToken,
                "sign": sign,
                "a_bogus": a_bogus,
                "device_time": timestamp
            }
            
        except Exception as e:
            logger.error(f"[Jimeng] Error getting token: {str(e)}")
            return None
            
    def _generate_ms_token(self):
        """生成msToken"""
        # 生成107位随机字符串
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return ''.join(random.choice(chars) for _ in range(107))
        
    def _generate_sign(self, api_path, timestamp):
        """生成sign
        Args:
            api_path: API路径
            timestamp: 时间戳
        Returns:
            str: sign字符串
        """
        # 使用固定的key
        sign_str = f"9e2c|{api_path[-7:]}|{self.platform_code}|{self.version_code}|{timestamp}||11ac"
        return hashlib.md5(sign_str.encode()).hexdigest()
        
    def _generate_a_bogus(self, api_path, timestamp):
        """生成a_bogus
        Args:
            api_path: API路径
            timestamp: 时间戳
        Returns:
            str: a_bogus字符串
        """
        # 生成32位随机字符串
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return ''.join(random.choice(chars) for _ in range(32))
        
    def _generate_cookie(self, account):
        """生成完整的cookie
        Args:
            account: 账号信息
        Returns:
            str: 完整的cookie字符串
        """
        try:
            # 获取基本信息
            sessionid = account.get("sessionid", "")
            timestamp = int(time.time())
            
            # 生成过期时间（60天后）
            expire_time = timestamp + 60 * 24 * 60 * 60
            expire_date = time.strftime("%a, %d-%b-%Y %H:%M:%S GMT", time.gmtime(expire_time))
            
            # 使用账号的web_id或生成新的
            web_id = account.get('web_id', self._generate_web_id())
            if not account.get('web_id'):
                account['web_id'] = web_id
            
            # 构建cookie部分
            cookie_parts = [
                f"sessionid={sessionid}",
                f"sessionid_ss={sessionid}",
                f"_tea_web_id={web_id}",
                f"web_id={web_id}",
                f"_v2_spipe_web_id={web_id}",
                f"uid_tt={self.user_id}",
                f"uid_tt_ss={self.user_id}",
                f"sid_tt={sessionid}",
                f"sid_guard={sessionid}%7C{timestamp}%7C5184000%7C{expire_date}",
                f"ssid_ucp_v1=1.0.0-{hashlib.md5((sessionid + str(timestamp)).encode()).hexdigest()}",
                f"sid_ucp_v1=1.0.0-{hashlib.md5((sessionid + str(timestamp)).encode()).hexdigest()}",
                "store-region=cn-gd",
                "store-region-src=uid",
                "is_staff_user=false"
            ]
            
            return "; ".join(cookie_parts)
            
        except Exception as e:
            logger.error(f"[Jimeng] Error generating cookie: {str(e)}")
            return ""

    def get_credit(self):
        """获取积分信息"""
        url = "https://jimeng.jianying.com/commerce/v1/benefits/user_credit"
        
        params = {
            "aid": "513695",
            "device_platform": "web",
            "region": "CN"
        }
        
        token_info = self.get_token("/commerce/v1/benefits/user_credit")
        if not token_info:
            return None
            
        headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'zh-CN,zh;q=0.9',
            'app-sdk-version': self.version_code,
            'appid': '513695',
            'appvr': self.version_code,
            'content-type': 'application/json',
            'cookie': token_info["cookie"],
            'device-time': token_info["device_time"],
            'sign': token_info["sign"],
            'sign-ver': '1',
            'pf': self.platform_code,
            'priority': 'u=1, i',
            'referer': 'https://jimeng.jianying.com/ai-tool/image/generate',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'msToken': token_info["msToken"],
            'x-bogus': token_info["a_bogus"]
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json={})
            result = response.json()
            
            if result.get("ret") == "0" and result.get("data"):
                credit_data = result["data"]["credit"]
                logger.info(f"[Jimeng] Credit info for account {self.current_account_index + 1}: gift={credit_data['gift_credit']}, purchase={credit_data['purchase_credit']}, vip={credit_data['vip_credit']}")
                return {
                    "gift_credit": credit_data["gift_credit"],
                    "purchase_credit": credit_data["purchase_credit"],
                    "vip_credit": credit_data["vip_credit"],
                    "total_credit": credit_data["gift_credit"] + credit_data["purchase_credit"] + credit_data["vip_credit"]
                }
            else:
                logger.error(f"[Jimeng] Failed to get credit info: {result}")
                return None
                
        except Exception as e:
            logger.error(f"[Jimeng] Error in get_credit: {str(e)}")
            return None
            
    def receive_daily_credit(self):
        """领取每日积分"""
        url = "https://jimeng.jianying.com/commerce/v1/benefits/credit_receive"
        
        params = {
            "aid": "513695",
            "device_platform": "web",
            "region": "CN"
        }
        
        token_info = self.get_token("/commerce/v1/benefits/credit_receive")
        if not token_info:
            return None
            
        headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'zh-CN,zh;q=0.9',
            'app-sdk-version': self.version_code,
            'appid': '513695',
            'appvr': self.version_code,
            'content-type': 'application/json',
            'cookie': token_info["cookie"],
            'device-time': token_info["device_time"],
            'sign': token_info["sign"],
            'sign-ver': '1',
            'pf': self.platform_code,
            'priority': 'u=1, i',
            'referer': 'https://jimeng.jianying.com/ai-tool/image/generate',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'msToken': token_info["msToken"],
            'x-bogus': token_info["a_bogus"]
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json={"time_zone": "Asia/Shanghai"})
            result = response.json()
            
            if result.get("ret") == "0" and result.get("data"):
                data = result["data"]
                logger.info(f"[Jimeng] Account {self.current_account_index + 1} received daily credit: {data['receive_quota']}, total credit: {data['cur_total_credits']}")
                return data['cur_total_credits']
            else:
                logger.error(f"[Jimeng] Failed to receive daily credit: {result}")
                return None
                
        except Exception as e:
            logger.error(f"[Jimeng] Error in receive_daily_credit: {str(e)}")
            return None

    def get_upload_token(self):
        """获取上传token"""
        url = "https://jimeng.jianying.com/mweb/v1/get_upload_token"
        
        params = {
            "aid": "513695",
            "device_platform": "web",
            "region": "CN"
        }
        
        # 获取最新的token信息
        token_info = self.get_token("/mweb/v1/get_upload_token")
        if not token_info:
            return None
            
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
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'msToken': token_info["msToken"],
            'x-bogus': token_info["a_bogus"]
        }
        
        data = {
            "scene": 2
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data)
            result = response.json()
            
            if result.get("ret") == "0" and result.get("data"):
                token_data = result["data"]
                return {
                    "access_key_id": token_data["access_key_id"],
                    "secret_access_key": token_data["secret_access_key"],
                    "session_token": token_data["session_token"],
                    "space_name": token_data["space_name"],
                    "upload_domain": token_data["upload_domain"]
                }
            else:
                logger.error(f"[Jimeng] Failed to get upload token: {result}")
                return None
                
        except Exception as e:
            logger.error(f"[Jimeng] Error in get_upload_token: {str(e)}")
            return None