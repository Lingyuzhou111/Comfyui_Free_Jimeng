from .jimeng_image_node import JimengImageNode
from .jimeng_hd_enhancer_node import JimengHDEnhancerNode
from .jimeng_video_node import JimengVideoNode

# 节点类映射 - 注册所有节点
NODE_CLASS_MAPPINGS = {
    "Jimeng_Image": JimengImageNode,
    "Jimeng_HD_Enhancer": JimengHDEnhancerNode,
    "Jimeng_Video": JimengVideoNode,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "Jimeng_Video": "即梦AI视频",
    "Jimeng_Image": "即梦AI生图",
    "Jimeng_HD_Enhancer": "即梦AI图片高清化",
}

__version__ = "2.1.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 