import qlib
from qlib.config import REG_CN  # 如需使用美国数据，则为 REG_US

provider_uri = "data/cn_data"  # 你的数据存放路径
qlib.init(provider_uri=provider_uri, region=REG_CN)