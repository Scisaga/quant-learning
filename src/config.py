import os, logging, sys
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录

## 读取 `.env` 作为环境变量
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(str(ENV_PATH))

# https://open.longbridge.com/account
LONGPORT_APP_KEY = os.getenv("LONGPORT_APP_KEY")
LONGPORT_APP_SECRET = os.getenv("LONGPORT_APP_SECRET")
LONGPORT_ACCESS_TOKEN = os.getenv("LONGPORT_ACCESS_TOKEN")

## 设定日志
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("qlearning") # 命名 logger
logger.setLevel(logging.INFO)
logger.propagate = False # 防止重复打印到 root logger

if not logger.handlers:
    
    # 文件日志
    fh = logging.FileHandler(os.path.join(log_dir, "qlearning.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # 控制台日志
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # 添加 handler
    logger.addHandler(fh)
    logger.addHandler(ch)
