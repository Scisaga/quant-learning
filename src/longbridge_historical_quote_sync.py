from config import logger
import pandas as pd
from longport.openapi import Config, QuoteContext

# Load configuration from environment variables
config = Config.from_env()

# Create a context for quote APIs
ctx = QuoteContext(config)

# Get basic information of securities
resp = ctx.quote(["600036.SH"])

print(resp)

df = pd.read_csv("data/a_share_list_filtered_20251021.csv")
symbols = df["symbol"].head(10).tolist()  # 取前10个测通路

print(symbols)