import os
from dotenv import load_dotenv
from agent.tools.web_search import web_search

# 1. 在脚本最开始自动加载项目根目录下的 .env 文件
# 这会将 .env 中的 EXA_API_KEY 自动注入到 os.environ 中
load_dotenv()

# 2. 检查环境变量是否成功加载（不要手动用 "dummy_key_to_trigger_request" 覆盖它）
exa_key = os.getenv("EXA_API_KEY")
if not exa_key:
    print("错误: 未在 .env 文件或环境变量中找到 EXA_API_KEY！")
else:
    print(f"成功读取 EXA_API_KEY: {exa_key[:8]}... (省略后半部分)")

# 调用 web_search，若处于无网环境，将立刻复现 NameResolutionError
print(web_search("Hello World"))