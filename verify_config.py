"""验证配置加载。"""

from agent.config import get_config

config = get_config()
print(f"数据库URI: {config.database.uri}")
print(f"LLM Provider: {config.llm.provider}")
print(f"LLM Model: {config.llm.model}")
