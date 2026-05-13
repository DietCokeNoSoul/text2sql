"""一键启动 Web 服务（FastAPI 后端）。

使用方法：
    uv run python web/start.py

前端开发模式（热更新）：
    cd web/frontend && npm run dev
    → 访问 http://localhost:5173

生产模式（先构建前端）：
    cd web/frontend && npm run build
    uv run python web/start.py
    → 访问 http://localhost:8000
"""

import os
import sys

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("WEB_PORT", "8000"))
    print(f"\n🚀 Text2SQL Web UI 启动中...")
    print(f"   后端 API: http://localhost:{port}/docs")
    print(f"   前端界面: http://localhost:{port}/")
    print(f"   (开发模式前端: http://localhost:5173/)\n")
    uvicorn.run(
        "web.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
