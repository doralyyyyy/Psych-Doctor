# Psych-Doctor

## 配置

1. 复制环境变量模板并重命名为 `.env`：

   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env`，填写实际配置：

   | 变量 | 说明 |
   |------|------|
   | `SECRET_KEY` | Flask 会话密钥，建议随机字符串 |
   | `GPT_BASE_URL` | GPT 兼容 API 的 base URL |
   | `GPT_API_KEY` | API 密钥 |
   | `GPT_MODEL` | 模型名称（如 `gpt-5`） |

## 运行

1. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 启动应用：

   ```bash
   python app.py
   ```

   默认访问地址：<http://127.0.0.1:5000>。
