import os
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv
from flask import (
    Flask, render_template, redirect,
    url_for, request, flash, jsonify
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# 加载 .env 文件中的环境变量
load_dotenv()

# ================== 基础配置 ==================

app = Flask(__name__)

# Flask 必需配置
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'change-this-secret-key')

# 数据库：用 sqlite 本地文件
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI', 'sqlite:///psych_doctor.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'  # 未登录访问受保护页面会跳转到 login

# ================== GPT API 配置 ==================

# 从环境变量里读，如果没设置就用默认值
GPT_BASE_URL = os.getenv('GPT_BASE_URL', 'https://aizex.top/v1')
GPT_API_KEY = os.getenv('GPT_API_KEY', '')
GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-5')

# 中国时区（UTC+8）
CHINA_TZ = timezone(timedelta(hours=8))


def utc_to_china_time(utc_dt):
    """
    将UTC时间转换为中国时区时间（UTC+8）
    处理SQLAlchemy返回的naive datetime对象（假设它们是UTC时间）
    """
    if utc_dt is None:
        return None
    try:
        # SQLAlchemy返回的datetime对象通常是naive（没有时区信息）
        # 我们假设数据库中存储的是UTC时间
        if hasattr(utc_dt, 'tzinfo') and utc_dt.tzinfo is None:
            # naive datetime，假设是UTC时间，添加UTC时区信息
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        elif not hasattr(utc_dt, 'tzinfo'):
            # 如果不是datetime对象，直接返回
            return utc_dt
        
        # 转换到中国时区
        return utc_dt.astimezone(CHINA_TZ)
    except (AttributeError, TypeError, ValueError) as e:
        # 如果转换失败，直接返回原时间（向后兼容）
        print(f"Warning: Timezone conversion failed for {utc_dt}: {e}")
        return utc_dt


def call_gpt_api(messages):
    """
    调用 GPT 兼容接口：
    - messages: OpenAI 风格的对话 [{"role": "...", "content": "..."}]
    - 返回：字符串回复；如果失败返回一个简单提示。
    """
    if not GPT_API_KEY:
        return "（后端提示：尚未配置 GPT_API_KEY，无法调用大模型。请联系管理员。）"

    url = GPT_BASE_URL.rstrip('/') + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {GPT_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GPT_MODEL,
        "messages": messages,
        "temperature": 0.8,  # 稍高的温度，使回复更自然、更有人情味
        "top_p": 0.9,  # 核采样，增加回复的多样性
        "max_tokens": 500,  # 限制最大长度，保持回复简洁
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # 打到后端日志里，方便调试
        print("GPT API ERROR:", e)
        return "（后端提示：调用大模型接口失败了，可以稍后再试，或联系管理员检查配置。）"


# ================== 数据模型 ==================

class User(UserMixin, db.Model):
    """
    用户表：
    - username 唯一（用户名判重）
    - password_hash 保存加密后的密码
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Message(db.Model):
    """
    聊天记录表：
    - user_id: 属于哪个用户
    - content: 内容
    - created_at: 时间（用于显示日期:时:分）
    - is_bot: True 表示心理医生机器人说的，False 表示用户
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id'),
        nullable=False
    )
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    is_bot = db.Column(db.Boolean, default=False, nullable=False)

    user = db.relationship(
        'User',
        backref=db.backref('messages', lazy=True)
    )


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



# ================== 生成心理医生回复（调用 GPT） ==================

def generate_psych_reply(user: User, user_text: str) -> str:
    """
    构造上下文 + system prompt，调用 GPT。
    要求 GPT：
    - 检测用户心情不好时要进行安慰
    - 记住历史聊天内容
    - 当用户问“之前说的啥”时，从历史中回顾回答
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return "我好像没有听清楚，你可以再说一遍吗？"

    # 取最近 N 条历史记录作为上下文
    # 包括用户 + 机器人，按照时间从旧到新
    # 增加历史记录数量以提供更好的上下文理解
    history = (
        Message.query
        .filter_by(user_id=user.id)
        .order_by(Message.created_at.asc())
        .limit(40)  # 增加历史记录数量，提供更丰富的上下文
        .all()
    )

    messages = []

    # system 提示：在这里定义"心理医生"的角色、情绪安慰、记住历史等要求
    system_prompt = """
你是一位专业、温暖、共情能力强的中文心理咨询师，正在通过文字与来访者进行在线心理咨询。

核心原则：
1. **共情理解**：首先真诚地理解来访者的感受，让ta感受到被理解和接纳。用"我能理解你的感受"、"听起来你真的很难受"等表达共情。

2. **自然对话**：用自然、温和、口语化的中文交流，就像朋友间的真诚对话。避免使用过于官方、刻板或鸡汤式的表达。适当使用"嗯"、"我明白"等自然语气词。

3. **情绪识别与回应**：
   - 识别关键词：难受、不开心、郁闷、抑郁、压力大、焦虑、害怕、愤怒、失望、孤独、绝望、崩溃、想哭、睡不着等
   - 当察觉负面情绪时：
     a) 先表达理解和共情（1-2句）
     b) 深入询问具体情况，帮助来访者表达内心感受
     c) 提供具体、可操作的建议或视角（2-3句）
     d) 给予希望和支持

4. **上下文记忆**：
   - 记住对话历史中的重要信息（工作、学习、人际关系等）
   - 当来访者提到"之前说的"、"刚才"、"上次"时，主动回忆并引用相关内容
   - 跟踪情绪变化，关注连续对话中的情绪发展

5. **危机干预**：
   - 识别危险信号：自杀念头、自伤行为、长期严重失眠、强烈的绝望感等
   - 温和但明确地建议寻求专业帮助（学校心理咨询中心、医院心理门诊、24小时心理热线等）
   - 表达关心和陪伴，不让来访者感到被抛弃

6. **回复长度**：
   - 一般情况：3-6句话，简洁有力
   - 深度共情场景：可以适当延长，但不超过10句话
   - 注意分段，使用换行让回复更易读

7. **个性化回应**：
   - 根据来访者的年龄、身份、问题类型调整语言风格
   - 对年轻学生：更亲切、鼓励性
   - 对职场人士：更理性、实用
   - 对情感问题：更细腻、温暖

8. **提问技巧**：
   - 使用开放式问题帮助来访者表达："能具体说说发生了什么吗？"
   - 避免连续提问，给来访者充分表达的空间
   - 在适当时候给予肯定："你能说出来已经很勇敢了"

请始终以专业、温暖、支持的态度陪伴来访者，帮助ta探索内心、缓解情绪、找到前进的方向。
"""
    messages.append({"role": "system", "content": system_prompt})

    # 把历史消息转换成 openai 风格 messages
    for m in history:
        role = "assistant" if m.is_bot else "user"
        messages.append({
            "role": role,
            "content": m.content
        })

    # 当前用户输入
    messages.append({"role": "user", "content": user_text})

    # 调用 GPT
    reply = call_gpt_api(messages)
    return reply


# ================== 路由：首页 / 注册 / 登录 / 登出 ==================

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    # 已登录则直接跳到聊天页
    if current_user.is_authenticated:
        return redirect(url_for('chat'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        password2 = request.form.get('password2', '').strip()

        if not username or not password:
            flash('用户名和密码不能为空', 'error')
            return redirect(url_for('register'))

        if password != password2:
            flash('两次输入的密码不一致', 'error')
            return redirect(url_for('register'))

        # 用户名判重
        if User.query.filter_by(username=username).first():
            flash('该用户名已存在，请换一个。', 'error')
            return redirect(url_for('register'))

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('注册成功，请登录。', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = User.query.filter_by(username=username).first()
        if user is None or not user.check_password(password):
            flash('用户名或密码错误。', 'error')
            return redirect(url_for('login'))

        login_user(user)
        flash('登录成功。', 'success')
        return redirect(url_for('chat'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('已退出登录。', 'success')
    return redirect(url_for('login'))


# ================== 路由：聊天 / 加载更多 / 删除消息 ==================

@app.route('/chat', methods=['GET'])
@login_required
def chat():
    """
    聊天页：
    - GET：显示最近 limit 条消息 + "加载更多"按钮
    """
    # 默认显示最近 10 条，可以通过 ?limit=20,30... 加载更多
    limit = request.args.get('limit', 10, type=int)
    if limit < 5:
        limit = 5

    total = Message.query.filter_by(user_id=current_user.id).count()
    start = max(0, total - limit)

    messages = (
        Message.query
        .filter_by(user_id=current_user.id)
        .order_by(Message.created_at.asc())
        .offset(start)
        .limit(limit)
        .all()
    )

    has_more = limit < total

    return render_template(
        'chat.html',
        messages=messages,
        limit=limit,
        has_more=has_more,
    )


@app.route('/api/send_message', methods=['POST'])
@login_required
def send_message():
    """
    API端点：异步发送消息并获取AI回复
    返回JSON格式：{user_message: {...}, bot_message: {...}}
    """
    data = request.get_json()
    content = (data.get('message', '') or '').strip()
    
    if not content:
        return jsonify({'error': '消息内容不能为空'}), 400

    try:
        # 1. 先保存用户说的话
        user_msg = Message(
            user_id=current_user.id,
            content=content,
            is_bot=False
        )
        db.session.add(user_msg)
        db.session.commit()

        # 2. 调用 GPT 生成心理医生回复
        reply_text = generate_psych_reply(current_user, content)

        # 3. 保存机器人回复
        bot_msg = Message(
            user_id=current_user.id,
            content=reply_text,
            is_bot=True
        )
        db.session.add(bot_msg)
        db.session.commit()

        # 返回JSON格式的消息数据，时间转换为中国时区
        user_time = utc_to_china_time(user_msg.created_at)
        bot_time = utc_to_china_time(bot_msg.created_at)
        
        return jsonify({
            'user_message': {
                'id': user_msg.id,
                'content': user_msg.content,
                'created_at': user_time.strftime('%Y-%m-%d %H:%M') if user_time else user_msg.created_at.strftime('%Y-%m-%d %H:%M'),
                'is_bot': False
            },
            'bot_message': {
                'id': bot_msg.id,
                'content': bot_msg.content,
                'created_at': bot_time.strftime('%Y-%m-%d %H:%M') if bot_time else bot_msg.created_at.strftime('%Y-%m-%d %H:%M'),
                'is_bot': True
            }
        })
    except Exception as e:
        db.session.rollback()
        print(f"Error sending message: {e}")
        return jsonify({'error': '发送消息时出错，请稍后再试'}), 500


# 注册Jinja2过滤器：将UTC时间转换为中国时区并格式化
@app.template_filter('china_time')
def china_time_filter(dt):
    """将UTC时间转换为中国时区并格式化为字符串"""
    if dt:
        try:
            china_dt = utc_to_china_time(dt)
            if china_dt:
                return china_dt.strftime('%Y-%m-%d %H:%M')
        except Exception as e:
            print(f"Warning: Template time filter error: {e}")
            # 如果转换失败，尝试直接格式化
            try:
                return dt.strftime('%Y-%m-%d %H:%M')
            except:
                return str(dt)
    return ''


@app.route('/delete_message/<int:message_id>', methods=['POST'])
@login_required
def delete_message(message_id):
    """
    删除任意一句话：
    - 只允许删除当前登录用户自己的消息（包括他自己的和机器人回给他的）
    """
    msg = Message.query.get_or_404(message_id)
    if msg.user_id != current_user.id:
        # 检查是否是Ajax请求（通过Content-Type或Accept头判断）
        wants_json = request.is_json or request.headers.get('Content-Type', '').startswith('application/json')
        if wants_json:
            return jsonify({'error': '你不能删除别人的消息。'}), 403
        flash('你不能删除别人的消息。', 'error')
        return redirect(url_for('chat'))

    db.session.delete(msg)
    db.session.commit()
    
    # 检查是否是Ajax请求
    wants_json = request.is_json or request.headers.get('Content-Type', '').startswith('application/json')
    if wants_json:
        return jsonify({'success': True})
    flash('已删除这一条消息。', 'success')
    return redirect(url_for('chat'))


# ================== 入口 ==================

if __name__ == '__main__':
    # 程序启动时建一次表
    with app.app_context():
        db.create_all()

    app.run(host='0.0.0.0', port=5000, debug=True)

