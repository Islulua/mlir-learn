# 部署说明

本文档说明如何部署 MLIR 学习笔记网站到不同的平台。

## 🚀 本地开发

### 环境要求
- Python 3.8+
- pip 包管理器

### 安装依赖
```bash
# 安装所有依赖
pip3 install --break-system-packages -r requirements.txt

# 或者使用用户安装（推荐）
pip3 install --user -r requirements.txt
```

### 启动开发服务器
```bash
# 设置 PATH
export PATH=$PATH:$HOME/.local/bin

# 启动服务器
mkdocs serve --dev-addr=0.0.0.0:8080

# 或者使用默认端口
mkdocs serve
```

### 访问网站
- 本地访问: http://localhost:8080/mlir-learn/
- 网络访问: http://your-ip:8080/mlir-learn/

## 🌐 生产部署

### 构建静态文件
```bash
# 构建网站
mkdocs build

# 构建后的文件在 site/ 目录中
ls -la site/
```

### 部署到 Web 服务器

#### Nginx 配置示例
```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/site;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # 静态资源缓存
    location ~* \.(css|js|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

#### Apache 配置示例
```apache
<VirtualHost *:80>
    ServerName your-domain.com
    DocumentRoot /path/to/site
    
    <Directory /path/to/site>
        Options Indexes FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>
    
    # 启用重写规则
    RewriteEngine On
    RewriteCond %{REQUEST_FILENAME} !-f
    RewriteCond %{REQUEST_FILENAME} !-d
    RewriteRule ^(.*)$ /index.html [L]
</VirtualHost>
```

## ☁️ 云平台部署

### GitHub Pages

#### 自动部署
```bash
# 安装 ghp-import
pip3 install --user ghp-import

# 部署到 GitHub Pages
mkdocs gh-deploy
```

#### 手动部署
```bash
# 构建网站
mkdocs build

# 推送到 gh-pages 分支
cd site
git init
git add .
git commit -m "Deploy website"
git branch -M gh-pages
git remote add origin https://github.com/username/repo.git
git push -u origin gh-pages
```

### Netlify

1. 连接 GitHub 仓库
2. 设置构建命令: `mkdocs build`
3. 设置发布目录: `site`
4. 自动部署

### Vercel

1. 导入 GitHub 仓库
2. 设置构建命令: `pip install -r requirements.txt && mkdocs build`
3. 设置输出目录: `site`
4. 自动部署

### Docker 部署

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制源代码
COPY . .

# 构建网站
RUN mkdocs build

# 使用 nginx 服务静态文件
FROM nginx:alpine
COPY --from=0 /app/site /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### 构建和运行
```bash
# 构建镜像
docker build -t mlir-docs .

# 运行容器
docker run -d -p 80:80 mlir-docs
```

## 🔧 环境配置

### 环境变量
```bash
# 设置网站 URL
export SITE_URL=https://your-domain.com

# 设置 Google Analytics
export GOOGLE_ANALYTICS_KEY=your-key

# 设置搜索配置
export SEARCH_INDEX_URL=https://your-domain.com/search_index.json
```

### 配置文件
```yaml
# mkdocs.yml
site_url: https://your-domain.com
extra:
  analytics:
    provider: google
    property: !ENV GOOGLE_ANALYTICS_KEY
```

## 📱 CDN 配置

### Cloudflare
1. 添加域名到 Cloudflare
2. 设置 DNS 记录
3. 启用 CDN 和 SSL
4. 配置页面规则

### AWS CloudFront
1. 创建 CloudFront 分发
2. 设置源为 S3 或自定义源
3. 配置缓存行为
4. 设置 SSL 证书

## 🔒 安全配置

### HTTPS 配置
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # 安全头
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
}
```

### 访问控制
```nginx
# 基本认证
auth_basic "Restricted Access";
auth_basic_user_file /path/to/.htpasswd;

# IP 白名单
allow 192.168.1.0/24;
allow 10.0.0.0/8;
deny all;
```

## 📊 监控和分析

### 日志配置
```nginx
# 访问日志
access_log /var/log/nginx/access.log combined;

# 错误日志
error_log /var/log/nginx/error.log warn;
```

### 性能监控
```bash
# 检查网站性能
curl -w "@curl-format.txt" -o /dev/null -s "http://your-domain.com"

# 监控响应时间
watch -n 1 'curl -s -w "%{time_total}\n" -o /dev/null http://your-domain.com'
```

## 🚨 故障排除

### 常见问题

#### 1. 端口被占用
```bash
# 查找占用端口的进程
lsof -i :8080
# 或者
netstat -tlnp | grep :8080

# 杀死进程
kill -9 <PID>
```

#### 2. 权限问题
```bash
# 检查文件权限
ls -la site/

# 修复权限
chmod -R 755 site/
chown -R www-data:www-data site/
```

#### 3. 依赖问题
```bash
# 清理缓存
pip3 cache purge

# 重新安装依赖
pip3 install --force-reinstall -r requirements.txt
```

### 调试模式
```bash
# 启用详细日志
mkdocs serve --verbose

# 检查配置
mkdocs --version
mkdocs --help
```

## 📈 性能优化

### 静态资源优化
```bash
# 压缩 CSS 和 JS
pip3 install --user csscompressor jsmin

# 优化图片
pip3 install --user Pillow
```

### 缓存策略
```nginx
# 静态资源缓存
location ~* \.(css|js|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}

# HTML 缓存
location ~* \.html$ {
    expires 1h;
    add_header Cache-Control "public, must-revalidate";
}
```

## 🔄 自动化部署

### GitHub Actions
```yaml
name: Deploy Website
on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Build website
      run: mkdocs build
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./site
```

### 本地脚本
```bash
#!/bin/bash
# deploy.sh

echo "Building website..."
mkdocs build

echo "Deploying to server..."
rsync -avz --delete site/ user@server:/var/www/html/

echo "Deployment complete!"
```

## 📝 维护说明

### 定期任务
- 更新依赖包
- 检查安全更新
- 备份网站数据
- 监控性能指标

### 更新流程
1. 修改源代码
2. 本地测试
3. 提交到版本控制
4. 自动或手动部署
5. 验证部署结果

---

**🎯 部署完成后，你的 MLIR 学习笔记网站就可以通过配置的域名访问了！**