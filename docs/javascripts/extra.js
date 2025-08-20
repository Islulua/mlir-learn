// MLIR 学习笔记自定义 JavaScript 功能

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    
    // 添加代码块复制按钮
    addCopyButtons();
    
    // 添加 MLIR 语法高亮
    highlightMLIRCode();
    
    // 添加表格排序功能
    addTableSorting();
    
    // 添加返回顶部按钮
    addBackToTopButton();
    
    // 添加进度条
    addReadingProgress();
});

// 为代码块添加复制按钮
function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach((codeBlock, index) => {
        const pre = codeBlock.parentElement;
        
        // 检查是否已经有复制按钮
        if (pre.querySelector('.copy-button')) {
            return;
        }
        
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.innerHTML = '📋';
        copyButton.title = '复制代码';
        copyButton.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            background: #f1f3f4;
            border: 1px solid #dadce0;
            border-radius: 4px;
            padding: 4px 8px;
            cursor: pointer;
            font-size: 12px;
            opacity: 0.8;
            transition: opacity 0.2s;
        `;
        
        // 设置 pre 为相对定位
        pre.style.position = 'relative';
        
        copyButton.addEventListener('click', function() {
            const text = codeBlock.textContent;
            navigator.clipboard.writeText(text).then(function() {
                copyButton.innerHTML = '✅';
                copyButton.style.background = '#34a853';
                copyButton.style.color = 'white';
                
                setTimeout(function() {
                    copyButton.innerHTML = '📋';
                    copyButton.style.background = '#f1f3f4';
                    copyButton.style.color = 'black';
                }, 2000);
            });
        });
        
        pre.appendChild(copyButton);
    });
}

// MLIR 语法高亮
function highlightMLIRCode() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(codeBlock => {
        if (codeBlock.textContent.includes('func') || 
            codeBlock.textContent.includes('tensor') ||
            codeBlock.textContent.includes('dialect')) {
            
            let html = codeBlock.innerHTML;
            
            // 高亮 MLIR 关键字
            html = html.replace(/\b(func|return|tensor|memref|dialect)\b/g, 
                '<span class="mlir-keyword">$1</span>');
            
            // 高亮操作符
            html = html.replace(/(=|\+|-|\*|\/|->)/g, 
                '<span class="mlir-operator">$1</span>');
            
            // 高亮类型
            html = html.replace(/(tensor<[^>]+>|memref<[^>]+>|i\d+|f\d+)/g, 
                '<span class="mlir-type">$1</span>');
            
            // 高亮值引用
            html = html.replace(/(%\w+)/g, 
                '<span class="mlir-value">$1</span>');
            
            codeBlock.innerHTML = html;
        }
    });
}

// 为表格添加排序功能
function addTableSorting() {
    const tables = document.querySelectorAll('table');
    
    tables.forEach(table => {
        const headers = table.querySelectorAll('th');
        
        headers.forEach((header, index) => {
            header.style.cursor = 'pointer';
            header.addEventListener('click', function() {
                sortTable(table, index);
            });
            
            // 添加排序指示器
            header.innerHTML += ' <span class="sort-indicator">↕</span>';
        });
    });
}

// 表格排序函数
function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    rows.sort((a, b) => {
        const aValue = a.cells[columnIndex].textContent.trim();
        const bValue = b.cells[columnIndex].textContent.trim();
        
        // 尝试数字排序
        const aNum = parseFloat(aValue);
        const bNum = parseFloat(bValue);
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return aNum - bNum;
        }
        
        // 字符串排序
        return aValue.localeCompare(bValue);
    });
    
    // 重新插入排序后的行
    rows.forEach(row => tbody.appendChild(row));
}

// 添加返回顶部按钮
function addBackToTopButton() {
    const button = document.createElement('button');
    button.innerHTML = '↑';
    button.title = '返回顶部';
    button.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 50px;
        height: 50px;
        background: #007acc;
        color: white;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        font-size: 20px;
        opacity: 0;
        transition: opacity 0.3s;
        z-index: 1000;
    `;
    
    document.body.appendChild(button);
    
    // 滚动时显示/隐藏按钮
    window.addEventListener('scroll', function() {
        if (window.pageYOffset > 300) {
            button.style.opacity = '1';
        } else {
            button.style.opacity = '0';
        }
    });
    
    // 点击返回顶部
    button.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// 添加阅读进度条
function addReadingProgress() {
    const progressBar = document.createElement('div');
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 0%;
        height: 3px;
        background: linear-gradient(90deg, #007acc, #00d4aa);
        z-index: 1001;
        transition: width 0.1s;
    `;
    
    document.body.appendChild(progressBar);
    
    // 更新进度条
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrollPercent = (scrollTop / docHeight) * 100;
        
        progressBar.style.width = scrollPercent + '%';
    });
}

// 添加搜索高亮功能
function highlightSearchTerms() {
    const urlParams = new URLSearchParams(window.location.search);
    const searchQuery = urlParams.get('q');
    
    if (searchQuery) {
        const content = document.querySelector('.md-content');
        const regex = new RegExp(`(${searchQuery})`, 'gi');
        
        content.innerHTML = content.innerHTML.replace(regex, 
            '<mark style="background-color: yellow; padding: 2px;">$1</mark>');
    }
}

// 页面可见性变化时的处理
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // 页面隐藏时的处理
        console.log('页面已隐藏');
    } else {
        // 页面显示时的处理
        console.log('页面已显示');
    }
});

// 键盘快捷键
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + K: 聚焦搜索框
    if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
        event.preventDefault();
        const searchInput = document.querySelector('.md-search__input');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Ctrl/Cmd + /: 切换侧边栏
    if ((event.ctrlKey || event.metaKey) && event.key === '/') {
        event.preventDefault();
        const sidebarToggle = document.querySelector('.md-nav__button');
        if (sidebarToggle) {
            sidebarToggle.click();
        }
    }
}); 