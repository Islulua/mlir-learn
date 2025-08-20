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

// 额外的JavaScript功能

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化代码高亮
    initCodeHighlighting();
    
    // 初始化搜索增强
    initSearchEnhancement();
    
    // 初始化导航增强
    initNavigationEnhancement();
    
    // 初始化响应式功能
    initResponsiveFeatures();
});

// 代码高亮功能
function initCodeHighlighting() {
    // 为MLIR代码块添加特殊样式
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        const text = block.textContent;
        if (text.includes('%') || text.includes('=') || text.includes(':')) {
            block.classList.add('mlir-code');
        }
    });
    
    // 添加复制按钮
    addCopyButtons();
}

// 添加复制按钮
function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre');
    codeBlocks.forEach(block => {
        // 检查是否已经有复制按钮
        if (block.querySelector('.copy-button')) {
            return;
        }
        
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button md-button';
        copyButton.textContent = '复制';
        copyButton.style.position = 'absolute';
        copyButton.style.top = '0.5em';
        copyButton.style.right = '0.5em';
        copyButton.style.fontSize = '0.8em';
        copyButton.style.padding = '0.2em 0.5em';
        
        copyButton.addEventListener('click', function() {
            const code = block.querySelector('code');
            if (code) {
                navigator.clipboard.writeText(code.textContent).then(() => {
                    copyButton.textContent = '已复制!';
                    setTimeout(() => {
                        copyButton.textContent = '复制';
                    }, 2000);
                }).catch(err => {
                    console.error('复制失败:', err);
                    copyButton.textContent = '复制失败';
                    setTimeout(() => {
                        copyButton.textContent = '复制';
                    }, 2000);
                });
            }
        });
        
        // 设置代码块的相对定位
        block.style.position = 'relative';
        block.appendChild(copyButton);
    });
}

// 搜索增强功能
function initSearchEnhancement() {
    const searchInput = document.querySelector('.md-search__input');
    if (searchInput) {
        // 添加搜索建议
        searchInput.addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            if (query.length > 2) {
                showSearchSuggestions(query);
            } else {
                hideSearchSuggestions();
            }
        });
        
        // 添加搜索历史
        loadSearchHistory();
    }
}

// 显示搜索建议
function showSearchSuggestions(query) {
    // 这里可以实现搜索建议逻辑
    // 例如从页面内容中搜索匹配的标题和关键词
    const suggestions = searchInContent(query);
    displaySearchSuggestions(suggestions);
}

// 在页面内容中搜索
function searchInContent(query) {
    const suggestions = [];
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    
    headings.forEach(heading => {
        const text = heading.textContent.toLowerCase();
        if (text.includes(query)) {
            suggestions.push({
                text: heading.textContent,
                url: '#' + heading.id,
                level: parseInt(heading.tagName.charAt(1))
            });
        }
    });
    
    return suggestions.slice(0, 5); // 限制建议数量
}

// 显示搜索建议
function displaySearchSuggestions(suggestions) {
    let suggestionBox = document.getElementById('search-suggestions');
    if (!suggestionBox) {
        suggestionBox = document.createElement('div');
        suggestionBox.id = 'search-suggestions';
        suggestionBox.className = 'search-suggestions';
        suggestionBox.style.cssText = `
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 4px 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            z-index: 1000;
            max-height: 300px;
            overflow-y: auto;
        `;
        
        const searchContainer = document.querySelector('.md-search');
        if (searchContainer) {
            searchContainer.style.position = 'relative';
            searchContainer.appendChild(suggestionBox);
        }
    }
    
    if (suggestions.length === 0) {
        suggestionBox.style.display = 'none';
        return;
    }
    
    suggestionBox.innerHTML = suggestions.map(suggestion => `
        <div class="search-suggestion-item" style="
            padding: 0.5em 1em;
            border-bottom: 1px solid #eee;
            cursor: pointer;
            font-size: 0.9em;
        ">
            <div style="font-weight: bold;">${suggestion.text}</div>
            <div style="color: #666; font-size: 0.8em;">${'#'.repeat(suggestion.level)} 标题</div>
        </div>
    `).join('');
    
    suggestionBox.style.display = 'block';
    
    // 添加点击事件
    suggestionBox.querySelectorAll('.search-suggestion-item').forEach((item, index) => {
        item.addEventListener('click', () => {
            const suggestion = suggestions[index];
            if (suggestion.url.startsWith('#')) {
                const element = document.querySelector(suggestion.url);
                if (element) {
                    element.scrollIntoView({ behavior: 'smooth' });
                }
            }
            hideSearchSuggestions();
        });
    });
}

// 隐藏搜索建议
function hideSearchSuggestions() {
    const suggestionBox = document.getElementById('search-suggestions');
    if (suggestionBox) {
        suggestionBox.style.display = 'none';
    }
}

// 加载搜索历史
function loadSearchHistory() {
    const history = JSON.parse(localStorage.getItem('mlir-search-history') || '[]');
    // 这里可以实现搜索历史显示逻辑
}

// 保存搜索历史
function saveSearchHistory(query) {
    let history = JSON.parse(localStorage.getItem('mlir-search-history') || '[]');
    history = history.filter(item => item !== query);
    history.unshift(query);
    history = history.slice(0, 10); // 只保留最近10条
    localStorage.setItem('mlir-search-history', JSON.stringify(history));
}

// 导航增强功能
function initNavigationEnhancement() {
    // 添加面包屑导航
    addBreadcrumbNavigation();
    
    // 添加页面目录
    addTableOfContents();
    
    // 添加返回顶部按钮
    addBackToTopButton();
}

// 添加面包屑导航
function addBreadcrumbNavigation() {
    const breadcrumb = document.createElement('nav');
    breadcrumb.className = 'breadcrumb-navigation';
    breadcrumb.style.cssText = `
        padding: 0.5em 0;
        margin-bottom: 1em;
        font-size: 0.9em;
        color: #666;
    `;
    
    const path = window.location.pathname;
    const segments = path.split('/').filter(segment => segment);
    
    let breadcrumbHTML = '<a href="/">首页</a>';
    let currentPath = '';
    
    segments.forEach((segment, index) => {
        currentPath += '/' + segment;
        const displayName = segment.replace(/-/g, ' ').replace(/_/g, ' ');
        breadcrumbHTML += ` > <a href="${currentPath}">${displayName}</a>`;
    });
    
    breadcrumb.innerHTML = breadcrumbHTML;
    
    const content = document.querySelector('.md-content');
    if (content) {
        content.insertBefore(breadcrumb, content.firstChild);
    }
}

// 添加页面目录
function addTableOfContents() {
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    if (headings.length < 3) return; // 如果标题太少，不显示目录
    
    const toc = document.createElement('div');
    toc.className = 'table-of-contents';
    toc.style.cssText = `
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 1em;
        margin: 1em 0;
        font-size: 0.9em;
    `;
    
    toc.innerHTML = '<h4 style="margin-top: 0;">目录</h4>';
    
    const tocList = document.createElement('ul');
    tocList.style.cssText = 'list-style: none; padding-left: 0; margin: 0;';
    
    headings.forEach((heading, index) => {
        if (!heading.id) {
            heading.id = `heading-${index}`;
        }
        
        const level = parseInt(heading.tagName.charAt(1));
        const indent = (level - 1) * 20;
        
        const listItem = document.createElement('li');
        listItem.style.cssText = `margin: 0.3em 0; padding-left: ${indent}px;`;
        
        const link = document.createElement('a');
        link.href = '#' + heading.id;
        link.textContent = heading.textContent;
        link.style.cssText = 'text-decoration: none; color: #333;';
        
        link.addEventListener('click', (e) => {
            e.preventDefault();
            heading.scrollIntoView({ behavior: 'smooth' });
        });
        
        listItem.appendChild(link);
        tocList.appendChild(listItem);
    });
    
    toc.appendChild(tocList);
    
    const content = document.querySelector('.md-content');
    if (content) {
        content.insertBefore(toc, content.firstChild);
    }
}

// 添加返回顶部按钮
function addBackToTopButton() {
    const backToTop = document.createElement('button');
    backToTop.className = 'back-to-top md-button';
    backToTop.textContent = '↑ 返回顶部';
    backToTop.style.cssText = `
        position: fixed;
        bottom: 2em;
        right: 2em;
        z-index: 1000;
        display: none;
        opacity: 0.8;
    `;
    
    backToTop.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    
    document.body.appendChild(backToTop);
    
    // 监听滚动事件
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            backToTop.style.display = 'block';
        } else {
            backToTop.style.display = 'none';
        }
    });
}

// 响应式功能
function initResponsiveFeatures() {
    // 移动端菜单优化
    optimizeMobileMenu();
    
    // 响应式表格
    makeTablesResponsive();
    
    // 触摸手势支持
    addTouchSupport();
}

// 移动端菜单优化
function optimizeMobileMenu() {
    const navToggle = document.querySelector('.md-nav__toggle');
    if (navToggle) {
        navToggle.addEventListener('click', function() {
            const nav = this.nextElementSibling;
            if (nav) {
                nav.style.display = nav.style.display === 'none' ? 'block' : 'none';
            }
        });
    }
}

// 响应式表格
function makeTablesResponsive() {
    const tables = document.querySelectorAll('table');
    tables.forEach(table => {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'overflow-x: auto; margin: 1em 0;';
        wrapper.className = 'table-wrapper';
        
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
    });
}

// 触摸手势支持
function addTouchSupport() {
    let startX = 0;
    let startY = 0;
    
    document.addEventListener('touchstart', function(e) {
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
    });
    
    document.addEventListener('touchend', function(e) {
        const endX = e.changedTouches[0].clientX;
        const endY = e.changedTouches[0].clientY;
        
        const diffX = startX - endX;
        const diffY = startY - endY;
        
        // 检测滑动方向
        if (Math.abs(diffX) > Math.abs(diffY) && Math.abs(diffX) > 50) {
            if (diffX > 0) {
                // 向左滑动
                console.log('向左滑动');
            } else {
                // 向右滑动
                console.log('向右滑动');
            }
        }
    });
}

// 工具函数
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// 导出函数供外部使用
window.MLIRDocs = {
    initCodeHighlighting,
    addCopyButtons,
    showSearchSuggestions,
    addBreadcrumbNavigation,
    addTableOfContents,
    addBackToTopButton,
    debounce,
    throttle
}; 