/* ============================================
   学屿 · 公共 JS 层
   - 应用壳渲染（顶栏 + 侧栏，真实数据）
   - API 封装 + 友好错误
   - ⌘K 全局搜索 / ⌘N 新建 / ⌘F 页内搜索
   - 轻量 Markdown 渲染（marked + DOMPurify + KaTeX）
   - 应用内通知
   ============================================ */
'use strict';

/* ---------- 基础工具 ---------- */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtSec = s => { s = Math.max(0, Math.round(s)); return String(Math.floor(s/60)).padStart(2,'0') + ':' + String(s%60).padStart(2,'0'); };
const fmtHM = s => { s = Math.round(s||0); const h = Math.floor(s/3600), m = Math.round((s%3600)/60); return h ? `${h} 小时 ${m} 分` : `${m} 分钟`; };
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const humanSize = n => n < 1024 ? n + ' B' : n < 1048576 ? (n/1024).toFixed(1) + ' KB' : (n/1048576).toFixed(1) + ' MB';
const ago = iso => {
  if (!iso) return '';
  const d = new Date(iso); if (isNaN(d)) return iso;
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return '刚刚';
  if (diff < 3600) return Math.floor(diff/60) + ' 分钟前';
  if (diff < 86400) return Math.floor(diff/3600) + ' 小时前';
  if (diff < 86400*7) return Math.floor(diff/86400) + ' 天前';
  return d.toLocaleDateString('zh-CN', {month:'numeric', day:'numeric'});
};

async function api(path, opts = {}) {
  const res = await fetch('/api' + path, {
    headers: {'Content-Type': 'application/json'},
    ...opts,
    body: opts.body != null ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = `请求失败（${res.status}）`;
    try { const j = await res.json(); if (j.detail) detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch {}
    const err = new Error(detail); err.status = res.status;
    if (res.status === 400 && detail.includes('设置')) err.goSettings = true;
    throw err;
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('json') ? res.json() : res.text();
}

/* ---------- 轻量 Markdown（含公式保护：marked 不会弄脏 LaTeX） ---------- */
function protectMath(text) {
  const store = [];
  let t = String(text || '');
  const sub = re => { t = t.replace(re, m => { store.push(m); return `%%MATH${store.length - 1}%%`; }); };
  sub(/\$\$[\s\S]*?\$\$/g);      // $$块$
  sub(/\\\[[\s\S]*?\\\]/g);   // \[...\]
  sub(/\\\([\s\S]*?\\\)/g);   // \(...\)
  sub(/\$[^$\n]+\$/g);            // $行内
  return {text: t, store};
}
function restoreMath(html, store) {
  return html.replace(/%%MATH(\d+)%%/g, (m, i) => store[+i] !== undefined ? store[+i] : m);
}
function md(text) {
  if (!window.marked) return esc(text);
  const {text: safe, store} = protectMath(text);
  let html = marked.parse(safe, {breaks: true, gfm: true});
  html = restoreMath(html, store);
  return window.DOMPurify ? DOMPurify.sanitize(html, {ADD_ATTR: ['target']}) : html;
}
function mdRender(el, text) {
  el.innerHTML = md(text);
  if (window.renderMathInElement) {
    try { renderMathInElement(el, {delimiters: [
      {left:'$$',right:'$$',display:true},{left:'\\[',right:'\\]',display:true},
      {left:'$',right:'$',display:false},{left:'\\(',right:'\\)',display:false}]});
    } catch {}
  }
  $$('#' + el.id + ' a, a', el).forEach(a => { if (!a.href.startsWith('#')) { a.target = '_blank'; a.rel = 'noopener'; } });
}

/* ---------- 应用内通知 ---------- */
function toast(title, body = '', kind = '', onclick = null, ms = 5000) {
  let stack = $('.toast-stack');
  if (!stack) { stack = document.createElement('div'); stack.className = 'toast-stack'; document.body.appendChild(stack); }
  const t = document.createElement('div');
  t.className = 'toast ' + kind;
  t.innerHTML = `<div style="font-size:18px">${kind === 'err' ? '⚠️' : kind === 'ok' ? '✅' : '🏝️'}</div>
    <div style="flex:1"><b>${esc(title)}</b>${body ? `<p>${esc(body)}</p>` : ''}</div>`;
  t.onclick = () => { t.remove(); onclick && onclick(); };
  stack.appendChild(t);
  if (ms) setTimeout(() => t.remove(), ms);
  return t;
}
const showError = e => toast('出错了', e.message || String(e), 'err', e.goSettings ? () => location.href = '/settings' : null, 7000);
window.addEventListener('unhandledrejection', ev => console.warn(ev.reason));

/* ---------- 应用壳（顶栏 + 侧栏） ---------- */
const SHELL = {
  active: document.body.dataset.page || '',
  subtitle: document.body.dataset.sub || '',
  tools: document.body.dataset.tools === '1',   // 页面自带额外工具（插到 #page-tools）
  overview: null,
};

function renderShell(ov) {
  // 静态壳已内嵌于各页面 HTML —— 这里只更新数据，绝不重建 DOM（避免导航残影/闪烁）
  SHELL.overview = ov;
  const setCnt = (id, html) => {
    const el = document.querySelector(id);
    if (el) el.innerHTML = html;
    else return;
  };
  // 收起状态持久化：切页/重启后保持
  if (localStorage.getItem('xueyu-no-nav') === '1') document.body.classList.add('no-nav');
  if (localStorage.getItem('xueyu-no-ai') === '1') document.body.classList.add('no-ai');
  if ($('#app-sidebar')) {
    setCnt('#cnt-today', ov.today ? (ov.today.total - ov.today.done || '') : '');
    setCnt('#cnt-plans', ov.stats.active_plans || '');
    setCnt('#cnt-notes', ov.notes_count || '');
    setCnt('#cnt-shelf', ov.docs_count || '');
    setCnt('#cnt-research', ov.research_running.length ? `<span class="live"></span>${ov.research_running.length}` : '');
    // AI 未配置：设置项加提示点
    const settingsLink = document.querySelector('a[data-nav="settings"]');
    if (settingsLink) {
      const dot = settingsLink.querySelector('.ai-dot');
      if (!ov.ai_configured && !dot) settingsLink.insertAdjacentHTML('beforeend', ' <span class="ai-dot" title="AI 模型未配置" style="width:7px;height:7px;border-radius:50%;background:var(--org);display:inline-block"></span>');
      if (ov.ai_configured && dot) dot.remove();
    }
    const scb = $('#side-collapse');
    if (scb && !scb._bound) {
      scb.onclick = () => {
        const rail = document.body.classList.toggle('no-nav');
        scb.textContent = rail ? '⤢' : '◧';
        scb.title = rail ? '展开侧边栏' : '收起侧边栏';
        localStorage.setItem('xueyu-no-nav', rail ? '1' : '0');
      };
      scb._bound = true;
    }
  }
  const gs = $('#global-search');
  if (gs && !gs._bound) { gs.onclick = () => openCmdK(); gs._bound = true; }
  renderPageTools();
}

/* 页面通过 XueYu.onPageTools() 声明自己的工具区事件 */
const _pageToolBinders = [];
function onPageTools(fn) { _pageToolBinders.push(fn); }
function renderPageTools() { _pageToolBinders.forEach(fn => { try { fn(); } catch {} }); }


async function bootShell() {
  if (!$('#app-header')) return;  // 复习模式等全屏页可没有 header
  try {
    renderShell(await api('/overview'));
    SHELL._readyResolve && SHELL._readyResolve();
  } catch (e) {
    document.body.innerHTML = `<div style="padding:60px;text-align:center;font-weight:800">
      😔 无法连接本地服务<br><small style="color:#a29880">${esc(e.message)}</small></div>`;
    SHELL._readyReject && SHELL._readyReject(e);
  }
}
/* 页面等 shell 数据就绪后再渲染主区 */
SHELL.ready = new Promise((res, rej) => { SHELL._readyResolve = res; SHELL._readyReject = rej; });

/* 新建：上下文感知 ⌘N */
function newInContext() {
  const page = SHELL.active;
  if (page === 'notes') return window.newNote && window.newNote();
  if (page === 'plan') return window.newPlanDialog && window.newPlanDialog();
  if (page === 'research') return $('#research-input')?.focus();
  location.href = '/notes?new=1';
}

/* ---------- ⌘K 全局搜索 ---------- */
let cmdkSel = 0;
async function openCmdK(initial = '') {
  let ov = $('#cmdk');
  if (!ov) {
    ov = document.createElement('div');
    ov.id = 'cmdk'; ov.className = 'overlay';
    ov.innerHTML = `<div class="modal">
      <input class="cmd-input" id="cmdk-input" placeholder="搜索笔记、文档、报告、计划…">
      <div class="modal-body" id="cmdk-results"></div>
      <div class="modal-foot"><span style="font-size:11px;color:var(--dim);font-weight:700">↑↓ 选择 · 回车打开 · Esc 关闭</span></div></div>`;
    document.body.appendChild(ov);
    ov.onclick = e => { if (e.target === ov) closeCmdK(); };
    $('#cmdk-input', ov).addEventListener('input', debounce(cmdkSearch, 180));
    $('#cmdk-input', ov).addEventListener('keydown', e => {
      const items = $$('#cmdk-results .res-it');
      if (e.key === 'ArrowDown') { e.preventDefault(); cmdkSel = Math.min(cmdkSel + 1, items.length - 1); markSel(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); cmdkSel = Math.max(cmdkSel - 1, 0); markSel(); }
      else if (e.key === 'Enter') { items[cmdkSel]?.click(); }
      else if (e.key === 'Escape') closeCmdK();
    });
  }
  ov.classList.add('show');
  const input = $('#cmdk-input'); input.value = initial; input.focus();
  cmdkSearch();
}
function closeCmdK() { $('#cmdk')?.classList.remove('show'); }
function markSel() { $$('#cmdk-results .res-it').forEach((el, i) => el.classList.toggle('sel', i === cmdkSel)); }
async function cmdkSearch() {
  const q = $('#cmdk-input').value.trim();
  const box = $('#cmdk-results');
  if (!q) { box.innerHTML = `<div class="empty"><span class="big">🔎</span>输入关键词，搜全部笔记 / 书架 / 报告 / 计划</div>`; return; }
  box.innerHTML = `<div class="empty">搜索中…</div>`;
  let r; try { r = await api('/search?q=' + encodeURIComponent(q)); } catch (e) { box.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  const groups = [
    ['📝 笔记', r.notes.map(n => ({icon: '📝', title: n.title, sub: n.snippet, href: `/notes?id=${n.id}`}))],
    ['📚 书架', r.docs.map(d => ({icon: d.fmt === 'pdf' ? '📕' : d.fmt === 'docx' ? '📘' : d.fmt === 'pptx' ? '📙' : '📗', title: d.title, sub: d.fmt.toUpperCase(), href: `/doc?id=${d.id}`}))],
    ['📰 报告', r.reports.map(t => ({icon: '📰', title: t.title, sub: '研究报告', href: `/report?id=${t.id}`}))],
    ['📅 计划', r.plans.map(p => ({icon: '📅', title: p.title, sub: p.stage_label, href: `/plan?item=${p.id}`}))],
  ].filter(g => g[1].length);
  if (!groups.length) { box.innerHTML = `<div class="empty"><span class="big">🫙</span>没有找到「${esc(q)}」相关内容</div>`; return; }
  cmdkSel = 0;
  box.innerHTML = groups.map(([label, items]) =>
    `<div class="side-label" style="padding:6px 12px 2px">${label}</div>` +
    items.map(x => `<div class="res-it" data-href="${x.href}"><span>${x.icon}</span><b style="white-space:nowrap">${esc(x.title)}</b><span class="snippet">${esc(x.sub || '')}</span></div>`).join('')
  ).join('');
  $$('#cmdk-results .res-it').forEach(el => el.onclick = () => { closeCmdK(); location.href = el.dataset.href; });
  markSel();
}

/* ---------- 通用选择器（选笔记 / 选文档 / 确认框） ---------- */
function pickDialog(title, items, {multi = false, onOk, preselected = []} = {}) {
  return new Promise(resolve => {
    const ov = document.createElement('div');
    ov.className = 'overlay show';
    const selected = new Set(preselected.map(String));
    const render = () => {
      ov.innerHTML = `<div class="modal">
        <div class="modal-head">${esc(title)}<span class="x" data-act="cancel">✕</span></div>
        <div class="modal-body">${items.length ? items.map(it => `
          <div class="pick-it ${selected.has(String(it.id)) ? 'on' : ''}" data-id="${esc(it.id)}">
            <span>${it.icon || '📄'}</span><span style="flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(it.title)}</span>
            <small>${esc(it.sub || '')}${multi && selected.has(String(it.id)) ? ' · ✓' : ''}</small></div>`).join('')
          : '<div class="empty"><span class="big">🍃</span>这里还没有内容</div>'}</div>
        <div class="modal-foot">${multi && selected.size ? `<span style="margin-right:auto;font-size:11.5px;color:var(--dim);font-weight:800">已选 ${selected.size} 项（再次点击可取消）</span>` : ''}
          <button class="btn sm ghost" data-act="cancel">取消</button>
          <button class="btn sm" data-act="ok">确定</button></div></div>`;
      $$('.pick-it', ov).forEach(el => el.onclick = () => {
        const id = el.dataset.id;
        if (multi) { selected.has(id) ? selected.delete(id) : selected.add(id); render(); }
        else { cleanup(); resolve(items.find(i => String(i.id) === id)); }
      });
      $$('[data-act]', ov).forEach(b => b.onclick = () => {
        cleanup();
        resolve(b.dataset.act === 'ok' ? items.filter(i => selected.has(String(i.id))) : null);
      });
    };
    const cleanup = () => ov.remove();
    document.body.appendChild(ov);
    render();
    ov._onOk = onOk;
  });
}

function promptDialog(title, {label = "", value = "", placeholder = "", multiline = false} = {}) {
  return new Promise(resolve => {
    const ov = document.createElement('div');
    ov.className = 'overlay show';
    ov.innerHTML = `<div class="modal" style="width:min(440px,92vw)">
      <div class="modal-head">${esc(title)}</div>
      <div style="padding:14px 18px">
        ${label ? `<div style="font-size:12px;font-weight:800;color:var(--dim);margin-bottom:6px">${esc(label)}</div>` : ''}
        ${multiline
          ? `<textarea class="input" id="pd-input" rows="4" placeholder="${esc(placeholder || '')}" style="resize:vertical">${esc(value)}</textarea>`
          : `<input class="input" id="pd-input" value="${esc(value)}" placeholder="${esc(placeholder || '')}">`}
      </div>
      <div class="modal-foot"><button class="btn sm ghost" data-r="0">取消</button><button class="btn sm" data-r="1">确定</button></div></div>`;
    document.body.appendChild(ov);
    const input = $('#pd-input', ov);
    input.focus(); input.select();
    const done = ok => { const v = input.value.trim(); ov.remove(); resolve(ok ? v : null); };
    $$('[data-r]', ov).forEach(b => b.onclick = () => done(b.dataset.r === '1'));
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && (!multiline || e.metaKey || e.ctrlKey)) { e.preventDefault(); done(true); }
      if (e.key === 'Escape') done(false);
    });
    ov.onclick = e => { if (e.target === ov) done(false); };
  });
}

function confirmDialog(title, text) {
  return new Promise(resolve => {
    const ov = document.createElement('div');
    ov.className = 'overlay show';
    ov.innerHTML = `<div class="modal" style="width:min(420px,92vw)">
      <div class="modal-head">${esc(title)}</div>
      <div style="padding:16px 18px;font-size:13px;color:#6b6353;font-weight:600">${esc(text)}</div>
      <div class="modal-foot"><button class="btn sm ghost" data-r="0">取消</button><button class="btn sm" data-r="1">确定</button></div></div>`;
    document.body.appendChild(ov);
    $$('[data-r]', ov).forEach(b => b.onclick = () => { ov.remove(); resolve(b.dataset.r === '1'); });
  });
}

/* ---------- 全局快捷键 ---------- */
document.addEventListener('keydown', e => {
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); openCmdK(); }
  else if (mod && e.key.toLowerCase() === 'n') { e.preventDefault(); newInContext(); }
  else if (mod && e.key.toLowerCase() === 'f') {
    if (window.pageSearch) { e.preventDefault(); window.pageSearch(); }
  } else if (e.key === 'Escape') { closeCmdK(); }
});

/* ---------- 思维导图渲染：markmap（自动布局/缩放/拖拽/折叠），兼容旧树数据 ---------- */
function treeToMarkdown(t) {
  const lines = ['# ' + (t.label || '思维导图')];
  const walk = (n, depth) => {
    (n.children || []).forEach(c => {
      lines.push('  '.repeat(depth) + '- ' + (c.label || ''));
      walk(c, depth + 1);
    });
  };
  walk(t, 1);
  return lines.join('\n');
}
function renderMindmap(container, data) {
  const mdText = typeof data === 'string' ? data
    : data && data.markdown ? data.markdown
    : treeToMarkdown(data);
  container.innerHTML = '<svg style="width:100%;height:560px"></svg>' +
    '<div style="font-size:10.5px;color:var(--dim);font-weight:700;text-align:center;margin-top:2px">滚轮缩放 · 拖拽平移 · 点节点折叠</div>';
  const svg = container.querySelector('svg');
  try {
    const { root } = new markmap.Transformer().transform(mdText);
    markmap.Markmap.create(svg, {fitRatio: 0.9, spacingVertical: 6, paddingX: 14, initialExpandLevel: 3}, root);
  } catch (e) {
    container.innerHTML = '<div class="empty">思维导图渲染失败：' + esc(e.message) + '</div>';
  }
}

/* ---------- 启动 ---------- */
window.XueYu = {$, $$, esc, api, md, mdRender, toast, showError, fmtSec, fmtHM, debounce, ago, humanSize, pickDialog, confirmDialog, promptDialog, renderMindmap, SHELL, onPageTools};
document.addEventListener('DOMContentLoaded', bootShell);
