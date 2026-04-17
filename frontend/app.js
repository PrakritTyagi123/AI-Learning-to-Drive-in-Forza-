/*
 * ForzaTek AI — Shared JavaScript
 * =================================
 * Loaded by every page. Provides:
 *   window.ForzaTek.mount(pageId)
 *
 * Responsibilities:
 *   1. Inject the sidebar into the DOM
 *   2. Highlight the active page
 *   3. Toggle between expanded / collapsed state (persists in localStorage)
 *   4. Poll /api/system/health every 3 s and show connection status
 *   5. Provide common helpers (api, fmt, conf class)
 */

(function () {
  const NAV = [
    { section: 'Overview' },
    { id: 'dashboard', label: 'Dashboard',  href: '/',           icon: 'grid' },

    { section: 'Data' },
    { id: 'record',    label: 'Record',     href: '/record',     icon: 'circle' },
    { id: 'ingest',    label: 'Ingest',     href: '/ingest',     icon: 'download' },
    { id: 'label',     label: 'Label',      href: '/label',      icon: 'pen' },

    { section: 'Model' },
    { id: 'train',     label: 'Train',      href: '/train',      icon: 'cpu' },

    { section: 'Runtime' },
    { id: 'telemetry', label: 'Telemetry',  href: '/telemetry',  icon: 'gauge' },
    { id: 'drive',     label: 'Drive',      href: '/drive',      icon: 'steering' },

    { section: 'System' },
    { id: 'settings',  label: 'Settings',   href: '/settings',   icon: 'sliders' },
    { id: 'help',      label: 'Help',       href: '/help',       icon: 'book' },
  ];

  const ICONS = {
    grid: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>',
    circle: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>',
    download: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 4v12M6 12l6 6 6-6M4 20h16"/></svg>',
    pen: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 20l4-1 10-10-3-3L5 16l-1 4zM13 6l3 3"/></svg>',
    cpu: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="6" y="6" width="12" height="12"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/></svg>',
    gauge: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 16a8 8 0 1116 0"/><path d="M12 16l4-6"/><circle cx="12" cy="16" r="1" fill="currentColor"/></svg>',
    steering: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="2"/><path d="M12 3v7M3.5 8.5L10 12M20.5 8.5L14 12M7 20l4-6M17 20l-4-6"/></svg>',
    sliders: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h14M18 18h2"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="16" cy="18" r="2"/></svg>',
    book: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 5a2 2 0 012-2h12v16H6a2 2 0 00-2 2V5z"/><path d="M4 19h14"/></svg>',
    collapse: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 6l-6 6 6 6"/></svg>',
    expand:   '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 6l6 6-6 6"/></svg>',
  };

  function renderSidebar(activeId, collapsed) {
    const items = NAV.map(item => {
      if (item.section) {
        return `<div class="sidebar-section">${item.section}</div>`;
      }
      const cls = item.id === activeId ? 'sidebar-link active' : 'sidebar-link';
      return `
        <a href="${item.href}" class="${cls}" title="${item.label}">
          ${ICONS[item.icon] || ''}
          <span class="lbl">${item.label}</span>
        </a>
      `;
    }).join('');

    return `
      <aside class="sidebar">
        <div class="sidebar-brand">
          <div class="logo">FT</div>
          <div class="title">
            <div class="n">ForzaTek</div>
            <div class="v">v1.0</div>
          </div>
        </div>
        <nav class="sidebar-nav">${items}</nav>
        <div class="sidebar-foot">
          <div class="sidebar-conn" id="ftk-conn">
            <span class="dot"></span>
            <span class="lbl">OFFLINE</span>
          </div>
          <button class="sidebar-toggle" id="ftk-toggle" title="Toggle sidebar">
            ${collapsed ? ICONS.expand : ICONS.collapse}
          </button>
        </div>
      </aside>
    `;
  }

  async function pollHealth() {
    const el = document.getElementById('ftk-conn');
    if (!el) return;
    try {
      const r = await fetch('/api/system/health', { cache: 'no-store' });
      const ok = r.ok;
      el.classList.toggle('ok', ok);
      el.querySelector('.lbl').textContent = ok ? 'ONLINE' : 'OFFLINE';
    } catch {
      el.classList.remove('ok');
      el.querySelector('.lbl').textContent = 'OFFLINE';
    }
  }

  // ── Public helpers ──
  const ForzaTek = {
    mount(activeId) {
      const collapsed = localStorage.getItem('ftk:collapsed') === '1';
      document.body.classList.add('with-sidebar');
      if (collapsed) document.body.classList.add('sidebar-collapsed');

      const wrap = document.createElement('div');
      wrap.innerHTML = renderSidebar(activeId, collapsed);
      document.body.insertBefore(wrap.firstElementChild, document.body.firstChild);

      document.getElementById('ftk-toggle').addEventListener('click', () => {
        const now = !document.body.classList.contains('sidebar-collapsed');
        document.body.classList.toggle('sidebar-collapsed', now);
        localStorage.setItem('ftk:collapsed', now ? '1' : '0');
        document.getElementById('ftk-toggle').innerHTML = now ? ICONS.expand : ICONS.collapse;
      });

      pollHealth();
      setInterval(pollHealth, 3000);
    },

    // API helpers
    api: {
      async get(path) {
        const r = await fetch(path);
        if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
        return r.json();
      },
      async post(path, body) {
        const r = await fetch(path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body || {}),
        });
        if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
        return r.json();
      },
    },

    // Format helpers
    fmt: {
      n(v, digits = 0) {
        if (v == null || isNaN(v)) return '—';
        return Number(v).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
      },
      pct(v, digits = 0) {
        if (v == null || isNaN(v)) return '—';
        return (v * 100).toFixed(digits) + '%';
      },
      bytes(b) {
        if (!b) return '0 B';
        const u = ['B', 'KB', 'MB', 'GB', 'TB'];
        let i = 0;
        while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
        return b.toFixed(i === 0 ? 0 : 1) + ' ' + u[i];
      },
      sec(s) {
        if (s == null || isNaN(s)) return '—';
        s = Math.floor(s);
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        const ss = s % 60;
        if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
        return `${m}:${String(ss).padStart(2, '0')}`;
      },
    },

    // Confidence color class
    confClass(c) {
      if (c == null) return '';
      if (c >= 0.80) return 'ok';
      if (c >= 0.50) return 'warn';
      return 'danger';
    },
  };

  window.ForzaTek = ForzaTek;
})();
