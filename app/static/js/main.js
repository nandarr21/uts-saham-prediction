/* FILE: app/static/js/main.js */

/* ── NAVBAR: tambah shadow saat scroll ────────────────────── */
window.addEventListener('scroll', () => {
  const nav = document.getElementById('mainNav');
  if (!nav) return;
  if (window.scrollY > 20) {
    nav.style.boxShadow = '0 4px 24px rgba(0,0,0,0.4)';
  } else {
    nav.style.boxShadow = 'none';
  }
});

/* ── FORMAT angka ribuan ──────────────────────────────────── */
function formatNumber(num, decimals = 2) {
  if (num === null || num === undefined || isNaN(num)) return '—';
  return parseFloat(num).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

/* ── Copy teks ke clipboard ───────────────────────────────── */
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast('Disalin ke clipboard!');
  }).catch(() => {
    showToast('Gagal menyalin.', 'danger');
  });
}

/* ── Toast notification ───────────────────────────────────── */
function showToast(message, type = 'success') {
  // Hapus toast lama jika ada
  const existing = document.getElementById('globalToast');
  if (existing) existing.remove();

  const colors = {
    success : { bg: 'rgba(16,185,129,0.15)',
                border: 'rgba(16,185,129,0.3)',
                text: '#6EE7B7' },
    danger  : { bg: 'rgba(239,68,68,0.15)',
                border: 'rgba(239,68,68,0.3)',
                text: '#FCA5A5' },
    warning : { bg: 'rgba(245,158,11,0.15)',
                border: 'rgba(245,158,11,0.3)',
                text: '#FCD34D' }
  };

  const c   = colors[type] || colors.success;
  const div = document.createElement('div');
  div.id    = 'globalToast';
  div.style.cssText = `
    position: fixed;
    bottom: 1.5rem;
    right : 1.5rem;
    z-index: 9999;
    padding: 0.75rem 1.25rem;
    border-radius: 10px;
    background: ${c.bg};
    border: 1px solid ${c.border};
    color: ${c.text};
    font-size: 0.875rem;
    font-weight: 500;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    animation: slideInToast 0.3s ease;
    max-width: 300px;
  `;
  div.textContent = message;
  document.body.appendChild(div);

  setTimeout(() => {
    div.style.opacity   = '0';
    div.style.transform = 'translateY(10px)';
    div.style.transition = 'all 0.3s ease';
    setTimeout(() => div.remove(), 300);
  }, 2800);
}

/* ── Animasi counter angka ────────────────────────────────── */
function animateCounter(el, from, to, duration = 1000) {
  const start = performance.now();
  const update = (time) => {
    const progress = Math.min((time - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3); // easeOutCubic
    el.textContent = Math.round(from + (to - from) * ease).toLocaleString();
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

/* ── Jalankan counter saat elemen terlihat ────────────────── */
function initCounters() {
  const counters = document.querySelectorAll('[data-counter]');
  if (!counters.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el  = entry.target;
        const to  = parseInt(el.dataset.counter, 10);
        animateCounter(el, 0, to, 1200);
        observer.unobserve(el);
      }
    });
  }, { threshold: 0.5 });

  counters.forEach(el => observer.observe(el));
}

/* ── Chart.js global defaults ─────────────────────────────── */
function setChartDefaults() {
  if (typeof Chart === 'undefined') return;
  Chart.defaults.color              = '#64748B';
  Chart.defaults.font.family        = 'Inter, sans-serif';
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.padding       = 16;
  Chart.defaults.animation.duration = 800;
  Chart.defaults.responsive         = true;
  Chart.defaults.maintainAspectRatio = true;
}

/* ── Highlight baris tabel terbaik ───────────────────────── */
function highlightBestRow() {
  const table = document.getElementById('comparisonTable');
  if (!table) return;

  const rows = table.querySelectorAll('tbody tr');
  let bestR2  = -Infinity;
  let bestRow = null;

  rows.forEach(row => {
    const cells = row.querySelectorAll('td');
    // Kolom R² ada di index 3
    if (cells[3]) {
      const val = parseFloat(cells[3].textContent);
      if (!isNaN(val) && val > bestR2) {
        bestR2  = val;
        bestRow = row;
      }
    }
  });

  if (bestRow) {
    bestRow.style.background  = 'rgba(245,158,11,0.06)';
    bestRow.style.borderLeft  = '3px solid #F59E0B';
    // Tambah badge "Best"
    const firstCell = bestRow.querySelector('td:first-child');
    if (firstCell) {
      const badge = document.createElement('span');
      badge.className = 'badge bg-warning text-dark ms-2';
      badge.style.fontSize = '0.65rem';
      badge.textContent = '★ Best';
      firstCell.appendChild(badge);
    }
  }
}

/* ── Smooth scroll untuk anchor link ─────────────────────── */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
}

/* ── Loading overlay untuk navigasi ──────────────────────── */
function initPageTransition() {
  document.querySelectorAll('a:not([href^="#"]):not([target])').forEach(link => {
    link.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      if (!href || href.startsWith('http') || href.startsWith('mailto')) return;
    });
  });
}

/* ── Animasi CSS keyframes ────────────────────────────────── */
const style = document.createElement('style');
style.textContent = `
  @keyframes slideInToast {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
  }
`;
document.head.appendChild(style);

/* ── INIT semua saat DOM ready ────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  setChartDefaults();
  initCounters();
  initSmoothScroll();
  initPageTransition();
  highlightBestRow();

  // Bootstrap tooltip global
  document.querySelectorAll('[data-bs-toggle="tooltip"]')
    .forEach(el => new bootstrap.Tooltip(el, { trigger: 'hover' }));
});