/* Modal figure expansion and clipboard helpers for Ouroboros project site. */
(() => {
  'use strict';

  const dialog = document.getElementById('figure-dialog');
  const expandedImg = document.getElementById('expanded-image');
  const caption = document.getElementById('expanded-caption');
  const closeBtn = document.getElementById('close-figure');

  document.querySelectorAll('[data-image]').forEach(btn => {
    btn.addEventListener('click', () => {
      if (!dialog || !expandedImg || !caption) return;
      expandedImg.src = btn.dataset.image;
      const img = btn.querySelector('img');
      expandedImg.alt = img ? img.alt : 'Expanded figure';
      caption.textContent = btn.dataset.caption || '';
      dialog.showModal();
    });
  });

  if (closeBtn && dialog) {
    closeBtn.addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', ev => {
      if (ev.target === dialog) dialog.close();
    });
  }

  function bindCopy(btnId, sourceId, statusId, labelText) {
    const btn = document.getElementById(btnId);
    const src = document.getElementById(sourceId);
    const status = document.getElementById(statusId);
    if (!btn || !src) return;
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(src.textContent);
        btn.textContent = 'Copied ✓';
        if (status) status.textContent = 'Copied to clipboard.';
        setTimeout(() => {
          btn.textContent = labelText;
          if (status) status.textContent = '';
        }, 2400);
      } catch {
        const sel = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(src);
        sel.removeAllRanges();
        sel.addRange(range);
        btn.textContent = 'Selected';
      }
    });
  }

  bindCopy('copy-quickstart', 'quickstart-code', 'quickstart-status', 'Copy commands');
  bindCopy('copy-citation', 'bibtex', 'copy-status', 'Copy citation');
})();
