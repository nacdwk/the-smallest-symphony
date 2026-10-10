// Programme UI: caption, readout panel, controls panel, hint and toast.
// Movements describe their controls declaratively; this module builds the DOM.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export class UI {
  constructor() {
    this.el = {
      hud: $('hud'), score: $('score'), tempo: $('cap-tempo'), title: $('cap-title'), body: $('cap-body'),
      fine: $('cap-fine'), fineBody: $('cap-fine-body'), readout: $('readout'), controls: $('controls'),
      hint: $('hint'), toast: $('toast'), prev: $('prev'), next: $('next'),
    };
    this._hintTimer = 0; this._toastTimer = 0;
    this._keys = new Map();
  }

  // ---- movement chrome (engine-owned) ------------------------------------------------
  buildScore(movements, onPick) {
    const s = this.el.score; s.innerHTML = '';
    movements.forEach((m, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.innerHTML = `<span class="num">${esc(m.numeral)}</span><span class="ttl">${esc(m.title)}</span>`;
      b.setAttribute('aria-label', `Movement ${m.numeral}: ${m.title}`);
      b.addEventListener('click', () => onPick(i));
      s.appendChild(b);
    });
  }
  setMovement(meta, index, count) {
    [...this.el.score.children].forEach((b, i) => { b.classList.toggle('on', i === index); b.setAttribute('aria-current', i === index ? 'step' : 'false'); });
    this.el.tempo.textContent = meta.tempo || '';
    this.el.title.innerHTML = `<span class="n">${esc(meta.numeral)}.</span>${esc(meta.title)}`;
    this.el.prev.disabled = index === 0;
    this.el.next.textContent = index === count - 1 ? 'Back to the beginning →' : 'Next movement →';
    this.setCaption({ body: meta.caption || '', fine: meta.fine || '' });
    this.clearPanels();
  }

  // ---- caption -----------------------------------------------------------------------
  /** body/fine are trusted HTML strings written by movement authors. */
  setCaption({ body, fine } = {}) {
    if (body !== undefined) this.el.body.innerHTML = body;
    if (fine !== undefined) {
      this.el.fineBody.innerHTML = fine;
      this.el.fine.hidden = !fine;
      this.el.fine.open = false;
    }
  }

  // ---- readout (top-right) ---------------------------------------------------------------
  /**
   * readout({ title, rows: [{ id, k, v, cls }], note, html })
   * Returns { set(id, value, cls?), note(html), title(text), el }.
   * Values are text unless they contain '<' (then HTML). Call set() every frame if needed; it skips no-op writes.
   */
  readout(spec = {}) {
    const r = this.el.readout; r.innerHTML = '';
    const cells = new Map();
    let titleEl = null, noteEl = null;
    if (spec.title !== undefined) { titleEl = document.createElement('p'); titleEl.className = 'rt'; titleEl.textContent = spec.title; r.appendChild(titleEl); }
    if (spec.html) { const d = document.createElement('div'); d.innerHTML = spec.html; r.appendChild(d); }
    (spec.rows || []).forEach((row) => {
      const d = document.createElement('div'); d.className = 'row';
      d.innerHTML = `<span class="k">${row.k}</span><span class="v ${row.cls || ''}"></span>`;
      const v = d.querySelector('.v'); r.appendChild(d); cells.set(row.id, { v, last: null });
      if (row.v !== undefined) write(row.id, row.v);
    });
    if (spec.note !== undefined) { noteEl = document.createElement('div'); noteEl.className = 'note'; noteEl.innerHTML = spec.note; r.appendChild(noteEl); }
    function write(id, value, cls) {
      const c = cells.get(id); if (!c) return;
      const s = String(value);
      if (s !== c.last) { if (s.includes('<')) c.v.innerHTML = s; else c.v.textContent = s; c.last = s; }
      if (cls !== undefined) c.v.className = 'v ' + cls;
    }
    return {
      el: r,
      set: write,
      note(html) { if (!noteEl) { noteEl = document.createElement('div'); noteEl.className = 'note'; r.appendChild(noteEl); } if (noteEl._last !== html) { noteEl.innerHTML = html; noteEl._last = html; } },
      title(text) { if (titleEl && titleEl.textContent !== text) titleEl.textContent = text; },
    };
  }

  // ---- controls (bottom-right) -------------------------------------------------------------
  /**
   * controls([...items]) where each item is one of
   *  { type:'slider', id, label, min, max, step, value, format(v)->string, onInput(v) }
   *  { type:'segmented', id, label, options:[{value,label}], value, onChange(value) }
   *  { type:'button', id, label, onClick(), kbd }          (several consecutive buttons share a row)
   *  { type:'toggle', id, label, value, onChange(bool) }
   *  { type:'keys', id, label, keys:[{ label, sub, color, kbd, onPress() }] }   (piano-style pads)
   *  { type:'html', html }
   * Returns { get(id) -> element, set(id, value), flash(id, keyIndex) }.
   * kbd: a KeyboardEvent.key value (e.g. '1', 'o', ' '). The engine routes keys to these handlers while the movement is active.
   */
  controls(items = []) {
    const root = this.el.controls; root.innerHTML = '';
    this._keys.clear();
    const reg = new Map();
    let btnRow = null;
    const fmt = (it, v) => (it.format ? it.format(v) : String(v));
    for (const it of items) {
      if (it.type !== 'button') btnRow = null;
      if (it.type === 'slider') {
        const w = document.createElement('div'); w.className = 'ctl';
        const id = 'c-' + it.id;
        w.innerHTML = `<label for="${id}"><span>${it.label}</span><output></output></label><input type="range" id="${id}" min="${it.min}" max="${it.max}" step="${it.step ?? 0.01}" value="${it.value}">`;
        const input = w.querySelector('input'), out = w.querySelector('output');
        const paint = () => { const p = (input.value - it.min) / (it.max - it.min) * 100; input.style.setProperty('--p', p + '%'); out.textContent = fmt(it, +input.value); };
        input.addEventListener('input', () => { paint(); it.onInput && it.onInput(+input.value); });
        input.addEventListener('keydown', (e) => e.stopPropagation());
        paint(); root.appendChild(w);
        reg.set(it.id, { el: input, set: (v) => { input.value = v; paint(); } });
      } else if (it.type === 'segmented') {
        const w = document.createElement('div'); w.className = 'ctl';
        w.innerHTML = it.label ? `<span class="lbl"><span>${it.label}</span></span>` : '';
        const seg = document.createElement('div'); seg.className = 'seg'; seg.setAttribute('role', 'group');
        const btns = it.options.map((o) => {
          const b = document.createElement('button'); b.type = 'button'; b.textContent = o.label; b.dataset.v = o.value;
          b.addEventListener('click', () => { mark(o.value); it.onChange && it.onChange(o.value); });
          if (o.kbd) this._keys.set(o.kbd.toLowerCase(), () => { mark(o.value); it.onChange && it.onChange(o.value); });
          seg.appendChild(b); return b;
        });
        const mark = (v) => btns.forEach((b) => { const on = b.dataset.v === String(v); b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); });
        mark(it.value); w.appendChild(seg); root.appendChild(w);
        reg.set(it.id, { el: seg, set: mark });
      } else if (it.type === 'button') {
        if (!btnRow) { btnRow = document.createElement('div'); btnRow.className = 'row-btns'; root.appendChild(btnRow); }
        const b = document.createElement('button'); b.type = 'button'; b.className = 'btn'; b.innerHTML = it.label;
        if (it.kbd) b.title = `Key: ${it.kbd === ' ' ? 'Space' : it.kbd.toUpperCase()}`;
        b.addEventListener('click', () => it.onClick && it.onClick());
        if (it.kbd) this._keys.set(it.kbd.toLowerCase(), () => { b.classList.add('on'); setTimeout(() => b.classList.remove('on'), 160); it.onClick && it.onClick(); });
        btnRow.appendChild(b);
        reg.set(it.id, { el: b, set: (v) => { if (typeof v === 'boolean') b.classList.toggle('on', v); else b.innerHTML = v; } });
      } else if (it.type === 'toggle') {
        const id = 'c-' + it.id;
        const w = document.createElement('label'); w.className = 'toggle'; w.htmlFor = id;
        w.innerHTML = `<span>${it.label}</span><input type="checkbox" role="switch" id="${id}" ${it.value ? 'checked' : ''}>`;
        const input = w.querySelector('input');
        input.addEventListener('change', () => it.onChange && it.onChange(input.checked));
        if (it.kbd) this._keys.set(it.kbd.toLowerCase(), () => { input.checked = !input.checked; it.onChange && it.onChange(input.checked); });
        root.appendChild(w);
        reg.set(it.id, { el: input, set: (v) => { input.checked = !!v; } });
      } else if (it.type === 'keys') {
        const w = document.createElement('div'); w.className = 'ctl';
        if (it.label) w.innerHTML = `<span class="lbl"><span>${it.label}</span><output></output></span>`;
        const k = document.createElement('div'); k.className = 'keys'; k.style.setProperty('--n', it.keys.length);
        const pads = it.keys.map((key, i) => {
          const b = document.createElement('button'); b.type = 'button';
          b.innerHTML = `${key.label}${key.sub ? `<small>${key.sub}</small>` : ''}`;
          if (key.color) b.style.setProperty('--kc', key.color);
          const hit = () => { b.classList.add('hit'); setTimeout(() => b.classList.remove('hit'), 180); key.onPress && key.onPress(); };
          b.addEventListener('pointerdown', (e) => { e.preventDefault(); hit(); });
          b.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); hit(); } });
          if (key.kbd) this._keys.set(key.kbd.toLowerCase(), hit);
          k.appendChild(b); return b;
        });
        w.appendChild(k); root.appendChild(w);
        const out = w.querySelector('output');
        reg.set(it.id, { el: k, set: (v) => { if (out) out.textContent = v; }, pads });
      } else if (it.type === 'html') {
        const d = document.createElement('div'); d.innerHTML = it.html; root.appendChild(d);
        if (it.id) reg.set(it.id, { el: d, set: (v) => { d.innerHTML = v; } });
      }
    }
    return {
      get: (id) => reg.get(id)?.el,
      set: (id, v) => reg.get(id)?.set(v),
      flash: (id, i) => { const p = reg.get(id)?.pads?.[i]; if (p) { p.classList.add('hit'); setTimeout(() => p.classList.remove('hit'), 180); } },
    };
  }

  /** Called by the engine for keys the movement did not consume. Returns true if a control handled it. */
  handleKey(e) {
    const f = this._keys.get(e.key.toLowerCase());
    if (f) { f(); return true; }
    return false;
  }

  clearPanels() {
    this.el.readout.innerHTML = '';
    this.el.controls.innerHTML = '';
    this._keys.clear();
    this.hideHint();
  }

  // ---- hint & toast ------------------------------------------------------------------------
  /** Small breathing instruction at the bottom centre. Hidden automatically after `ms` (default: until hideHint). */
  hint(text, ms = 0) {
    clearTimeout(this._hintTimer);
    this.el.hint.textContent = text;
    this.el.hint.classList.add('show');
    if (ms) this._hintTimer = setTimeout(() => this.hideHint(), ms);
  }
  hideHint() { clearTimeout(this._hintTimer); this.el.hint.classList.remove('show'); }
  /** Transient message at the top centre (plain text or trusted HTML). */
  toast(html, ms = 3200) {
    clearTimeout(this._toastTimer);
    this.el.toast.innerHTML = html;
    this.el.toast.classList.add('show');
    this._toastTimer = setTimeout(() => this.el.toast.classList.remove('show'), ms);
  }
}
