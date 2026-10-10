// Real-time headless screenshot over the Chrome DevTools Protocol.
// Usage: node tools/snap.mjs "<query>" out.png [width] [height] [waitMs]
// Widths under 760 emulate a phone (true viewport size, touch, coarse pointer).
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';

const [q = '', out = 'shot.png', ws = '1440', hs = '900', waitS = '6000'] = process.argv.slice(2);
const W = +ws, H = +hs, WAIT = +waitS;
const PORT = process.env.PORT || 8765;
const phone = W < 760;
const url = `http://127.0.0.1:${PORT}/?${q}`;
const CH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const prof = mkdtempSync(join(tmpdir(), 'snapcdp-'));

const chrome = spawn(CH, [
  '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${prof}`, '--hide-scrollbars', '--mute-audio',
  '--no-first-run', '--no-default-browser-check', '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
  '--disable-backgrounding-occluded-windows', `--window-size=${Math.max(W, 800)},${Math.max(H, 600)}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });

const kill = () => { try { chrome.kill('SIGKILL'); } catch (e) {} try { rmSync(prof, { recursive: true, force: true }); } catch (e) {} };
const timer = setTimeout(() => { console.log('snap: timed out'); kill(); process.exit(2); }, WAIT + 45000);

const wsUrl = await new Promise((res, rej) => {
  let buf = '';
  chrome.stderr.on('data', (d) => { buf += d; const m = /DevTools listening on (ws:\/\/\S+)/.exec(buf); if (m) res(m[1]); });
  chrome.on('exit', () => rej(new Error('chrome exited')));
});

const sock = new WebSocket(wsUrl);
await new Promise((r) => sock.addEventListener('open', r, { once: true }));
let id = 0; const pending = new Map(); const handlers = [];
sock.addEventListener('message', (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) { const { res, rej } = pending.get(msg.id); pending.delete(msg.id); msg.error ? rej(new Error(msg.error.message)) : res(msg.result); }
  else handlers.forEach((h) => h(msg));
});
const send = (method, params = {}, sessionId) => new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); sock.send(JSON.stringify({ id: i, method, params, sessionId })); });

const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
const S = (m, p) => send(m, p, sessionId);

const fmt = (args) => args.map((a) => (a.value !== undefined ? (typeof a.value === 'string' ? a.value : JSON.stringify(a.value)) : a.description || a.type)).join(' ');
handlers.push((msg) => {
  if (msg.sessionId !== sessionId) return;
  if (msg.method === 'Runtime.consoleAPICalled') console.log(`CONSOLE(${msg.params.type}): ${fmt(msg.params.args).slice(0, 600)}`);
  if (msg.method === 'Runtime.exceptionThrown') { const d = msg.params.exceptionDetails; console.log(`Uncaught ${(d.exception && d.exception.description) || d.text} @${d.url || ''}:${d.lineNumber}`.slice(0, 900)); }
  if (msg.method === 'Log.entryAdded' && /error|warning/.test(msg.params.entry.level)) console.log(`LOG(${msg.params.entry.level}): ${msg.params.entry.text} ${msg.params.entry.url || ""}`.slice(0, 400));
});
await S('Runtime.enable'); await S('Log.enable'); await S('Page.enable');
await S('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: phone, screenWidth: W, screenHeight: H });
if (phone) { await S('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 }); await S('Emulation.setEmitTouchEventsForMouse', { enabled: true }); }
const loaded = new Promise((r) => handlers.push((m) => { if (m.sessionId === sessionId && m.method === 'Page.loadEventFired') r(); }));
await S('Page.navigate', { url });
await Promise.race([loaded, new Promise((r) => setTimeout(r, 15000))]);
await new Promise((r) => setTimeout(r, WAIT));
const { data } = await S('Page.captureScreenshot', { format: 'png' });
const outPath = resolve(out);
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, Buffer.from(data, 'base64'));
console.log(`${Buffer.from(data, 'base64').length} bytes written to file ${outPath}`);
clearTimeout(timer);
sock.close(); kill();
process.exit(0);
