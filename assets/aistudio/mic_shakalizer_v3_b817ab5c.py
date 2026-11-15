"""
Mic Shakalizer v3 (pywebview).
pip install sounddevice numpy scipy pywebview
Выход -> CABLE Input (VB-Cable), в Discord микрофон -> CABLE Output.
"""
import json
import os

import numpy as np
import sounddevice as sd
import webview
from scipy.signal import lfilter

SR, BLOCK = 48000, 512
CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakal_config_v3.json")

# (ключ, подпись, min, max, шаг, по умолчанию)
GROUPS = [
    ("Микрофон", [
        ("in_gain", "Громкость входа", 0, 4, 0.05, 1.0),
        ("hp", "Срез низов, Гц", 20, 900, 10, 20),
        ("lowpass", "Срез верха, Гц", 300, 20000, 100, 20000),
        ("mic_res", "Резонанс капсюля, дБ", 0, 20, 0.5, 0)]),
    ("Грязь", [
        ("target_sr", "Частота дискретизации, Гц", 1000, 24000, 500, 24000),
        ("bits", "Битность", 1, 16, 1, 16),
        ("drive", "Дисторшн", 1, 30, 0.5, 1),
        ("noise", "Шум", 0, 1, 0.01, 0),
        ("agc", "AGC / накачка шума", 0, 1, 0.01, 0)]),
    ("Вода", [
        ("w_win", "Окно (8=256 … 11=2048)", 8, 11, 1, 10),
        ("w_smear", "Размазывание во времени", 0, 0.95, 0.01, 0),
        ("w_quant", "Грубость спектра", 0, 1, 0.01, 0),
        ("w_holes", "Дыры в спектре", 0, 0.8, 0.01, 0),
        ("w_phase", "Развал фазы", 0, 1, 0.01, 0)]),
    ("Питч", [
        ("ps_semi", "Сдвиг, полутонов", -12, 12, 0.5, 0),
        ("ps_win", "Окно, мс (меньше = грязнее)", 20, 120, 5, 50),
        ("ps_mix", "Микс (0.5 = гармония)", 0, 1, 0.01, 1)]),
    ("Комната", [
        ("rv_mix", "Реверб: микс", 0, 1, 0.01, 0),
        ("rv_size", "Реверб: размер", 0, 1, 0.01, 0.5),
        ("rv_damp", "Реверб: глушение верха, Гц", 1000, 12000, 100, 6000),
        ("echo_mix", "Эхо: громкость", 0, 1, 0.01, 0),
        ("echo_ms", "Эхо: задержка, мс", 60, 800, 10, 300),
        ("echo_fb", "Эхо: повторы", 0, 0.9, 0.01, 0.4),
        ("wall", "За стеной / в шкафу", 0, 1, 0.01, 0)]),
    ("Бас", [
        ("bass_db", "Усиление низов, дБ", 0, 24, 0.5, 0),
        ("bass_fc", "Частота среза, Гц", 40, 300, 5, 120),
        ("bass_sat", "Плотность (гармоники баса)", 0, 1, 0.01, 0.3)]),
    ("Связь / выход", [
        ("radio", "Рация", 0, 1, 0.01, 0),
        ("loss", "Потери пакетов", 0, 0.6, 0.01, 0),
        ("out_gain", "Громкость выхода", 0, 4, 0.05, 1.0)]),
]
PD = {p[0]: p[5] for _, ps in GROUPS for p in ps}
PRESETS = {
    "Чистый": {},
    "Фикспрайс": dict(in_gain=1.5, hp=350, lowpass=3800, mic_res=10, target_sr=11000, bits=8,
                      w_win=9, w_smear=0.3, w_quant=0.3, w_holes=0.1, w_phase=0.1,
                      agc=0.6, drive=2, noise=0.15),
    "Под водой": dict(hp=100, lowpass=1800, mic_res=4, target_sr=8000, bits=10, w_win=10,
                      w_smear=0.6, w_quant=0.5, w_holes=0.25, w_phase=0.4, agc=0.4, noise=0.05),
    "Тотальный шакал": dict(in_gain=1.5, hp=400, lowpass=2800, mic_res=14, target_sr=6000,
                            bits=5, w_win=10, w_smear=0.7, w_quant=0.7, w_holes=0.35,
                            w_phase=0.6, agc=0.8, drive=3, noise=0.25),
    "Бурундук": dict(ps_semi=7, ps_win=40),
    "Демон": dict(ps_semi=-7, ps_win=70, rv_mix=0.3, rv_size=0.7, rv_damp=3000, drive=2),
    "Гремлин": dict(ps_semi=5, ps_win=30, hp=300, lowpass=3500, bits=7, target_sr=10000,
                    w_smear=0.4, w_quant=0.4, w_holes=0.15, agc=0.5, noise=0.1),
    "Взрыв вселенной": dict(bass_db=18, bass_fc=120, bass_sat=0.6, rv_mix=0.25, rv_size=0.8, rv_damp=2500),
    "Ванная": dict(rv_mix=0.5, rv_size=0.6, rv_damp=9000),
    "Пещера": dict(rv_mix=0.7, rv_size=0.95, rv_damp=3500, echo_mix=0.35, echo_ms=450, echo_fb=0.5),
    "За стеной": dict(wall=0.85, rv_mix=0.15, rv_size=0.4),
    "Рация": dict(radio=0.9, noise=0.1, loss=0.05, bits=10, target_sr=8000),
    "Глючный звонок": dict(loss=0.25, lowpass=3500, target_sr=8000, bits=9, w_win=9,
                           w_smear=0.4, w_quant=0.4, w_holes=0.2, agc=0.5),
}


def onepole(x, fc, zi):
    a = np.exp(-2 * np.pi * min(fc, SR / 2 - 100) / SR)
    return lfilter([1 - a], [1, -a], x, zi=zi)


def peaking(fc, gain_db, q=2.0):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / SR
    al = np.sin(w0) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w0), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w0), 1 - al / A])
    return b / a[0], a / a[0]


class FIFO:
    """Линия задержки; длина n должна быть >= размера блока."""

    def __init__(self, n):
        self.buf = np.zeros(n)

    def tap(self, k):
        return self.buf[:k]

    def push(self, x):
        self.buf = np.concatenate((self.buf[len(x):], x))


class Reverb:
    C, A = [1307, 1423, 1559, 1699], [601, 743]

    def __init__(self):
        self.c = [FIFO(n) for n in self.C]
        self.cz = [np.zeros(1) for _ in self.C]
        self.a = [FIFO(n) for n in self.A]

    def process(self, x, size, damp):
        fb, L, out = 0.6 + 0.37 * size, len(x), np.zeros(len(x))
        for i, f in enumerate(self.c):
            d, self.cz[i] = onepole(f.tap(L), damp, self.cz[i])
            y = (1 - fb) * x + fb * d
            f.push(y)
            out += y
        out *= 0.75
        for f in self.a:
            vd = f.tap(L)
            v = out + 0.5 * vd
            f.push(v)
            out = -0.5 * v + vd
        return out


class Water:
    def __init__(self):
        self.reset(1024)

    def reset(self, n):
        self.n, self.hop = n, n // 2
        self.win = np.sqrt(np.hanning(n + 1)[:-1])
        self.inb, self.acc, self.outq, self.prev = np.zeros(0), np.zeros(n), np.zeros(n), None

    def process(self, x, p):
        n = 2 ** int(p["w_win"])
        if n != self.n:
            self.reset(n)
        self.inb = np.concatenate((self.inb, x))
        nb = n // 2 + 1
        while len(self.inb) >= n:
            spec = np.fft.rfft(self.inb[:n] * self.win)
            mag, ph = np.abs(spec), np.angle(spec)
            if self.prev is None:
                self.prev = mag
            mag = p["w_smear"] * self.prev + (1 - p["w_smear"]) * mag
            self.prev = mag
            if p["w_quant"] > 0.01:
                st = p["w_quant"] * 3
                mag = np.exp(np.round(np.log(mag + 1e-6) / st) * st)
            if p["w_holes"] > 0:
                mag = mag * np.repeat(np.random.rand(nb // 4 + 1) > p["w_holes"], 4)[:nb]
            if p["w_phase"] > 0:
                ph = ph + (np.random.rand(nb) - 0.5) * 2 * np.pi * p["w_phase"]
            self.acc += np.fft.irfft(mag * np.exp(1j * ph), n) * self.win
            self.outq = np.concatenate((self.outq, self.acc[:self.hop]))
            self.acc = np.concatenate((self.acc[self.hop:], np.zeros(self.hop)))
            self.inb = self.inb[self.hop:]
        L = len(x)
        if len(self.outq) < L:
            self.outq = np.concatenate((self.outq, np.zeros(L - len(self.outq))))
        out, self.outq = self.outq[:L], self.outq[L:]
        return out


class Pitch:
    """Питч-шифт: две модулируемые линии задержки с перекрёстным затуханием."""
    M = 6144  # > максимальное окно (120 мс = 5760 сэмплов)

    def __init__(self):
        self.hist, self.ph = np.zeros(self.M), 0.0

    def process(self, x, semi, win_ms):
        L, W = len(x), win_ms * SR / 1000
        rate = (1 - 2 ** (semi / 12)) / W
        ph = (self.ph + np.arange(L) * rate) % 1.0
        self.ph = (self.ph + L * rate) % 1.0
        ext = np.concatenate((self.hist, x))
        self.hist = ext[-self.M:]
        out = np.zeros(L)
        for off in (0.0, 0.5):
            p = (ph + off) % 1.0
            pos = self.M + np.arange(L) - (1 + p * W)
            i0 = np.floor(pos).astype(np.int64)
            fr = pos - i0
            out += (ext[i0] * (1 - fr) + ext[i0 + 1] * fr) * np.sin(np.pi * p) ** 2
        return out


class Engine:
    def __init__(self):
        self.p = dict(PD, bypass=False, host="MME", **{"in": "", "out": ""})
        self.stream, self.level = None, 0.0
        self.water = Water()
        self.reset_state()

    def reset_state(self):
        self.counter, self.held, self.gain = 0, 0.0, 1.0
        self.z = {k: np.zeros(1) for k in ("hp", "lp", "w1", "w2", "r1", "r2", "b1", "b2")}
        self.zpk = np.zeros(2)
        self.rev, self.echo, self.ps = Reverb(), FIFO(2880), Pitch()
        self.last = np.zeros(BLOCK)
        self.water.reset(2 ** int(self.p["w_win"]))

    def callback(self, indata, outdata, L, t, status):
        p, z = self.p, self.z
        y = indata[:, 0].astype(np.float64) * p["in_gain"]
        if not p["bypass"]:
            # --- питч (на чистом голосе, дальше всё портится) ---
            if p["ps_semi"] != 0:
                sh = self.ps.process(y, p["ps_semi"], p["ps_win"])
                y = (1 - p["ps_mix"]) * y + p["ps_mix"] * sh
            # --- комната ---
            if p["rv_mix"] > 0:
                wet = self.rev.process(y, p["rv_size"], p["rv_damp"])
                y = (1 - 0.5 * p["rv_mix"]) * y + p["rv_mix"] * wet
            if p["echo_mix"] > 0:
                D = int(p["echo_ms"] * SR / 1000)
                if len(self.echo.buf) != D:
                    self.echo = FIFO(D)
                d = self.echo.tap(L).copy()
                self.echo.push(y + p["echo_fb"] * d)
                y = y + p["echo_mix"] * d
            if p["wall"] > 0:
                fc = 300 + 7000 * (1 - p["wall"]) ** 2
                y, z["w1"] = onepole(y, fc, z["w1"])
                y, z["w2"] = onepole(y, fc, z["w2"])
                y = y * (1 - 0.4 * p["wall"])
            # --- дешёвый микрофон ---
            low, z["hp"] = onepole(y, p["hp"], z["hp"])
            y = y - low
            if p["mic_res"] > 0:
                b, a = peaking(2600, p["mic_res"])
                y, self.zpk = lfilter(b, a, y, zi=self.zpk)
            # --- понижение частоты дискретизации ---
            f = max(1, int(round(SR / max(1000, p["target_sr"]))))
            g = self.counter + np.arange(L, dtype=np.int64)
            src = (g // f) * f - self.counter
            ext = np.concatenate(([self.held], y))
            y = ext[np.maximum(src, -1) + 1]
            self.held = ext[src[-1] + 1]
            self.counter = (self.counter + L) % (2 ** 31)
            y, z["lp"] = onepole(y, p["lowpass"], z["lp"])
            # --- вода ---
            if p["w_smear"] > 0 or p["w_quant"] > 0.01 or p["w_holes"] > 0 or p["w_phase"] > 0:
                y = self.water.process(y, p)
            # --- AGC, дисторшн, шум, битность ---
            if p["agc"] > 0:
                rms = np.sqrt(np.mean(y * y)) + 1e-5
                want = 1 + (np.clip(0.15 / rms, 0.5, 20) - 1) * p["agc"]
                new = self.gain + (want - self.gain) * (0.3 if want < self.gain else 0.02)
                y = y * np.linspace(self.gain, new, L)
                self.gain = new
            if p["drive"] > 1.0:
                y = np.tanh(y * p["drive"]) / np.tanh(p["drive"])
            if p["noise"] > 0:
                y = y + np.random.randn(L) * p["noise"] * 0.05
            lv = 2 ** (int(p["bits"]) - 1)
            y = np.round(np.clip(y, -1, 1) * lv) / lv
            # --- связь ---
            if p["radio"] > 0:
                r = p["radio"]
                lo, z["r1"] = onepole(y, 300, z["r1"])
                h, z["r2"] = onepole(y - lo, 3000, z["r2"])
                h = np.tanh(h * (1 + 6 * r)) * 0.8 + np.random.randn(L) * 0.02 * r
                y = (1 - r) * y + r * h
            if p["loss"] > 0:
                if np.random.rand() < p["loss"]:
                    y = self.last if (np.random.rand() < 0.7 and len(self.last) == L) else np.zeros(L)
                else:
                    self.last = y.copy()
            # --- бас: усиливаем и насыщаем ТОЛЬКО низкую полосу ---
            if p["bass_db"] > 0:
                lo, z["b1"] = onepole(y, p["bass_fc"], z["b1"])
                lo, z["b2"] = onepole(lo, p["bass_fc"], z["b2"])
                boosted = lo * 10 ** (p["bass_db"] / 20) * (1 + 3 * p["bass_sat"])
                y = (y - lo) + 0.9 * np.tanh(boosted / 0.9)
        y = np.clip(y * p["out_gain"], -1, 1)
        self.level = float(np.max(np.abs(y)))
        outdata[:, 0] = y.astype(np.float32)

    def start(self, i, o):
        self.stop()
        self.reset_state()
        self.stream = sd.Stream(samplerate=SR, blocksize=BLOCK, dtype="float32", device=(i, o),
                                channels=(1, 1), callback=self.callback)
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class Api:
    # все атрибуты с "_", иначе pywebview попытается их обойти
    def __init__(self):
        self._e = Engine()
        self._imap, self._omap = {}, {}
        try:
            with open(CFG, encoding="utf-8") as f:
                self._e.p.update(json.load(f))
        except Exception:
            pass

    def init(self):
        p = self._e.p
        hosts = [h["name"] for h in sd.query_hostapis()]
        return {"groups": GROUPS, "p": {k: p[k] for k in PD}, "hosts": hosts,
                "host": p["host"] if p["host"] in hosts else "MME",
                "presets": list(PRESETS), "bypass": p["bypass"]}

    def devices(self, host):
        p = self._e.p
        p["host"] = host
        hi = next(i for i, h in enumerate(sd.query_hostapis()) if h["name"] == host)
        devs = [(i, d) for i, d in enumerate(sd.query_devices()) if d["hostapi"] == hi]
        self._imap = {d["name"]: i for i, d in devs if d["max_input_channels"] > 0}
        self._omap = {d["name"]: i for i, d in devs if d["max_output_channels"] > 0}
        ins, outs = list(self._imap), list(self._omap)
        sin = p["in"] if p["in"] in ins else (ins[0] if ins else "")
        sout = next((n for n in outs if "CABLE Input" in n), p["out"] if p["out"] in outs else (outs[0] if outs else ""))
        return {"ins": ins, "outs": outs, "sin": sin, "sout": sout}

    def set(self, k, v):
        self._e.p[k] = float(v)

    def preset(self, name):
        vals = dict(PD, **PRESETS[name])
        self._e.p.update(vals)
        return vals

    def bypass(self, b):
        self._e.p["bypass"] = bool(b)

    def start(self, i, o):
        try:
            self._e.start(self._imap[i], self._omap[o])
            self._e.p["in"], self._e.p["out"] = i, o
            return ""
        except Exception as ex:
            return str(ex)[:120]

    def stop(self):
        self._e.stop()

    def level(self):
        return self._e.level

    def _shutdown(self):
        self._e.stop()
        try:
            with open(CFG, "w", encoding="utf-8") as f:
                json.dump(self._e.p, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


HTML = """<!doctype html><html><head><meta charset=utf-8><style>
:root{--bg:#14161a;--card:#1d2026;--fg:#e6e8ec;--mut:#8b919c;--acc:#5b8cff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px 'Segoe UI',sans-serif;height:100vh;display:flex;flex-direction:column}
header{display:flex;gap:10px;align-items:flex-end;padding:12px 16px;background:var(--card);flex-wrap:wrap}
label{display:flex;flex-direction:column;gap:3px;color:var(--mut);font-size:11px}
select,button{background:#262a32;color:var(--fg);border:1px solid #333843;border-radius:6px;padding:6px 8px;font:inherit;max-width:230px}
button{cursor:pointer}
#go{background:var(--acc);border:0;font-weight:600;padding:8px 20px}#go.on{background:#d9534f}
#by{font-weight:600}#by.on{background:#e0a030;color:#000;border:0}
#st{color:var(--mut);align-self:center}
#meter{height:4px;background:#262a32}#lvl{height:100%;width:0;background:#4ec97b}
main{flex:1;overflow:auto;display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;padding:14px 16px;align-content:start}
.card{background:var(--card);border-radius:10px;padding:10px 14px}
h3{margin:0 0 4px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--acc)}
.row{display:grid;grid-template-columns:1fr 50px;gap:0 8px;margin:8px 0}
.row span{grid-column:1/3;color:var(--mut)}.row output{text-align:right}
input[type=range]{width:100%;accent-color:var(--acc)}
</style></head><body>
<header>
<label>Host API<select id=host></select></label>
<label>Вход (микрофон)<select id=din></select></label>
<label>Выход (CABLE Input)<select id=dout></select></label>
<label>Пресет<select id=pre></select></label>
<button id=by></button><button id=go>Старт</button><span id=st>Остановлено</span>
</header><div id=meter><div id=lvl></div></div><main id=m></main>
<script>
window.addEventListener('pywebviewready',async()=>{
const a=pywebview.api,st=await a.init(),$=i=>document.getElementById(i);
st.groups.forEach(([t,ps])=>{
 const c=document.createElement('div');c.className='card';c.innerHTML='<h3>'+t+'</h3>';
 ps.forEach(([k,l,lo,hi,step])=>{
  const r=document.createElement('div');r.className='row';
  r.innerHTML=`<span>${l}</span><input type=range id=s_${k} min=${lo} max=${hi} step=${step} value=${st.p[k]}><output id=o_${k}>${st.p[k]}</output>`;
  r.querySelector('input').oninput=e=>{const v=+e.target.value;$('o_'+k).textContent=v;a.set(k,v)};
  c.appendChild(r)});
 $('m').appendChild(c)});
st.hosts.forEach(h=>$('host').add(new Option(h,h)));$('host').value=st.host;
st.presets.forEach(n=>$('pre').add(new Option(n,n)));
const dev=async()=>{const d=await a.devices($('host').value);
 for(const [id,l,s] of [['din',d.ins,d.sin],['dout',d.outs,d.sout]]){$(id).innerHTML='';l.forEach(n=>$(id).add(new Option(n,n)));$(id).value=s}};
$('host').onchange=dev;await dev();
$('pre').onchange=async()=>{const v=await a.preset($('pre').value);for(const k in v){$('s_'+k).value=v[k];$('o_'+k).textContent=v[k]}};
let on=false;
$('go').onclick=async()=>{
 if(on){await a.stop();on=false}
 else{const r=await a.start($('din').value,$('dout').value);if(r){$('st').textContent='Ошибка: '+r;return}on=true}
 $('go').textContent=on?'Стоп':'Старт';$('go').classList.toggle('on',on);$('st').textContent=on?'Работает':'Остановлено'};
let by=st.bypass;const sb=()=>{$('by').classList.toggle('on',by);$('by').textContent=by?'Режим: НОРМАЛЬНЫЙ':'Режим: ШАКАЛ'};
$('by').onclick=()=>{by=!by;a.bypass(by);sb()};sb();
setInterval(async()=>{$('lvl').style.width=(await a.level()*100)+'%'},80);
});
</script></body></html>"""

if __name__ == "__main__":
    api = Api()
    w = webview.create_window("Mic Shakalizer", html=HTML, js_api=api,
                              width=1180, height=640, min_size=(760, 400))
    w.events.closed += api._shutdown
    webview.start()