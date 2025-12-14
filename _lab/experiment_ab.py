import os, time, json, csv, random, threading
import tkinter as tk
from tkinter import ttk

try:
    import speech_recognition as sr
except Exception:
    sr = None

APP_TITLE = "R&E AB Experiment (A: Grid Select, B: File Task)"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

MODE_A = "A"
MODE_B = "B"

COND_MOUSE = "C1_mouse_keyboard"
COND_STT = "C2_stt_only"
COND_GAZE = "C3_gaze_only"
COND_MULTI = "C4_multimodal"

CONDITIONS = [COND_MOUSE, COND_STT, COND_GAZE, COND_MULTI]

def now_ms():
    return int(time.time() * 1000)

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("980x680")

        self.mode = MODE_A
        self.cond = COND_MULTI

        self.trial_active = False
        self.trial_id = 0
        self.trial_start_ms = None
        self.goal = None

        self.hover_target = None
        self.hover_since_ms = None

        self.dwell_ms = 800

        self.events = []
        self.metrics = {}

        self.stt_enabled = False
        self.stt_thread = None
        self.stt_last_text = ""
        self.stt_queue = []
        self.stt_lock = threading.Lock()

        self._ui()
        self._binds()
        self._tick()

    def _ui(self):
        top = ttk.Frame(self.root)
        top.pack(side="top", fill="x", padx=10, pady=10)

        self.lbl_mode = ttk.Label(top, text=f"MODE: {self.mode}")
        self.lbl_mode.pack(side="left")

        self.lbl_cond = ttk.Label(top, text=f"  COND: {self.cond}")
        self.lbl_cond.pack(side="left")

        self.lbl_trial = ttk.Label(top, text="  TRIAL: -")
        self.lbl_trial.pack(side="left")

        self.lbl_goal = ttk.Label(top, text="  GOAL: -")
        self.lbl_goal.pack(side="left")

        self.lbl_stt = ttk.Label(top, text="  STT: OFF")
        self.lbl_stt.pack(side="left")

        mid = ttk.Frame(self.root)
        mid.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(mid, bg="#111", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(mid, width=280)
        right.pack(side="right", fill="y")

        self.txt = tk.Text(right, height=28, width=34)
        self.txt.pack(side="top", fill="both", expand=True)

        btns = ttk.Frame(right)
        btns.pack(side="bottom", fill="x")

        ttk.Button(btns, text="Start/Stop (Space)", command=self.toggle_trial).pack(fill="x")
        ttk.Button(btns, text="Next Cond (C)", command=self.next_condition).pack(fill="x")
        ttk.Button(btns, text="Switch Mode (Tab)", command=self.switch_mode).pack(fill="x")
        ttk.Button(btns, text="Toggle STT (S)", command=self.toggle_stt).pack(fill="x")
        ttk.Button(btns, text="Save Log (Auto at End)", command=self.save_log).pack(fill="x")

        self._build_scene()

    def _binds(self):
        self.root.bind("<space>", lambda e: self.toggle_trial())
        self.root.bind("<Tab>", lambda e: self.switch_mode())
        self.root.bind("c", lambda e: self.next_condition())
        self.root.bind("C", lambda e: self.next_condition())
        self.root.bind("s", lambda e: self.toggle_stt())
        self.root.bind("S", lambda e: self.toggle_stt())
        self.root.bind("<Return>", lambda e: self.confirm_action(source="enter"))
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_click)

    def _log(self, typ, **data):
        ev = {"t_ms": now_ms(), "type": typ, **data}
        self.events.append(ev)
        self.txt.insert("end", json.dumps(ev, ensure_ascii=False) + "\n")
        self.txt.see("end")

    def _set_status(self):
        self.lbl_mode.config(text=f"MODE: {self.mode}")
        self.lbl_cond.config(text=f"  COND: {self.cond}")
        self.lbl_trial.config(text=f"  TRIAL: {self.trial_id if self.trial_active else '-'}")
        self.lbl_goal.config(text=f"  GOAL: {self.goal if self.goal else '-'}")
        self.lbl_stt.config(text=f"  STT: {'ON' if self.stt_enabled else 'OFF'}")

    def _build_scene(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 640
        h = self.canvas.winfo_height() or 540

        if self.mode == MODE_A:
            self._draw_grid(w, h)
        else:
            self._draw_filepane(w, h)

    def _draw_grid(self, w, h):
        pad = 60
        gw = w - pad*2
        gh = h - pad*2
        cell_w = gw / 3
        cell_h = gh / 3

        self.grid_cells = {}
        letters = ["A","B","C"]
        for r in range(3):
            for c in range(3):
                x1 = pad + c*cell_w
                y1 = pad + r*cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                key = f"{letters[c]}{r+1}"
                rect = self.canvas.create_rectangle(x1,y1,x2,y2, outline="#666", width=2)
                txt = self.canvas.create_text((x1+x2)/2,(y1+y2)/2, text=key, fill="#ddd", font=("Consolas", 26))
                self.grid_cells[key] = {"rect": rect, "txt": txt, "bbox": (x1,y1,x2,y2)}
        self._log("scene", mode="A_grid_ready")

    def _draw_filepane(self, w, h):
        pad = 40
        x1, y1, x2, y2 = pad, pad, w-pad, h-pad
        self.canvas.create_rectangle(x1,y1,x2,y2, outline="#666", width=2)

        self.files = []
        for i in range(6):
            ext = "pdf" if i < 3 else "txt"
            name = f"file_{i+1}.{ext}"
            self.files.append(name)

        self.file_items = {}
        row_h = 52
        fx1, fy = x1+30, y1+30
        for idx, name in enumerate(self.files):
            iy1 = fy + idx*row_h
            iy2 = iy1 + row_h - 8
            rx1, rx2 = fx1, x2-30
            rect = self.canvas.create_rectangle(rx1, iy1, rx2, iy2, outline="#444", width=2)
            txt = self.canvas.create_text(rx1+12, (iy1+iy2)/2, anchor="w", text=name, fill="#ddd", font=("Consolas", 18))
            self.file_items[name] = {"rect": rect, "txt": txt, "bbox": (rx1,iy1,rx2,iy2), "alive": True}

        self.b_actions = {"open": None, "delete": None}
        ax = x1+30
        ay = y2-110
        for j, act in enumerate(["open","delete"]):
            bx1 = ax + j*160
            by1 = ay
            bx2 = bx1 + 140
            by2 = by1 + 60
            rect = self.canvas.create_rectangle(bx1,by1,bx2,by2, outline="#555", width=2)
            txt = self.canvas.create_text((bx1+bx2)/2,(by1+by2)/2, text=act.upper(), fill="#ddd", font=("Consolas", 18))
            self.b_actions[act] = {"rect": rect, "txt": txt, "bbox": (bx1,by1,bx2,by2)}
        self._log("scene", mode="B_file_ready")

    def switch_mode(self):
        if self.trial_active:
            return
        self.mode = MODE_B if self.mode == MODE_A else MODE_A
        self.goal = None
        self.hover_target = None
        self.hover_since_ms = None
        self._build_scene()
        self._set_status()

    def next_condition(self):
        if self.trial_active:
            return
        i = CONDITIONS.index(self.cond)
        self.cond = CONDITIONS[(i+1) % len(CONDITIONS)]
        self._log("condition", cond=self.cond)
        self._set_status()

    def toggle_trial(self):
        if not self.trial_active:
            self.start_trial()
        else:
            self.end_trial(success=False, reason="manual_stop")

    def start_trial(self):
        self.trial_active = True
        self.trial_id += 1
        self.trial_start_ms = now_ms()
        self.events = []
        self.metrics = {}
        self.hover_target = None
        self.hover_since_ms = None
        self.goal = self._new_goal()
        self._log("trial_start", trial=self.trial_id, mode=self.mode, cond=self.cond, goal=self.goal)
        self._set_status()
        self._highlight_goal()

    def _new_goal(self):
        if self.mode == MODE_A:
            return random.choice(list(self.grid_cells.keys()))
        else:
            alive = [f for f,info in self.file_items.items() if info["alive"]]
            if not alive:
                for f in self.file_items:
                    self.file_items[f]["alive"] = True
                    self.canvas.itemconfig(self.file_items[f]["rect"], outline="#444")
                    self.canvas.itemconfig(self.file_items[f]["txt"], fill="#ddd")
                alive = list(self.file_items.keys())
            target = random.choice(alive)
            act = random.choice(["open","delete"])
            return f"{act}:{target}"

    def _highlight_goal(self):
        if self.mode == MODE_A:
            for k,info in self.grid_cells.items():
                self.canvas.itemconfig(info["rect"], outline="#666")
            self.canvas.itemconfig(self.grid_cells[self.goal]["rect"], outline="#0f0")
        else:
            for n,info in self.file_items.items():
                if info["alive"]:
                    self.canvas.itemconfig(info["rect"], outline="#444")
            for act,info in self.b_actions.items():
                self.canvas.itemconfig(info["rect"], outline="#555")
            act, target = self.goal.split(":",1)
            if self.file_items[target]["alive"]:
                self.canvas.itemconfig(self.file_items[target]["rect"], outline="#0f0")
            self.canvas.itemconfig(self.b_actions[act]["rect"], outline="#0f0")

    def end_trial(self, success, reason=""):
        if not self.trial_active:
            return
        t_end = now_ms()
        t_total = (t_end - self.trial_start_ms) / 1000.0
        self._log("trial_end", trial=self.trial_id, success=bool(success), reason=reason, total_s=t_total)
        self.trial_active = False
        self._set_status()
        self.save_log()

    def save_log(self):
        if not self.events:
            return
        base = f"{int(time.time())}_trial{self.trial_id}_{self.mode}_{self.cond}"
        jpath = os.path.join(LOG_DIR, base + ".jsonl")
        cpath = os.path.join(LOG_DIR, base + ".csv")

        with open(jpath, "w", encoding="utf-8") as f:
            for ev in self.events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

        keys = sorted({k for ev in self.events for k in ev.keys()})
        with open(cpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for ev in self.events:
                w.writerow(ev)

        self._log("log_saved", jsonl=jpath, csv=cpath)

    def on_mouse_move(self, e):
        x,y = e.x, e.y
        if not self.trial_active:
            return
        tgt = self._hit_test(x,y)
        if tgt != self.hover_target:
            self.hover_target = tgt
            self.hover_since_ms = now_ms() if tgt else None
            self._log("hover", target=str(tgt))
        self._apply_hover_ui(tgt)

        if self.cond in [COND_GAZE, COND_MULTI] and tgt:
            if self.hover_since_ms and (now_ms() - self.hover_since_ms) >= self.dwell_ms:
                if self.cond == COND_GAZE:
                    self.confirm_action(source="dwell")
                else:
                    pass

    def on_click(self, e):
        if not self.trial_active:
            return
        if self.cond != COND_MOUSE:
            self._log("click_ignored", cond=self.cond)
            return
        self.confirm_action(source="mouse_click")

    def confirm_action(self, source=""):
        if not self.trial_active:
            return
        if self.mode == MODE_A:
            if not self.hover_target:
                self._log("confirm_fail", source=source, reason="no_target")
                return
            chosen = self.hover_target
            self._log("confirm", source=source, chosen=chosen)
            if chosen == self.goal:
                self.end_trial(True, reason="correct")
            else:
                self._log("error", kind="wrong_target", chosen=chosen, goal=self.goal)
                self.end_trial(False, reason="wrong_target")
        else:
            if not self.hover_target:
                self._log("confirm_fail", source=source, reason="no_target")
                return
            act, target = self.goal.split(":",1)
            chosen = self.hover_target
            self._log("confirm", source=source, chosen=str(chosen))

            if isinstance(chosen, tuple) and chosen[0] == "file":
                picked = chosen[1]
                if picked != target:
                    self._log("error", kind="wrong_file", picked=picked, goal=target)
                    self.end_trial(False, reason="wrong_file")
                    return
                self._log("file_selected", file=picked)
                return

            if isinstance(chosen, tuple) and chosen[0] == "action":
                picked_act = chosen[1]
                if picked_act != act:
                    self._log("error", kind="wrong_action", picked=picked_act, goal=act)
                    self.end_trial(False, reason="wrong_action")
                    return

                if picked_act == "open":
                    self._log("action_open", file=target)
                    self.end_trial(True, reason="open_ok")
                else:
                    if self.file_items[target]["alive"]:
                        self.file_items[target]["alive"] = False
                        self.canvas.itemconfig(self.file_items[target]["rect"], outline="#222")
                        self.canvas.itemconfig(self.file_items[target]["txt"], fill="#444")
                    self._log("action_delete", file=target)
                    self.end_trial(True, reason="delete_ok")
                return

            self._log("confirm_fail", source=source, reason="unknown_target")

    def _hit_test(self, x,y):
        if self.mode == MODE_A:
            for k,info in self.grid_cells.items():
                x1,y1,x2,y2 = info["bbox"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return k
            return None
        else:
            for n,info in self.file_items.items():
                if not info["alive"]:
                    continue
                x1,y1,x2,y2 = info["bbox"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return ("file", n)
            for act,info in self.b_actions.items():
                x1,y1,x2,y2 = info["bbox"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return ("action", act)
            return None

    def _apply_hover_ui(self, tgt):
        if self.mode == MODE_A:
            for k,info in self.grid_cells.items():
                self.canvas.itemconfig(info["rect"], width=2)
            if tgt and tgt in self.grid_cells:
                self.canvas.itemconfig(self.grid_cells[tgt]["rect"], width=4)
        else:
            for n,info in self.file_items.items():
                if info["alive"]:
                    self.canvas.itemconfig(info["rect"], width=2)
            for act,info in self.b_actions.items():
                self.canvas.itemconfig(info["rect"], width=2)

            if isinstance(tgt, tuple) and tgt[0] == "file":
                self.canvas.itemconfig(self.file_items[tgt[1]]["rect"], width=4)
            if isinstance(tgt, tuple) and tgt[0] == "action":
                self.canvas.itemconfig(self.b_actions[tgt[1]]["rect"], width=4)

    def toggle_stt(self):
        if sr is None:
            self._log("stt_unavailable", reason="speechrecognition_not_installed")
            return
        self.stt_enabled = not self.stt_enabled
        self._set_status()
        self._log("stt_toggle", enabled=self.stt_enabled)
        if self.stt_enabled and (self.stt_thread is None or not self.stt_thread.is_alive()):
            self.stt_thread = threading.Thread(target=self._stt_loop, daemon=True)
            self.stt_thread.start()

    def _stt_loop(self):
        r = sr.Recognizer()
        mic = None
        try:
            mic = sr.Microphone()
        except Exception as e:
            self._log("stt_error", where="Microphone", err=str(e))
            self.stt_enabled = False
            self._set_status()
            return

        while True:
            if not self.stt_enabled:
                time.sleep(0.2)
                continue
            try:
                with mic as source:
                    r.adjust_for_ambient_noise(source, duration=0.2)
                    audio = r.listen(source, timeout=2, phrase_time_limit=2.5)
                t0 = now_ms()
                text = r.recognize_google(audio, language="ko-KR")
                t1 = now_ms()
                with self.stt_lock:
                    self.stt_queue.append((t0,t1,text))
            except Exception:
                pass

    def _consume_stt(self):
        if not self.trial_active:
            return
        if self.cond not in [COND_STT, COND_MULTI]:
            return
        with self.stt_lock:
            if not self.stt_queue:
                return
            t0,t1,text = self.stt_queue.pop(0)

        self.stt_last_text = text
        self._log("stt_text", t_listen_ms=t0, t_done_ms=t1, text=text)

        if self.mode == MODE_A:
            parsed = self._parse_cell(text)
            if self.cond == COND_STT:
                if parsed:
                    self.hover_target = parsed
                    self.hover_since_ms = now_ms()
                    self._apply_hover_ui(parsed)
                    self.confirm_action(source="stt_only")
            else:
                if ("선택" in text) or ("클릭" in text) or ("확정" in text):
                    self.confirm_action(source="stt_confirm")
                elif parsed:
                    self.hover_target = parsed
                    self.hover_since_ms = now_ms()
                    self._apply_hover_ui(parsed)
        else:
            act = None
            if ("열어" in text) or ("오픈" in text) or ("open" in text.lower()):
                act = "open"
            if ("삭제" in text) or ("지워" in text):
                act = "delete"

            fname = self._parse_file(text)
            if self.cond == COND_STT:
                if fname:
                    self.hover_target = ("file", fname)
                    self.hover_since_ms = now_ms()
                    self._apply_hover_ui(self.hover_target)
                if act:
                    self.hover_target = ("action", act)
                    self.hover_since_ms = now_ms()
                    self._apply_hover_ui(self.hover_target)
                if fname and act:
                    self.confirm_action(source="stt_only")
            else:
                if act and (("실행" in text) or ("해" in text) or ("확정" in text) or ("클릭" in text)):
                    self.hover_target = ("action", act)
                    self.hover_since_ms = now_ms()
                    self._apply_hover_ui(self.hover_target)
                    self.confirm_action(source="stt_confirm")
                else:
                    if fname:
                        self.hover_target = ("file", fname)
                        self.hover_since_ms = now_ms()
                        self._apply_hover_ui(self.hover_target)
                    if act:
                        self.hover_target = ("action", act)
                        self.hover_since_ms = now_ms()
                        self._apply_hover_ui(self.hover_target)

    def _parse_cell(self, text):
        t = text.strip().upper().replace(" ", "")
        for key in ["A1","A2","A3","B1","B2","B3","C1","C2","C3"]:
            if key in t:
                return key
        return None

    def _parse_file(self, text):
        t = text.strip().lower().replace(" ", "")
        candidates = [f for f,info in self.file_items.items() if info["alive"]]
        for f in candidates:
            if f.lower().replace("_","") in t:
                return f
        for f in candidates:
            base = f.split(".")[0].lower().replace("_","")
            if base in t:
                return f
        return None

    def _tick(self):
        self._set_status()
        self._consume_stt()
        self.root.after(60, self._tick)

    def run(self):
        self.root.after(200, self._build_scene)
        self.root.mainloop()

if __name__ == "__main__":
    App().run()
