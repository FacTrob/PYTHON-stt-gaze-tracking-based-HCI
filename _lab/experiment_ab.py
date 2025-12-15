import os,time,json,csv,datetime,threading
import tkinter as tk
from tkinter import ttk
from pynput import keyboard

LOG_DIR="logs"
os.makedirs(LOG_DIR,exist_ok=True)

SENTENCE="정의로운 과학으로 창의적인 도전을\n바른 인성을 갖춘 노벨과학인재 육성"

COND_STT_ONLY="C1_STT_ONLY"
COND_STT_GAZE="C2_STT_GAZE_YESNO"
CONDS=[COND_STT_ONLY,COND_STT_GAZE]

def now_ms(): return int(time.time()*1000)

class App:
    def __init__(self):
        self.root=tk.Tk()
        self.root.title("R&E Experiment Controller (STT + Gaze Yes/No)")
        self.root.geometry("980x640")

        self.cond=COND_STT_GAZE
        self.trial_id=0
        self.trial_active=False
        self.phase="idle"  # idle, wait_stt, wait_yesno, do_task, done
        self.t_start=None

        self.events=[]
        self.lock=threading.Lock()

        self._ui()
        self._listener=keyboard.Listener(on_press=self._on_key)
        self._listener.daemon=True
        self._listener.start()
        self._tick()

    def _ui(self):
        top=ttk.Frame(self.root); top.pack(fill="x", padx=10, pady=10)
        self.lbl=ttk.Label(top, text=""); self.lbl.pack(side="left")
        ttk.Button(top, text="Next Condition", command=self.next_cond).pack(side="right")

        info=ttk.Frame(self.root); info.pack(fill="x", padx=10, pady=6)
        self.lbl2=ttk.Label(info, text="", justify="left"); self.lbl2.pack(anchor="w")

        mid=ttk.Frame(self.root); mid.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt=tk.Text(mid, height=26)
        self.txt.pack(fill="both", expand=True)

        guide=(
            "Hotkeys (global)\n"
            "F8 : Start/Stop trial (Stop => fail)\n"
            "F9 : STT recognized '메모장 실행'\n"
            "F10: YES (Left gaze)\n"
            "F11: NO  (Right gaze)\n"
            "F12: Task done (sentence typed)\n\n"
            "Task sentence:\n"
            +SENTENCE+"\n"
        )
        self.txt.insert("end", guide+"\n")
        self.txt.see("end")

    def _status(self):
        t=f"COND: {self.cond} | TRIAL: {self.trial_id if self.trial_active else '-'} | PHASE: {self.phase}"
        self.lbl.config(text=t)
        d=(
            "Procedure\n"
            "1) Press F8 to start\n"
            "2) Participant uses external STT to say: '메모장 실행'\n"
            "3) When STT program recognized, press/send F9\n"
            "4) If condition is STT+Gaze: decide with F10(YES)/F11(NO)\n"
            "5) If YES: Notepad is opened by participant or operator; participant types the sentence\n"
            "6) When typing finished, press F12\n"
        )
        self.lbl2.config(text=d)

    def _log(self, typ, **data):
        ev={"t_ms":now_ms(),"type":typ,"cond":self.cond,"trial":self.trial_id,**data}
        self.events.append(ev)
        self.txt.insert("end", json.dumps(ev, ensure_ascii=False)+"\n")
        self.txt.see("end")

    def next_cond(self):
        if self.trial_active: return
        i=CONDS.index(self.cond)
        self.cond=CONDS[(i+1)%len(CONDS)]
        self._log("cond_change", cond=self.cond)
        self._status()

    def toggle_trial(self):
        if not self.trial_active:
            self.start_trial()
        else:
            self.end_trial(False,"manual_stop")

    def start_trial(self):
        self.trial_id+=1
        self.trial_active=True
        self.phase="wait_stt"
        self.t_start=now_ms()
        self.events=[]
        self._log("trial_start")
        self._status()

    def end_trial(self, success, reason):
        if not self.trial_active: return
        t_end=now_ms()
        total_s=(t_end-self.t_start)/1000.0 if self.t_start else None
        self._log("trial_end", success=bool(success), reason=reason, total_s=total_s)
        self.trial_active=False
        self.phase="idle"
        self._status()
        self.save_log()

    def save_log(self):
        if not self.events: return
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base=f"{ts}_trial{self.trial_id}_{self.cond}"
        j=os.path.join(LOG_DIR, base+".jsonl")
        c=os.path.join(LOG_DIR, base+".csv")

        with open(j,"w",encoding="utf-8") as f:
            for ev in self.events:
                f.write(json.dumps(ev, ensure_ascii=False)+"\n")

        keys=sorted({k for ev in self.events for k in ev.keys()})
        with open(c,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for ev in self.events:
                w.writerow(ev)

        self.txt.insert("end", f'LOG_SAVED jsonl="{j}" csv="{c}"\n')
        self.txt.see("end")

    def stt_recognized(self):
        if not self.trial_active: return
        if self.phase!="wait_stt": 
            self._log("stt_ignored", reason="phase_not_wait_stt", phase=self.phase); 
            return
        self._log("stt_ok", cmd="run_notepad")
        if self.cond==COND_STT_ONLY:
            self._log("auto_yes", why="stt_only_condition")
            self.phase="do_task"
            self._log("task_begin", action="type_sentence")
        else:
            self.phase="wait_yesno"
            self._log("ask_yesno", mapping="F10=YES(left) F11=NO(right)")
        self._status()

    def gaze_yes(self):
        if not self.trial_active: return
        if self.phase!="wait_yesno":
            self._log("yes_ignored", reason="phase_not_wait_yesno", phase=self.phase)
            return
        self._log("decision", yes=True)
        self.phase="do_task"
        self._log("task_begin", action="type_sentence")
        self._status()

    def gaze_no(self):
        if not self.trial_active: return
        if self.phase!="wait_yesno":
            self._log("no_ignored", reason="phase_not_wait_yesno", phase=self.phase)
            return
        self._log("decision", yes=False)
        self.end_trial(True,"user_no")

    def task_done(self):
        if not self.trial_active: return
        if self.phase!="do_task":
            self._log("done_ignored", reason="phase_not_do_task", phase=self.phase)
            return
        self._log("task_done", typed=True)
        self.end_trial(True,"ok")

    def _on_key(self, key):
        try:
            if key==keyboard.Key.f8:
                with self.lock: self.toggle_trial()
            elif key==keyboard.Key.f9:
                with self.lock: self.stt_recognized()
            elif key==keyboard.Key.f10:
                with self.lock: self.gaze_yes()
            elif key==keyboard.Key.f11:
                with self.lock: self.gaze_no()
            elif key==keyboard.Key.f12:
                with self.lock: self.task_done()
        except Exception:
            pass

    def _tick(self):
        self._status()
        self.root.after(200, self._tick)

    def run(self):
        self.root.mainloop()

if __name__=="__main__":
    App().run()
