import prodigy
from prodigy.util import set_hashes
import spacy

print("Loaded bullying_workflow v0.0.30")

@prodigy.recipe(
    "bullying-workflow",
    dataset=("Dataset to save to", "positional", None, str),
    source=("Source data (txt or jsonl)", "positional", None, str),
    loader=("Loader: txt or jsonl", "option", "l", str),
    model=("spaCy model for spans tokenization (default: en_core_web_sm)", "option", "m", str)
)
def bullying_workflow(dataset: str, source: str, loader: str = "txt", model: str = "en_core_web_sm"):
    # Load tokenizer once, like spans.manual does
    nlp = spacy.load(model)

    # Stream: txt or jsonl
    if loader.lower() == "txt":
        def stream_from_txt(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    txt = line.strip()
                    if txt:
                        yield {"text": txt}
        base_stream = stream_from_txt(source)
    else:
        from prodigy.components.loaders import JSONL
        base_stream = JSONL(source)

    # Add tokens so spans_manual can highlight text
    def add_tokens(stream):
        for eg in stream:
            if eg.get("text"):
                doc = nlp.make_doc(eg["text"])
                eg["tokens"] = [
                    {"text": t.text, "start": t.idx, "end": t.idx + len(t.text), "id": i}
                    for i, t in enumerate(doc)
                ]
            yield eg

    def with_hashes(s):
        for eg in s:
            eg.setdefault("step", 1)  # start at step 1
            yield set_hashes(eg)

    # Server validation
    def validate_answer(eg):
        b = eg.get("binary")
        r = eg.get("role")
        sev = eg.get("severity")
        spans = eg.get("spans") or []

        if b == "nonbullying":
            if r or sev or spans:
                return (False, "Non-cyberbullying selected. Do not annotate role, severity, or spans.")
            return True

        if b not in ("bullying", "nonbullying"):
            return (False, "Step 1: choose Cyberbullying or Non-cyberbullying.")
        if r not in ("bully", "victim", "bystander"):
            return (False, "Step 2: choose exactly one role.")
        if sev not in ("high", "medium", "low"):
            return (False, "Step 3: choose exactly one severity.")
        if not spans:
            return (False, "Step 4: highlight at least one span and assign a category.")
        return True

    return {
        "dataset": dataset,
        "stream": with_hashes(add_tokens(base_stream)),
        "view_id": "blocks",
        "config": {
            "blocks": [
                # Text first (copyable)
                {"view_id": "text", "text_field": "text"},

                # Wizard UI (non-copyable, ultra-compact)
                {
                    "view_id": "html",
                    "html_template": r"""
                        <div id="cj-wizard" inert aria-hidden="true" style="font-family:system-ui;line-height:1.1">
                        <div id="panel-1" data-step="1">
                            <h3>Step 1 — Binary</h3>
                            <label><input type="radio" name="binary" value="bullying"> Cyberbullying</label>
                            <label style="margin-left:.18rem"><input type="radio" name="binary" value="nonbullying"> Non-cyberbullying</label>
                            <div class="btn-row"><button id="next-1" class="prodigy-button prodigy-button--primary" type="button">Next</button></div>
                        </div>

                        <div id="panel-2" data-step="2" style="display:none">
                            <h3>Step 2 — Role</h3>
                            <label><input type="radio" name="role" value="bully"> Bully</label>
                            <label style="margin-left:.18rem"><input type="radio" name="role" value="victim"> Victim</label>
                            <label style="margin-left:.18rem"><input type="radio" name="role" value="bystander"> Bystander</label>
                            <div class="btn-row"><button id="next-2" class="prodigy-button prodigy-button--primary" type="button">Next</button></div>
                        </div>

                        <div id="panel-3" data-step="3" style="display:none">
                            <h3>Step 3 — Severity</h3>
                            <label><input type="radio" name="severity" value="high"> High</label>
                            <label style="margin-left:.18rem"><input type="radio" name="severity" value="medium"> Medium</label>
                            <label style="margin-left:.18rem"><input type="radio" name="severity" value="low"> Low</label>
                            <div class="btn-row"><button id="next-3" class="prodigy-button prodigy-button--primary" type="button">Next</button></div>
                        </div>

                        <div id="hint" aria-hidden="true"></div>
                        </div>
                    """
                },

                # Spans manual — always mounted, but CSS hides it except on step 4
                {
                    "view_id": "spans_manual",
                    "text_field": "text",
                    "labels": ["VISUAL", "VERBAL", "EXCLUSION", "OTHER"]
                }
            ],

            "javascript": r"""
                // ===== Helpers =====
                const q  = (s)=>document.querySelector(s);
                const qa = (s)=>Array.from(document.querySelectorAll(s));
                const getTask = ()=> (window.prodigy && window.prodigy.content) || {};
                const update  = (fields)=>{ const t=getTask(); Object.assign(t, fields); window.prodigy.update({task:t}); };

                const panels = {
                1: ()=>q('#panel-1'),
                2: ()=>q('#panel-2'),
                3: ()=>q('#panel-3')
                };

                function setRootStep(step){
                const root = document.documentElement;
                root.classList.remove('cj-step-1','cj-step-2','cj-step-3','cj-step-4');
                root.classList.add(`cj-step-${step}`);
                }

                function showStep(n){
                // Show the right wizard panel, keep wizard itself visible for consistent layout
                Object.entries(panels).forEach(([k,get])=>{
                    const el = get(); if (el) el.style.display = (parseInt(k,10)===n)? '' : 'none';
                });

                // Footer buttons only on step 4 (via CSS class as well)
                setRootStep(n);
                }

                // Mirror radios from task + switch step
                document.addEventListener('prodigyupdate', ev=>{
                const t = (ev.detail && ev.detail.task) || {};
                const step = parseInt(t.step || 1, 10);

                qa('input[name="binary"]').forEach(x => x.checked = (t.binary === x.value));
                qa('input[name="role"]').forEach(x => x.checked = (t.role === x.value));
                qa('input[name="severity"]').forEach(x => x.checked = (t.severity === x.value));

                showStep(step);
                });

                // Initial: start at step 1 class (spans hidden by CSS)
                document.addEventListener('DOMContentLoaded', ()=>{
                setRootStep(1);
                });

                // Next buttons
                document.addEventListener('click', (e)=>{
                const id = e.target && e.target.id;
                if (!id) return;

                if (id === 'next-1'){
                    const b = qa('input[name="binary"]').find(x=>x.checked)?.value;
                    if (!b){ alert("Choose Cyberbullying or Non-cyberbullying."); return; }
                    if (b === 'nonbullying'){
                    update({ binary:'nonbullying', role:null, severity:null });
                    window.prodigy.answer('accept');
                    } else {
                    update({ binary:'bullying', step:2 });
                    const hint = q('#hint'); if (hint) hint.textContent = "Pick a role to continue.";
                    setRootStep(2);
                    }
                }

                if (id === 'next-2'){
                    const r = qa('input[name="role"]').find(x=>x.checked)?.value;
                    if (!r){ alert("Choose a role to continue."); return; }
                    update({ role:r, step:3 });
                    const hint = q('#hint'); if (hint) hint.textContent = "Pick a severity to continue.";
                    setRootStep(3);
                }

                if (id === 'next-3'){
                    const s = qa('input[name="severity"]').find(x=>x.checked)?.value;
                    if (!s){ alert("Choose a severity to continue."); return; }
                    update({ severity:s, step:4 });
                    const hint = q('#hint'); if (hint) hint.textContent = "Highlight spans and click Accept.";
                    setRootStep(4);
                    // ensure the token overlay measures correctly once it becomes visible
                    setTimeout(()=>window.dispatchEvent(new Event('resize')), 0);
                }
                }, true);
            """,

            "global_css": """
                /* ===== Tight global rhythm ===== */
                .prodigy-card { padding: .2rem .35rem !important; }
                .prodigy-content { padding-top: .08rem !important; padding-bottom: .08rem !important; }
                .prodigy-content > .prodigy-block { margin: .04rem 0 !important; }

                /* Text view: compact lines, copyable */
                .prodigy-view-id-text { line-height: 1.15 !important; }

                /* ===== Wizard: ultra-compact and non-copyable ===== */
                #cj-wizard, #cj-wizard * { -webkit-user-select: none; user-select: none; }
                #cj-wizard { display: inline-block; vertical-align: top; }
                #cj-wizard h3 { margin: .06rem 0 .08rem 0 !important; font-size: .95rem !important; line-height:1.05 !important; }
                #cj-wizard label { margin: 0 .18rem 0 0 !important; line-height:1.05 !important; }
                #cj-wizard input[type="radio"] { transform: translateY(1px); }
                #cj-wizard .btn-row { display: inline; }
                #cj-wizard .prodigy-button { margin: 0 .06rem !important; padding: .12rem .28rem !important; }

                #hint { margin: .06rem 0 0 0 !important; font-size: .85rem !important; color:#666 !important;
                        -webkit-user-select: none; user-select: none; }

                /* ===== Spans: hidden by default, only visible on step 4 ===== */
                .prodigy-view-id-spans_manual { display: none !important; margin-top: .12rem !important; }
                .cj-step-4 .prodigy-view-id-spans_manual { display: block !important; }

                /* NER block tight */
                .prodigy-ner { line-height: 1.08 !important; }
                .prodigy-ner .prodigy-spans { margin-top: .08rem !important; }

                /* Footer buttons hidden except step 4 */
                .prodigy-buttons { display: none !important; margin-top: .12rem !important; }
                .cj-step-4 .prodigy-buttons { display: flex !important; }
            """
        },
        "validate_answer": validate_answer
    }