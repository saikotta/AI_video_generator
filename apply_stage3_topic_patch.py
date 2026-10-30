"""
Makes ONE small edit to app/services/pipeline_orchestrator.py so the visual planner
receives the video topic (stage 3). Safe to run: it backs the file up first, only edits if the
exact original text is found once, and does nothing if it was already applied.

Run from C:\\AiVideoGenerator:
    .\\.venv\\Scripts\\python.exe apply_stage3_topic_patch.py
"""
import shutil, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else r"app\services\pipeline_orchestrator.py"

OLD = '''            provider = self.get_script_provider()
            # Bounded parallel execution reusing existing orchestrator function
            specs = await process_visual_specifications_parallel(
                provider=provider, scenes=scenes, max_concurrency=settings.GROQ_MAX_CONCURRENCY
            )'''
NEW = '''            provider = self.get_script_provider()
            async with AsyncSessionLocal() as _session:
                _res = await _session.execute(select(Project.topic).where(Project.id == project_id))
                project_topic = _res.scalar() or ""
            # Bounded parallel execution reusing existing orchestrator function
            specs = await process_visual_specifications_parallel(
                provider=provider, scenes=scenes, max_concurrency=settings.GROQ_MAX_CONCURRENCY,
                topic=project_topic,
            )'''

with open(PATH, "r", encoding="utf-8", newline="") as f:
    text = f.read()
crlf = "\r\n" in text
old = OLD.replace("\n", "\r\n") if crlf else OLD
new = NEW.replace("\n", "\r\n") if crlf else NEW

if "topic=project_topic" in text:
    print("Already applied - nothing to do.")
    sys.exit(0)
if text.count(old) != 1:
    print(f"NOT APPLIED: expected the original snippet exactly once, found {text.count(old)}. "
          f"Upload {PATH} and I will adapt the edit.")
    sys.exit(1)

shutil.copyfile(PATH, PATH + ".bak_before_topic_patch")
with open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(text.replace(old, new))
print(f"Applied. Backup saved as {PATH}.bak_before_topic_patch")
