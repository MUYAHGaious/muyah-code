"""Lessons: what MUYAH-CODE learned from its own successes and failures.

Lifecycle
  1. Signals are collected during a turn (tool errors, errors that were later fixed, loops, step limits)
     and from the user (/good, /bad, corrections like "no, that's wrong").
  2. When a turn has meaningful signals, the Reflector asks the model for 0-3 generalizable lessons.
  3. Lessons are stored as JSONL: ~/.muyah/lessons.jsonl (global) and <project>/.muyah/lessons.jsonl.
     Near-duplicates are merged (reinforced) instead of piling up.
  4. Each new prompt retrieves the most relevant lessons (BM25 x usefulness score) into the system prompt.
  5. Outcomes feed back: lessons used in a turn the user accepts gain score; lessons used in a turn the
     user corrects or marks /bad lose score. Consistently unhelpful lessons are pruned.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from muyah_code.agent.context import render_transcript

STOPWORDS = set("""a an the and or but if then else of to in on at for with by from as is are was were be been being
it its this that these those i you he she we they me my your our their do does did done not no yes can could should
would will shall may might must have has had having so such than too very just also into out up down over under
about after before again further once here there when where why how all any both each few more most other some own
same only what which who whom please make use using used get got run fix add file files code change update create
write need want help thing things now new let lets try work working""".split())
TOKEN_RE = re.compile(r"[a-z0-9_][a-z0-9_.\-]*[a-z0-9_]|[a-z0-9]")
CORRECTION_RE = re.compile(
    r"^\s*(no\b|nope\b|wrong\b|that'?s (not|wrong)|that is (not|wrong)|not what|don'?t\b|do not\b|stop\b|instead\b|"
    r"why did you|you (didn'?t|did not|forgot|broke|missed|should have)|it (still )?(doesn'?t|does not|fails|failed)|"
    r"still (broken|failing|not working)|this is (wrong|broken|not))",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def is_correction(prompt: str) -> bool:
    return bool(CORRECTION_RE.search(prompt or ""))


@dataclass
class Lesson:
    trigger: str
    lesson: str
    tags: list[str] = field(default_factory=list)
    scope: str = "project"  # project | global
    source: str = "reflection"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created: float = field(default_factory=time.time)
    uses: int = 0
    wins: int = 0
    losses: int = 0
    reinforced: int = 0

    @property
    def score(self) -> float:
        """Laplace-smoothed usefulness in (0, 1); reinforcement counts as weak evidence of value."""
        return (self.wins + 0.5 * self.reinforced + 1) / (self.wins + 0.5 * self.reinforced + self.losses + 2)

    def text(self) -> str:
        return f"{self.trigger} {self.lesson} {' '.join(self.tags)}"

    def render(self) -> str:
        return f"- When {self.trigger.rstrip('.')}: {self.lesson}"


def _similarity(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class LessonStore:
    DEDUPE_THRESHOLD = 0.55

    def __init__(self, global_path: Path, project_path: Path | None):
        self.paths = {"global": global_path, "project": project_path}
        self.lessons: list[Lesson] = []
        self.load()

    def load(self) -> None:
        self.lessons = []
        for scope, path in self.paths.items():
            if path is None or not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    data["scope"] = scope
                    known = {k: v for k, v in data.items() if k in Lesson.__dataclass_fields__}
                    self.lessons.append(Lesson(**known))
                except (json.JSONDecodeError, TypeError):
                    continue

    def save(self) -> None:
        for scope, path in self.paths.items():
            if path is None:
                continue
            items = [x for x in self.lessons if x.scope == scope]
            if not items and not path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(asdict(x), ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
            tmp.replace(path)

    def add(self, lesson: Lesson) -> tuple[Lesson, bool]:
        """Add, or reinforce a near-duplicate. Returns (stored lesson, merged?)."""
        if lesson.scope == "project" and self.paths["project"] is None:
            lesson.scope = "global"
        best, best_sim = None, 0.0
        for existing in self.lessons:
            sim = _similarity(existing.text(), lesson.text())
            if sim > best_sim:
                best, best_sim = existing, sim
        if best is not None and best_sim >= self.DEDUPE_THRESHOLD:
            best.reinforced += 1
            if len(lesson.lesson) > len(best.lesson) and best.losses > best.wins:
                best.lesson = lesson.lesson  # replace a lesson that was not working with the new phrasing
            best.tags = sorted(set(best.tags) | set(lesson.tags))[:8]
            self.save()
            return best, True
        self.lessons.append(lesson)
        self.save()
        return lesson, False

    def remove(self, lesson_id: str) -> bool:
        before = len(self.lessons)
        self.lessons = [x for x in self.lessons if not x.id.startswith(lesson_id)]
        if len(self.lessons) != before:
            self.save()
            return True
        return False

    def get(self, lesson_id: str) -> Lesson | None:
        return next((x for x in self.lessons if x.id.startswith(lesson_id)), None)

    def search(self, query: str, k: int = 5, min_relevance: float = 0.5) -> list[Lesson]:
        """BM25 over trigger+lesson+tags, weighted by usefulness score."""
        q = tokenize(query)
        if not q or not self.lessons:
            return []
        docs = [tokenize(x.text()) for x in self.lessons]
        n = len(docs)
        avgdl = sum(len(d) for d in docs) / n or 1
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        k1, b = 1.4, 0.75
        scored = []
        for lesson, d in zip(self.lessons, docs, strict=True):
            tf = Counter(d)
            s = 0.0
            for term in set(q):
                if term not in tf:
                    continue
                # floor keeps small stores useful: with 1-3 lessons plain BM25 idf is ~0
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5)) + 0.5
                s += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(d) / avgdl))
            if s >= min_relevance:
                scored.append((s * (0.5 + lesson.score), lesson))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x for _, x in scored[:k]]

    def record_use(self, lessons: list[Lesson]) -> None:
        for x in lessons:
            x.uses += 1
        if lessons:
            self.save()

    def record_outcome(self, lesson_ids: list[str], success: bool) -> None:
        changed = False
        for x in self.lessons:
            if x.id in lesson_ids:
                if success:
                    x.wins += 1
                else:
                    x.losses += 1
                changed = True
        if changed:
            self.prune()
            self.save()

    def prune(self) -> list[Lesson]:
        dropped = [x for x in self.lessons if x.losses >= 3 and x.score < 0.3]
        self.lessons = [x for x in self.lessons if x not in dropped]
        return dropped


REFLECT_SYSTEM = """\
You are the self-improvement module of MUYAH-CODE, a coding agent. You study one episode of the agent's work \
and extract lessons that will make it better in FUTURE sessions.

A good lesson is:
- generalizable: it applies to future tasks, not just this one file or this one moment;
- actionable: it says what to do (or avoid) in a concrete situation;
- grounded: it is supported by what actually happened in the episode (an error, a fix, the user's feedback).

Do NOT record: one-off facts about a single file, things that went fine without incident, vague advice \
("be careful"), or anything already covered by the existing lessons listed below.

Respond with ONLY a JSON array (possibly empty) of at most 3 objects:
[{"trigger": "<the situation, e.g. 'running pytest on Windows in this project'>",
  "lesson": "<what to do, imperative, <= 200 chars>",
  "scope": "project" | "global",
  "tags": ["<2-5 keywords>"]}]
Use scope "project" for things specific to this codebase/environment, "global" for general coding know-how."""


class Reflector:
    def __init__(self, llm, store: LessonStore):
        self.llm = llm
        self.store = store

    @staticmethod
    def worth_reflecting(signals: list[dict], status: str) -> bool:
        kinds = [s.get("type") for s in signals]
        if any(k in ("recovered", "user_correction", "user_feedback", "loop") for k in kinds):
            return True
        if status in ("max_steps", "loop"):
            return True
        return kinds.count("tool_error") >= 3

    def reflect(self, prompt: str, signals: list[dict], transcript: list[dict], status: str) -> list[tuple[Lesson, bool]]:
        episode = render_transcript(transcript, per_message=800)
        if len(episode) > 9000:
            episode = episode[:2500] + "\n...\n" + episode[-6500:]
        related = self.store.search(prompt + " " + " ".join(str(s) for s in signals), k=8, min_relevance=0.2)
        existing = "\n".join(x.render() for x in related) or "(none)"
        sig_lines = "\n".join("- " + json.dumps(s, ensure_ascii=False)[:600] for s in signals[-12:]) or "(none)"
        user = (f"User request:\n{prompt[:1500]}\n\nOutcome status: {status}\n\nSignals:\n{sig_lines}\n\n"
                f"Existing lessons:\n{existing}\n\nEpisode transcript:\n{episode}")
        try:
            out = self.llm.chat([{"role": "system", "content": REFLECT_SYSTEM}, {"role": "user", "content": user}],
                                max_tokens=700, temperature=0, purpose="reflect")
        except Exception:
            return []
        source = next((s["type"] for s in signals if s.get("type") in ("user_correction", "user_feedback",
                                                                         "recovered")), "reflection")
        stored = []
        for item in parse_lessons(out.content):
            stored.append(self.store.add(Lesson(
                trigger=item["trigger"], lesson=item["lesson"], tags=item["tags"], scope=item["scope"], source=source,
            )))
        return stored


def parse_lessons(text: str) -> list[dict]:
    text = (text or "").strip()
    start = text.find("[")
    if start == -1:
        return []
    try:
        data, _ = json.JSONDecoder(strict=False).raw_decode(text[start:])
    except json.JSONDecodeError:
        return []
    out = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        trigger = " ".join(str(item.get("trigger", "")).split())[:200]
        lesson = " ".join(str(item.get("lesson", "")).split())[:300]
        if len(trigger) < 5 or len(lesson) < 10:
            continue
        scope = item.get("scope") if item.get("scope") in ("project", "global") else "project"
        tags = [str(t).lower()[:30] for t in (item.get("tags") or []) if str(t).strip()][:6]
        out.append({"trigger": trigger, "lesson": lesson, "scope": scope, "tags": tags})
    return out[:3]
