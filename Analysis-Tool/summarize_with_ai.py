from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from ai_providers import build_provider
from analyzer import LogEvent, analyze


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "analyzer_output" / "top-12-messages"
DEFAULT_OUTPUT = SCRIPT_DIR / "analyzer_output" / "ai-summary.md"


DEFAULT_PROMPT = """You are summarizing the most frequent Jenkins log message templates.

Read the ranked template list from stdin and produce a concise operator-facing summary.

Requirements:
- Start with a 2-3 sentence overview of the dominant patterns.
- Then provide a short bullet list of the most important recurring issues.
- Call out which items look noisy or expected versus which may need investigation.
- End with 2-4 concrete next steps.
- Use plain Markdown.
- Do not repeat the full raw list back unless necessary.
 - Ground claims in the provided evidence snippets when available.
"""

SEVERE_LEVELS = {"SEVERE", "FATAL", "ERROR"}


# Inputs: none (reads CLI arguments from sys.argv).
# Output: configured argparse.ArgumentParser instance.
# Options: supports --input, --output, --model, --provider, retrieval-related flags, and prompt flags.
# Pre: module constants are available.
# Post: parser contains all supported command-line options for this script.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize analyzer top-message output using a pluggable AI provider."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to top-N message file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write the summary markdown (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model name for the selected provider.",
    )
    parser.add_argument(
        "--provider",
        default="copilot",
        choices=["copilot", "bob"],
        help="AI provider to use (default: copilot for Microsoft Copilot).",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=SCRIPT_DIR.parent / "sample-logs" / "var" / "log" / "jenkins",
        help="Path to a log file or directory used for template-first retrieval evidence.",
    )
    parser.add_argument(
        "--disable-retrieval",
        action="store_true",
        help="Disable template-first retrieval and summarize only the top-N template file.",
    )
    parser.add_argument(
        "--retrieval-template-budget",
        type=int,
        default=12,
        help="Maximum number of templates to include in retrieval evidence.",
    )
    parser.add_argument(
        "--retrieval-examples-per-template",
        type=int,
        default=2,
        help="Maximum event examples per selected template.",
    )
    parser.add_argument(
        "--retrieval-rare-threshold",
        type=int,
        default=2,
        help="Treat templates with count <= this threshold as rare.",
    )
    parser.add_argument(
        "--retrieval-max-files",
        type=int,
        default=6,
        help="Maximum number of log files to parse for retrieval evidence.",
    )
    parser.add_argument(
        "--extra-instructions",
        default="",
        help="Extra text appended to the default summarization prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompt and input path without invoking an AI provider.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the full prompt (system prompt + retrieval-augmented input) sent to the AI provider.",
    )
    return parser


# Inputs: extra_instructions text appended by caller.
# Output: finalized system prompt string.
# Options: appends extra instructions only when non-empty after strip().
# Pre: DEFAULT_PROMPT is defined.
# Post: returned prompt is never surrounded by leading/trailing blank space from DEFAULT_PROMPT.
def build_prompt(extra_instructions: str) -> str:
    prompt = DEFAULT_PROMPT.strip()
    if extra_instructions.strip():
        prompt = f"{prompt}\n\nAdditional instructions:\n{extra_instructions.strip()}"
    return prompt


# Inputs: path to the analyzer top-N templates file.
# Output: non-empty file content as a string.
# Options: none.
# Pre: input file exists and is readable.
# Post: returns stripped text; raises FileNotFoundError/ValueError when file is missing or empty.
def load_input(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run the analyzer first so it generates top-N message output."
        )
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(
            f"Input file is empty: {path}\n"
            "If this is a top-N template file, make sure drain3 is available when running the analyzer."
        )
    return content


# Inputs: log_root file/directory and max_files limit.
# Output: list of candidate log file paths.
# Options: prefers files matching jenkins.log* and falls back to any non-gzip files.
# Pre: max_files should be positive.
# Post: returns at most max_files paths; returns [] when log_root is invalid or no matches are found.
def discover_log_files(log_root: Path, max_files: int) -> list[Path]:
    if log_root.is_file():
        return [log_root]
    if not log_root.exists() or not log_root.is_dir():
        return []

    candidates = [
        p for p in sorted(log_root.rglob("jenkins.log*"))
        if p.is_file() and not p.name.endswith(".gz")
    ]
    if not candidates:
        # Fall back to non-gzip files if naming differs.
        candidates = [
            p for p in sorted(log_root.rglob("*"))
            if p.is_file() and not p.name.endswith(".gz")
        ]
    return candidates[:max_files]


# Inputs: iterable of log file paths.
# Output: flattened list of parsed LogEvent objects.
# Options: skips files that raise parse errors.
# Pre: analyzer.analyze is importable and file paths are accessible.
# Post: each returned event has source_path attached for retrieval evidence context.
def load_events(paths: Iterable[Path]) -> list[LogEvent]:
    events: list[LogEvent] = []
    for path in paths:
        try:
            file_events = analyze(path)
        except Exception:
            continue
        for ev in file_events:
            # Attach source for evidence output without changing core model.
            ev.source_path = str(path)
        events.extend(file_events)
    return events


# Inputs: single LogEvent instance.
# Output: stable grouping key string for retrieval buckets.
# Options: chooses template_id first, then template text, then message prefix fallback.
# Pre: event has at least one of template_id/template/message fields.
# Post: returns a non-empty key suitable for dictionary grouping.
def _template_key(ev: LogEvent) -> str:
    if ev.template_id is not None:
        return f"template_id:{ev.template_id}"
    if ev.template:
        return f"template:{ev.template}"
    return f"message:{ev.message[:120]}"


# Inputs: single LogEvent instance.
# Output: tuple used for sorting events by severity and timestamp.
# Options: severe levels are defined by SEVERE_LEVELS.
# Pre: event fields may be None.
# Post: severe events rank before non-severe events when sorting reverse=True.
def _event_sort_key(ev: LogEvent):
    level = (ev.level or "").upper()
    sev_rank = 1 if level in SEVERE_LEVELS else 0
    ts = ev.timestamp.isoformat() if ev.timestamp else ""
    return (sev_rank, ts)


# Inputs: parsed events list and retrieval tuning values.
# Output: markdown retrieval evidence block for prompt augmentation.
# Options: controls template selection via template budget, per-template examples, and rare threshold.
# Pre: events are parsed from logs and can include optional metadata.
# Post: returns either a detailed evidence section or "No retrieval evidence available." when empty.
def build_retrieval_context(
    events: list[LogEvent],
    template_budget: int,
    examples_per_template: int,
    rare_threshold: int,
) -> str:
    if not events:
        return "No retrieval evidence available."

    groups: dict[str, list[LogEvent]] = defaultdict(list)
    for ev in events:
        groups[_template_key(ev)].append(ev)

    counts = Counter({k: len(v) for k, v in groups.items()})
    severe_keys = [
        key for key, items in groups.items()
        if any(((ev.level or "").upper() in SEVERE_LEVELS) for ev in items)
    ]
    severe_keys.sort(key=lambda k: counts[k], reverse=True)

    rare_keys = [k for k, c in counts.items() if c <= rare_threshold]
    rare_keys.sort(key=lambda k: counts[k])

    selected: list[str] = []
    for key in severe_keys + rare_keys:
        if key not in selected:
            selected.append(key)
        if len(selected) >= template_budget:
            break

    # If severe+rare is small, fill with globally frequent templates.
    if len(selected) < template_budget:
        for key, _ in counts.most_common():
            if key not in selected:
                selected.append(key)
            if len(selected) >= template_budget:
                break

    lines = [
        "Template-first retrieval evidence:",
        f"- events_parsed: {len(events)}",
        f"- templates_seen: {len(groups)}",
        f"- selected_templates: {len(selected)}",
        "",
    ]

    for key in selected:
        bucket = groups[key]
        exemplar = bucket[0]
        levels = Counter((ev.level or "UNKNOWN") for ev in bucket)
        level_summary = ", ".join(f"{lvl}:{cnt}" for lvl, cnt in levels.most_common(3))
        template_text = exemplar.template or exemplar.message
        lines.append(f"## {key} count={len(bucket)} levels=[{level_summary}]")
        lines.append(f"template: {template_text}")

        ranked = sorted(bucket, key=_event_sort_key, reverse=True)
        for ev in ranked[:examples_per_template]:
            source = getattr(ev, "source_path", "unknown")
            lines.append(
                f"- [{ev.level or 'UNKNOWN'}] {ev.timestamp or 'no-ts'} {source}:{ev.line_start}-{ev.line_end}"
            )
            lines.append(f"  message: {ev.message}")
            if ev.stack_trace:
                stack_lines = ev.stack_trace.splitlines()[:2]
                for s in stack_lines:
                    lines.append(f"  stack: {s}")
        lines.append("")

    return "\n".join(lines).strip()


# Inputs: CLI arguments and optional environment/provider configuration.
# Output: process exit code (0 success, 1 failure).
# Options: supports dry-run/show-prompt and optional retrieval augmentation.
# Pre: input file and provider credentials are available unless running dry-run.
# Post: writes summary markdown to output path on success.
def main() -> int:
    args = build_parser().parse_args()
    prompt = build_prompt(args.extra_instructions)

    try:
        input_text = load_input(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    final_input = input_text
    if not args.disable_retrieval:
        files = discover_log_files(args.log_root, args.retrieval_max_files)
        events = load_events(files)
        retrieval_block = build_retrieval_context(
            events,
            template_budget=args.retrieval_template_budget,
            examples_per_template=args.retrieval_examples_per_template,
            rare_threshold=args.retrieval_rare_threshold,
        )
        if files:
            final_input = (
                f"{input_text}\n\n"
                "---\n"
                "Use this additional retrieval evidence from raw logs.\n\n"
                f"{retrieval_block}"
            )
        else:
            print(
                f"No log files found under {args.log_root}; running without retrieval evidence.",
                file=sys.stderr,
            )

    if args.dry_run or args.show_prompt:
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Provider: {args.provider}")
        print(f"Log root: {args.log_root}")
        print(f"Retrieval enabled: {not args.disable_retrieval}")
        print("\n=== SYSTEM PROMPT ===\n")
        print(prompt)
        print("\n=== USER CONTENT (input + retrieval evidence) ===\n")
        print(final_input)
        if args.dry_run:
            return 0

    try:
        provider = build_provider(args.provider)
        summary = provider.summarize(final_input, prompt, args.model)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary + "\n", encoding="utf-8")
    print(f"Wrote summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())