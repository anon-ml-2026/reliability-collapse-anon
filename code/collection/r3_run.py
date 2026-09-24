"""Paper 2 R3 — 官方直连 API 实验执行器

使用官方 API（非中转站），支持多 provider 路由（OpenAI / Anthropic / DeepSeek / GLM / Qwen / Mimo）。

Usage:
    # 试运行：验证配置和探针
    python run.py --dry-run

    # 只跑 2 个模型 × 前 4 条探针（快速测试）
    python run.py --model gpt-4o-mini --model deepseek-v3-0324 --count 4

    # 只跑 UNREG 领域
    python run.py --domain UNREG

    # 全量运行
    python run.py

    # 使用改写版探针（而非前缀版）
    python run.py --variants docs/probes_variants180.md

    # 限制探针数量 + 指定角色线
    python run.py --count 12 --role with-role --role boundary-role

匿名发布说明：本驱动依赖实验框架的 core.probe_loader / core.providers 模块与
config.yaml（provider 端点与密钥），这些不属于匿名发布内容；脚本收录用于说明
采集流程，全部已采集响应见 data/responses/，探针见 docs/probes_base60.md 与
docs/probes_variants180.md。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

# ── Path setup ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent  # repository root
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "paper2"))

from core.probe_loader import (
    ProbeSet,
    auto_parse_variants,
    summarize as summarize_probes,
)
from core.providers import (
    ProviderRegistry,
    call_model_api,
    find_and_load_env,
)

# ── Config dataclasses ─────────────────────────────────────────────────────

_FATAL_EXCEPTIONS = (AuthenticationError, BadRequestError)
_RETRYABLE_EXCEPTIONS = (APIConnectionError, RateLimitError)


@dataclass
class R3Config:
    experiment: dict
    models: List[dict]
    roles: Dict[str, Dict[str, str]]
    framings: List[str]
    probes: dict
    output: dict
    project_root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "R3Config":
        path = Path(path).resolve()
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        # config is at experiments/paper2/runs/r3/config.yaml → 5 levels to project root
        return cls(
            experiment=raw["experiment"],
            models=raw["models"],
            roles=raw["roles"],
            framings=raw["framings"],
            probes=raw["probes"],
            output=raw["output"],
            project_root=path.parent.parent.parent.parent.parent,
        )

    def resolve_path(self, relative: str) -> Path:
        return self.project_root / relative

    @property
    def model_ids(self) -> List[str]:
        return [m["id"] for m in self.models]

    @property
    def role_names(self) -> List[str]:
        return list(self.roles.keys())


# ── Trial spec ─────────────────────────────────────────────────────────────

@dataclass
class TrialSpec:
    trial_id: str
    probe_id: str
    domain: str
    framing: str
    role_line: str
    model_id: str
    model_label: str
    system_prompt: str
    user_message: str
    char_count: int

    @staticmethod
    def compute_id(probe_id: str, framing: str, role_line: str, model_id: str) -> str:
        key = f"{probe_id}|{framing}|{role_line}|{model_id}"
        return hashlib.md5(key.encode()).hexdigest()[:8]


# ── Trial builder ──────────────────────────────────────────────────────────

def resolve_system_prompt(roles: Dict[str, Dict[str, str]], role_name: str, domain: str) -> str:
    role = roles[role_name]
    return role.get(domain) or role.get("default", "你是AI助手。")


def build_trial_matrix(
    config: R3Config,
    probes: List[ProbeSet],
    *,
    model_filter: Optional[List[str]] = None,
    role_filter: Optional[List[str]] = None,
    framing_filter: Optional[List[str]] = None,
    domain_filter: Optional[List[str]] = None,
    probe_filter: Optional[List[str]] = None,
    max_per_domain: Optional[int] = None,
) -> List[TrialSpec]:
    models = [m for m in config.models if not model_filter or m["id"] in model_filter]
    roles = [r for r in config.role_names if not role_filter or r in role_filter]
    framings = [f for f in config.framings if not framing_filter or f in framing_filter]

    # Apply per-domain probe cap
    if max_per_domain is not None:
        domain_groups: Dict[str, List[ProbeSet]] = {}
        for p in probes:
            domain_groups.setdefault(p.domain, []).append(p)
        selected: List[ProbeSet] = []
        for dom in ["REG-H", "REG-S", "UNREG", "CTL"]:
            group = domain_groups.get(dom, [])
            selected.extend(group[:max_per_domain])
        probes = selected

    trials: List[TrialSpec] = []
    for probe in probes:
        if domain_filter and probe.domain not in domain_filter:
            continue
        if probe_filter and probe.probe_id not in probe_filter:
            continue

        for framing_name in framings:
            variant = probe.variants.get(framing_name)
            if not variant:
                continue
            for role_name in roles:
                sys_prompt = resolve_system_prompt(config.roles, role_name, probe.domain)
                for model in models:
                    tid = TrialSpec.compute_id(probe.probe_id, framing_name, role_name, model["id"])
                    trials.append(TrialSpec(
                        trial_id=tid,
                        probe_id=probe.probe_id,
                        domain=probe.domain,
                        framing=framing_name,
                        role_line=role_name,
                        model_id=model["id"],
                        model_label=model["label"],
                        system_prompt=sys_prompt,
                        user_message=variant.text,
                        char_count=variant.char_count,
                    ))
    return trials


def summarize_plan(trials: List[TrialSpec]) -> str:
    n = len(trials)
    models = sorted(set(t.model_id for t in trials))
    domains = sorted(set(t.domain for t in trials))
    framings = sorted(set(t.framing for t in trials))
    roles = sorted(set(t.role_line for t in trials))
    lines = [
        f"Trial plan: {n} total trials",
        f"  Models ({len(models)}): {', '.join(models)}",
        f"  Domains ({len(domains)}): {', '.join(domains)}",
        f"  Framings ({len(framings)}): {', '.join(framings)}",
        f"  Role lines ({len(roles)}): {', '.join(roles)}",
    ]
    for model in models:
        model_n = sum(1 for t in trials if t.model_id == model)
        lines.append(f"  {model}: {model_n} trials")
    return "\n".join(lines)


# ── R3 Executor ────────────────────────────────────────────────────────────

def _make_result(trial: TrialSpec) -> dict:
    return {
        "trial_id": trial.trial_id,
        "probe_id": trial.probe_id,
        "domain": trial.domain,
        "framing": trial.framing,
        "role_line": trial.role_line,
        "model_id": trial.model_id,
        "model_label": trial.model_label,
        "system_prompt": trial.system_prompt,
        "user_message": trial.user_message,
        "char_count": trial.char_count,
        "response": None,
        "response_length": None,
        "error": None,
        "retry_count": 0,
        "timestamp": "",
    }


class R3Executor:
    """Executes trials with multi-provider API routing, retry, and checkpointing."""

    def __init__(self, config: R3Config, registry: ProviderRegistry, output_dir: Path):
        self.config = config
        self.registry = registry
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        exp = config.experiment
        self._max_retries = exp["max_retries"]
        self._retry_base_delay = exp["retry_base_delay"]
        self._request_timeout = exp["request_timeout"]
        self._workers = exp["workers"]
        self._seed = exp["seed"]

        self._lock = threading.Lock()
        self._completed_ids: Set[str] = set()
        self._results: List[dict] = []
        self._last_flushed_idx: int = 0
        self._checkpoint_path = output_dir / "checkpoint.json"
        self._results_path = output_dir / "results.jsonl"
        self._interrupted = False

    def execute(self, trials: List[TrialSpec], resume: bool = True) -> List[dict]:
        if resume and self._checkpoint_path.exists():
            self._load_checkpoint()

        remaining = [t for t in trials if t.trial_id not in self._completed_ids]
        skipped = len(trials) - len(remaining)
        if skipped:
            print(f"Resuming: {skipped} completed, {len(remaining)} remaining")

        if not remaining:
            print("All trials completed.")
            return self._results

        rng = random.Random(self._seed)
        rng.shuffle(remaining)

        total = len(remaining)
        done = 0
        checkpoint_every = self.config.output["checkpoint_interval"]

        print(f"Starting {total} trials with {self._workers} workers...")
        start = time.time()

        try:
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                futures = {pool.submit(self._run_one, t): t for t in remaining}
                for future in as_completed(futures):
                    result = future.result()
                    done += 1
                    should_ckpt = False
                    with self._lock:
                        self._results.append(result)
                        self._completed_ids.add(result["trial_id"])
                        if done % checkpoint_every == 0:
                            should_ckpt = True
                    if should_ckpt:
                        self._save_checkpoint()
                        self._flush_results()
                        elapsed = time.time() - start
                        rate = done / elapsed if elapsed else 0
                        eta = (total - done) / rate if rate else 0
                        errors = sum(1 for r in self._results if r.get("error"))
                        print(f"[{done}/{total}] {done/total*100:.1f}% "
                              f"({rate:.2f} trials/s, ETA {eta:.0f}s, {errors} errors)")
        except KeyboardInterrupt:
            print("\nInterrupted — saving progress...")
            self._save_checkpoint()
            self._flush_results()
            self._interrupted = True
            raise
        finally:
            if not self._interrupted:
                self._save_checkpoint()
                self._save_results_full()

        elapsed = time.time() - start
        errors = [r for r in self._results if r.get("error")]
        print(f"Done: {done} trials in {elapsed:.0f}s ({done/elapsed:.2f} trials/s)")
        if errors:
            print(f"Errors: {len(errors)}/{len(self._results)}")
            for e in errors[:5]:
                print(f"  {e['trial_id']} ({e['model_id']}): {e['error']}")
        return self._results

    def _run_one(self, trial: TrialSpec) -> dict:
        exp = self.config.experiment
        result = _make_result(trial)
        last_error = None

        for attempt in range(self._max_retries):
            try:
                text = call_model_api(
                    self.registry,
                    trial.model_id,
                    trial.system_prompt,
                    trial.user_message,
                    max_tokens=exp["max_tokens"],
                    temperature=exp["temperature"],
                    seed=exp["seed"],
                )
                result["response"] = text
                result["response_length"] = len(text)
                result["retry_count"] = attempt
                result["timestamp"] = datetime.now(timezone.utc).isoformat()
                return result
            except _FATAL_EXCEPTIONS as e:
                last_error = str(e)
                break
            except _RETRYABLE_EXCEPTIONS as e:
                last_error = str(e)
                if attempt < self._max_retries - 1:
                    delay = self._retry_base_delay * (2 ** attempt) * (0.5 + random.random())
                    time.sleep(delay)
            except Exception as e:
                last_error = str(e)
                if attempt < self._max_retries - 1:
                    delay = self._retry_base_delay * (2 ** attempt) * (0.5 + random.random())
                    time.sleep(delay)

        result["error"] = last_error
        result["retry_count"] = self._max_retries
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        return result

    # ── Checkpoint I/O ─────────────────────────────────────────────────

    def _save_checkpoint(self):
        data = {
            "completed_ids": sorted(self._completed_ids),
            "result_count": len(self._results),
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self._checkpoint_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(self._checkpoint_path)

    def _load_checkpoint(self):
        try:
            with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"WARNING: Corrupted checkpoint, starting fresh")
            return
        self._completed_ids = set(data.get("completed_ids", []))
        if self._results_path.exists():
            results = []
            try:
                with open(self._results_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            except IOError:
                pass
            seen: Dict[str, dict] = {}
            for r in results:
                seen[r["trial_id"]] = r
            self._results = list(seen.values())
            self._last_flushed_idx = len(self._results)
        result_ids = {r["trial_id"] for r in self._results}
        missing = self._completed_ids - result_ids
        if missing:
            print(f"WARNING: {len(missing)} trials in checkpoint but missing from results — re-running")
            self._completed_ids -= missing
        print(f"Loaded checkpoint: {len(self._completed_ids)} completed, {len(self._results)} results")

    def _flush_results(self):
        new = self._results[self._last_flushed_idx:]
        if not new:
            return
        with open(self._results_path, "a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._last_flushed_idx = len(self._results)

    def _save_results_full(self):
        seen: Dict[str, dict] = {}
        for r in self._results:
            seen[r["trial_id"]] = r
        tmp = self._results_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in seen.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(self._results_path)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Paper 2 R3 — Direct API experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"),
                        help="Path to YAML config")
    parser.add_argument("--variants", default=None,
                        help="Override probe variants file path")
    parser.add_argument("--model", action="append", dest="models",
                        help="Only use specific model(s). Repeat for multiple.")
    parser.add_argument("--domain", action="append", dest="domains",
                        help="Only specific domain(s): REG-H, REG-S, UNREG, CTL")
    parser.add_argument("--role", action="append", dest="roles",
                        help="Only specific role(s): no-role, with-role, boundary-role")
    parser.add_argument("--framing", action="append", dest="framings",
                        help="Only specific framing(s): F-Expert, F-Friend, F-Tool")
    parser.add_argument("--probes", type=str, default=None,
                        help="Comma-separated probe IDs (e.g. REG-H-01,UNREG-03)")
    parser.add_argument("--count", type=int, default=None,
                        help="Limit to first N probes per domain")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print plan without API calls")
    parser.add_argument("--fresh", action="store_true",
                        help="Start in new timestamped directory")
    parser.add_argument("--workers", type=int, help="Override concurrency")

    args = parser.parse_args()

    # Load env
    env_file = find_and_load_env()
    print(f"Loaded env: {env_file}")

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / args.config
    print(f"Loading config: {config_path}")
    config = R3Config.from_yaml(str(config_path))

    if args.workers:
        config.experiment["workers"] = args.workers

    # Check model availability
    registry = ProviderRegistry()
    all_model_ids = list(set(config.model_ids) | set(args.models or []))
    available = set(registry.available_models(all_model_ids))
    requested = set(args.models) if args.models else set(config.model_ids)
    missing = requested - available
    if missing:
        print(f"WARNING: Models with no API credentials: {', '.join(sorted(missing))}")
        print("  Set the corresponding *_API_KEY in .env, or install anthropic SDK.")
        print("  These models will be skipped.")
    effective_models = [m for m in args.models or config.model_ids if m in available]
    if not effective_models:
        print("ERROR: No models available with current .env credentials. Aborting.")
        sys.exit(1)
    if args.models:
        args.models = effective_models

    # Load probes
    variants_path = args.variants or config.probes["variants"]
    variants_path = config.resolve_path(variants_path)
    print(f"Loading variants: {variants_path}")
    probes = auto_parse_variants(str(variants_path))
    print(summarize_probes(probes))

    # Parse probe filter
    probe_filter = None
    if args.probes:
        probe_filter = [p.strip() for p in args.probes.split(",")]

    # Build trial matrix
    trials = build_trial_matrix(
        config, probes,
        model_filter=args.models,
        role_filter=args.roles,
        framing_filter=args.framings,
        domain_filter=args.domains,
        probe_filter=probe_filter,
        max_per_domain=args.count,
    )
    print(summarize_plan(trials))

    if args.dry_run:
        print("\n[Dry-run] Config and trial plan validated. No API calls made.")
        return

    # Output directory
    if args.fresh:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = config.resolve_path(config.output["dir"]) / f"{config.experiment['name']}_{ts}"
    else:
        output_dir = config.resolve_path(config.output["dir"]) / config.experiment["name"]
        # Isolate by model when filtering to a single model
        if args.models and len(args.models) == 1:
            output_dir = output_dir / args.models[0].replace("/", "_").replace("\\", "_")

    print(f"\nOutput: {output_dir}")
    print(f"Providers: {', '.join(registry.available_providers())}")

    # Execute
    executor = R3Executor(config, registry, output_dir)
    results = executor.execute(trials, resume=not args.fresh)

    errors = [r for r in results if r.get("error")]
    successes = [r for r in results if not r.get("error")]
    print(f"\nFinal: {len(successes)} success, {len(errors)} errors, {len(results)} total")
    print(f"Results: {output_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()
