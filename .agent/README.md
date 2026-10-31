---
repo: AgentTune
upstream: https://github.com/Lexsi-Labs/AgentTune
track: supporting
pipeline_role: agent-training
version: 1.1.3 (tag; pyproject fallback and README badge say 1.1.0)
commit: d4db4c1 (2026-10-07)
license: LSAL-1.1 (source-available, noncommercial)
python: ">=3.12"
install: 'pip install -e .  # extras: .[vllm] Linux+CUDA only'
extras: [vllm, service, docs, dev, flash-attn]
hardware: GPU for training; notebooks use an L4 24GB; <=3B use cases fit 48GB; the EventLog spine runs on CPU
gated_models: [CohereLabs/tiny-aya-*, CohereLabs/aya-expanse-*, CohereLabs/aya-vision-*]
api_keys: [HF_TOKEN]
tags:
  [agentic-rl, tool-calling, grpo, ppo, dpo, rloo, bco, eventlog,
   trajectory, distillation, self-heal, decide, rag, multi-agent, provenance]
evidence_level: docs+code inspected; no end-to-end run in this workspace
last_reviewed: 2026-10-30
---

# AgentTune — agent capability brief

> Trains and operates tool-calling LLM agents: one EventLog trajectory schema
> through build → collect → evaluate → train → distill → heal, with GRPO/PPO/DPO/
> RLOO/BCO and a YAML decision-workflow engine. Supporting repo, not one of the
> four scored track repos.

## 1. Role in a safety pipeline

AgentTune is the agent-training and agent-lifecycle sibling. It sits on top of an
agent framework and moves one agent artifact through its lifecycle using a
normalized trajectory schema (`EventLog`). It trains tool-calling policies with
RL, distills a stronger teacher into a small student, detects and repairs failing
agent loops through DECIDE, and writes `trajectories.jsonl` plus a model/adapter
that AuditKIT can read. In a safety project it is the tool for measuring and
repairing unsafe tool use, looping/self-healing failures, and tool-call policy
drift, rather than model-level refusal behaviour.

## 2. Capability map

Status legend: `stable` = code + runnable example/test exists; `documented` =
docs claim it and code exists but no runnable example/test found; `experimental`
= repo flags it as partial; `absent` = not implemented.

| Capability | Entry point | Status | Evidence | Safety-pipeline use |
|---|---|---|---|---|
| EventLog spine | `from agenttune.agentic import EventLog, Event, EventKind` | stable (schema frozen under semver) | `src/agenttune/agentic/events.py:20-44`; `CHANGELOG.md:51-52` | One trace schema across build/collect/eval/train/distill/heal |
| Agentic RL | `create_agentic_trainer(algorithm="grpo"\|"ppo"\|"dpo"\|"rloo"\|"bco")`; CLI `agenttune train` | stable factory; PPO/BCO use `trl.experimental` | `src/agenttune/core/backend_factory.py:130`; `docs/algorithms/overview.md` | Train tool-calling policies and safety-relevant agent behaviour |
| Distillation | `create_distill_trainer` (package root or `agenttune.agentic`) | stable: documented + examples | `src/agenttune/agentic/distill_trainer.py:277`; `docs/user-guide/distillation.md` | Compress a safe teacher policy into a small student |
| Self-heal / DECIDE | `agenttune.decide.GraphRunner`; CLI `agenttune decide run`; `FullClosedLoop`; `SelfHealLoop` | experimental; closed loop partially verified, audit-log extractors broken, runners stub-injected | `src/agenttune/decide/graph_runner.py:27`; `src/agenttune/decide/closed_loop/full_loop.py:148`; `src/agenttune/agentic/heal_loop.py:69` | Detect/classify/retrain unsafe or looping agents, gate redeploy |
| Agentic RAG | `create_rag_trainer` | experimental; base path works, but `HybridBackend.index` and curriculum path raise `NotImplementedError` | `src/agenttune/rag/trainer.py:98`; `src/agenttune/rag/retrieval/hybrid_backend.py:96` | Train retrieval agents against a retrieval reward |
| Evaluation | `agenttune.eval.BaseEvaluator`, `RLEvaluator`, `LMEvalRunner`, `run_eval` | documented; `run_eval` exported but has zero internal callers | `src/agenttune/eval/evaluator.py:36`; `src/agenttune/eval/lm_eval_integration.py:97`; `src/agenttune/eval/agent_eval.py:1190` | Tool-use scoring, standard benchmarks, before/after agent metrics |
| Tool-calling fallback | `create_agentic_trainer(..., tools_fallback_prompt=..., tool_result_format=...)` | stable: documented + examples | `src/agenttune/agentic/rollout_engines/rollout_factory.py:1549,1573,2175-2196` | Make Cohere models usable for tool calls despite templates |
| Multi-agent orchestration | `AgentTuneGraph` (LangGraph) | documented | `src/agenttune/agentic/langgraph_orchestrator.py:1-13`; `src/agenttune/langgraph/langgraph.py:177` | Compose rollouts + judges into a GRPO reward |
| Trajectory export | GRPO/RLOO write `trajectories.jsonl`; model/adapter + tokenizer + provenance | stable: GRPO/RLOO only | `src/agenttune/agentic/trajectory/dataset.py:43-67`; `src/agenttune/backends/trl/agentic/grpo/agentic_grpo.py:454-464` | AuditKIT-readable evidence of agent behaviour |

## 3. Minimum viable usage

No model, no GPU, no network:

```python
from agenttune.agentic import Project, DictToolHarness, ReActStrategy

harness = DictToolHarness({"add": lambda a, b: a + b}, max_steps=4)
strategy = ReActStrategy(lambda s: {"name": "add", "arguments": {"a": 2, "b": 3}}
                         if s.step == 0 else {"name": "finish", "arguments": {"answer": "5"}})

proj = Project(strategy=strategy, harness=harness)
proj.infer("what is 2+3?")                                    # one episode -> EventLog
proj.evaluate([{"task": "what is 2+3?", "expected": "5"}])    # mean_score
```

Tool-calling GRPO on Tiny Aya (CPU or a small GPU; the fallback warns because
Cohere templates ignore `tools`):

```python
trainer = create_agentic_trainer(
    algorithm="grpo", model="CohereLabs/tiny-aya-global",
    train_dataset=train_dataset, tools=[add], reward_funcs=[used_tool],
    output_dir="./output/tiny_aya_tools", max_steps=10,
    per_device_train_batch_size=2, num_generations=2,
    use_vllm=False, report_to="none",
)
trainer.train()
```

## 4. Constraints and failure modes

- **Python >=3.12**, `torch==2.11.0`, `transformers>=5.15,<6`,
  `trl==1.7.1`, `torchao>=0.17,<0.18`. `[vllm]` resolves to vLLM 0.22.1 and is
  Linux+CUDA only; the base install works on CPU/macOS. No published VRAM floor;
  training needs a GPU.
- **Cohere/Aya tool calling:** every hackathon Cohere model's chat template
  ignores `tools` (Tiny Aya renders a `tool` turn as empty). AgentTune falls back
  by putting tool schemas in the system prompt and folding tool results into user
  turns, with a warning. `tools_fallback_prompt=None` raises. North's model card
  says tool calling is unsupported; full fine-tuning needs `beta=0`.
- **Silent failures to check:** `DistilledJudge` can return all zeros; ROUGE/BLEU
  can be all zeros; `StageWiseRunner` ignores routing; audit DPO/BCO extractors
  are always empty; the CLI lacks `--tools`; registry blanket-import can crash;
  an unconfigured SQL tool is a silent trap; GRPO on a small dataset can silently
  exit at step 0.
- **Export gaps:** only GRPO/RLOO write `trajectories.jsonl`; DPO/PPO/BCO do not.
  `reward` is null unless `keep_reward`, because scoring happens after logging.
  There is no built-in EventLog JSON serializer.
- **Version drift:** tag 1.1.3 vs pyproject fallback/README badge 1.1.0;
  `CHANGELOG.md` latest entry is 1.1.0.

## 5. Gaps and extension points

- `git_sha` in provenance is always `None`. Cross-artifact joins lack a shared
  episode/call ID.
- `Project.train` only supports `sft`/`grpo`. `HybridBackend.index` and the
  curriculum path raise `NotImplementedError`.
- The deployment gate can approve on an empty test set; closed-loop runners are
  documented as stub-injected; destination routing is a no-op; cost tracking is
  a TODO in DECIDE stages.
- Docs/code mismatches: README says every trajectory tool call is
  `{id,name,arguments}`, but bare actions pass through; docs say `run_eval` is
  unexported while code exports it.
- Extension surfaces: stable episode/call IDs and real `git_sha`; export
  DPO/PPO/BCO trajectories, non-null reward, and an EventLog JSON writer; harden
  AuditKIT importers; Cohere/Aya fallback/parser plus tool-call evaluation;
  fix DECIDE's audit-log bridge (`extract_dpo_pairs`/`extract_bco_labels`), seed
  the deployment gate, and verify a real retrain → gate → deploy loop.

## 6. Interop contract

- **`trajectories.jsonl`:** one JSON per rollout with `task`,
  `steps[{step_number,state,action,observation,thought,reward}]`, top-level
  `reward`, `trajectory_id`, `final_response`, and `metadata.conversation`
  (training-only keys stripped). Tool calls are normalized to
  `{id,name,arguments}` only when `action` carries `tool_calls`.
- **EventLog:** `{events:[{kind,payload,token_span?,logprobs?,scope?}],
  tier:"light"|"full", id:uuid}`. AuditKIT accepts the object or this dict.
- **AgentTune → AuditKIT:** `episodes_from_agenttune(path)` reads
  trajectories, EventLogs, traces, reports, and run folders;
  `episode_from_eventlog(log)` takes a live object. AuditKIT never imports
  AgentTune.
- **CuratorKIT → AgentTune:** a curated dataset folder's own
  `lexsi_provenance.json` is embedded under `inputs[].provenance`.
- **Output layout:** model or PEFT adapter (`adapter_config.json`) plus
  tokenizer/processor, `<algo>_config.yaml`, `training_stats.json`,
  `lexsi_provenance.json`, and `trajectories.jsonl` for GRPO/RLOO.
- **Provenance:** `lexsi.provenance/1` with `library="agenttune"`, `version`,
  `git_sha` (currently null), `created_at`, `base_model`, `method`, `inputs[]`,
  `params`.
- **No AlignTune bridge** in this snapshot; both are separate trainers.

## 7. Verification recipes

Run these from the repo root.

| Check | Command | Expected |
|---|---|---|
| Spine quickstart | `PYTHONPATH=src python -c "from agenttune.agentic import Project, DictToolHarness, ReActStrategy; h=DictToolHarness({'add': lambda a,b: a+b}, max_steps=4); s=ReActStrategy(lambda st: {'name':'add','arguments':{'a':2,'b':3}} if st.step==0 else {'name':'finish','arguments':{'answer':'5'}}); print(Project(strategy=s,harness=h).evaluate([{'task':'what is 2+3?','expected':'5'}]))"` | `mean_score: 1.0`; 1–3s, no GPU/network |
| Tag | `git describe --tags --always` | `1.1.3`; <1s |
| AuditKIT reader | `PYTHONPATH=../AuditKIT/src:src python -c "from auditkit.agent_eval import episodes_from_agenttune; e=episodes_from_agenttune('../AuditKIT/tests/fixtures/lexsi_real/agenttune/trace_finder.jsonl'); print(len(e), e[0].source_format)"` | `2 agenttune_trace`; 2–5s |

## 8. Sources

Read for this brief: `README.md`, `pyproject.toml`, `CHANGELOG.md`,
`docs/` (community/known-issues, algorithms, user-guide RL/distillation/
self-healing/RAG, reference architecture, notebooks), `src/agenttune/`
(agentic spine, core/backend_factory, trajectory dataset, rollout engines,
decide, rag, eval, provenance), `examples/`, `tests/test_auditkit_interop.py`.
