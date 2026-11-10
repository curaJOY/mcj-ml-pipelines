# AGENTS.md - mcj-ml-pipelines

Data and machine-learning pipelines. Follow README.md and existing directory conventions.

<!-- BEGIN MCJ SHARED RULES -->

## GitHub Access - No Inference Without Direct Access

When asked to review a PR, issue, or any GitHub resource:

1. Use a direct GitHub integration (`MCP` tools, `gh` CLI, or GitHub API).
   Do not infer PR or issue content from git branches, diffs, or log output.
2. If no GitHub integration is available, stop immediately.
   State the blocker exactly: "I cannot access GitHub metadata - no MCP tools, `gh` CLI, or API token are available."
3. Never guess which branch corresponds to a PR number.
4. Never present inferred content as retrieved fact.

## Minimal Change Principle

- Only modify files directly related to the task.
- Do not refactor, reformat, or "improve" adjacent code.
- Do not add dependencies without explicit approval.
- If a change touches more than 5 files, pause and explain the scope before proceeding.

## Environment Constraints

- If a command requires a tool, runtime, or permission you do not have, say so immediately.
- Do not simulate command output. If you cannot run it, say: "I cannot run this command because..."
- Do not assume environment variables or secrets are available unless you have verified that they are.
- Never uncomment or assign empty values to env vars marked as runtime-injected in mcj-ms-envs S3 env files. Lines like `# KEY=` are documentation placeholders for ECS-injected secrets. Setting `KEY=` exports an empty string that overwrites the injected value at container startup.

## Development Process - Dev Is The Test Environment

`develop` deploys to dev. Dev has no real users: it is where features are built and tested, and
`main` (real users) is reached only by promotion after testing on dev. Plan, version and gate work
for that process:

- **Never shadow first.** Do not run a new path in shadow mode (computed but not served, or compared
  silently against the old path) before turning it on. A shadowed path never serves, so nobody on
  dev can see or test it. Turn it on in dev and test it there.
- **Do not version what has not shipped.** Do not add a `v2`, a parallel endpoint, field, job, trigger
  or scoring path, or a flag, to protect behaviour that has never reached `main`. Change it in place
  on dev. Keep an old path or add a version only when the old shape is already live on `main`, or a
  deployed consumer outside your change still depends on it, and name that consumer.
- **Unfinished features are normal on dev.** A feature may land on dev incomplete and be finished in
  later PRs. Do not hide it behind extra flags, cohorts or proof gates on dev to make it look done.
  A switch that is on in `develop` and off in `main` until promotion is fine.
- **Evidence gates promotion, not dev.** Evaluations, cohort comparisons, quality and latency targets
  and rollback rehearsals gate promotion to `main` only where the feature's acceptance criteria
  require them. Do not invent these requirements for every change or use them to delay dev enablement.
  A dev-side precondition must name its functional dependency (for example "the coaches route is
  deployed"), affected environment and blocked action. Continue independent authorized work.
- **Correctness rules still apply everywhere.** Authorization, privacy, age gates, data integrity,
  logging rules and each repository's migration-safety rules are code behaviour, not rollout caution.
- **Older docs that say otherwise are superseded.** Plans and prompt packs, including done and
  archived ones, that prescribe shadow-first rollout or parallel versions of unreleased work are
  superseded by this section.

## Finish And Extend The Existing Implementation

- Repair the missing integrations of an unfinished feature and extend its existing contracts,
  workers and consumers in place. Keep one intended end-to-end flow across the normal service
  boundaries; incompleteness alone does not justify a successor engine, alternate pipeline,
  parallel contract or compatibility framework.
- A second implementation needs a concrete product or end-user benefit, or a named deployed
  dependency that cannot reasonably use the existing path. Explain that reason briefly in the
  existing plan or PR. Speculative future compatibility, a newer version number or passing
  component tests is insufficient; no separate audit or evidence pack is required.
- Incomplete features may land on dev incrementally. Report **implemented**, **enabled on dev**,
  **validated on dev** and **production-ready** separately, only when established. State what a
  tester can actually do on dev, where that journey stops and what work remains. A tag, component
  merge or successful deployment does not establish end-to-end feature completion.
- Where avoidable parallel paths already exist, consolidate them as part of completing the
  feature when practical. Do not introduce another abstraction solely to reconcile that duplication.
- Prompt packs must describe the same intended journey, missing integrations and applicable
  acceptance criteria. Production promotion evidence stays at the production boundary; this rule
  adds no per-PR end-to-end test requirement or new rollout ceremony.

## Apply Rules Within Their Scope

Determine whether a rule applies to the action and environment before resolving conflicts.
Production-only precautions do not constrain development enablement. Follow instruction precedence
for applicable rules; prefer the safer interpretation only for equally authoritative rules about
the same action and environment. Authorization, privacy, secret handling, data integrity and
migration safety remain applicable throughout development.

<!-- END MCJ SHARED RULES -->
