---
name: graphify-graph-rag
description: Answer codebase-structure questions ("where is X defined", "what connects A to B", "what depends on Y") by querying the Graphify knowledge graph instead of glob/grep/read sweeps; build or refresh the graph when it's missing or stale.
license: MIT
---

# Graph-RAG over the codebase (Graphify)

## When to use

- Any structure question: where a symbol is defined, what calls/imports it,
  how two modules connect, blast radius of a change.
- Before a broad `glob` + `grep` + multi-file `read_file` sweep — a graph
  query answers the same question in a fraction of the tokens, with
  file:line citations.

## How

If the graph tools are connected (`query_graph`, `get_node`, `shortest_path`):

1. `query_graph` with a natural-language question for a scoped subgraph.
2. `get_node` for one entity's details (definition site, relationships).
3. `shortest_path` to trace how two concepts connect (e.g. "auth" → "db").

Cite the file:line references the graph returns. Only fall back to
`grep`/`read_file` for exact code bodies the graph doesn't carry — read the
specific cited lines, not whole files.

## When the graph is missing or stale

- No graph tools connected: the graph may not be built. Suggest the user run
  `/graphify build` (or, via the bash subagent: `graphify .` in the repo
  root — the CLI installs with `uv tool install graphifyy`).
- Results look outdated after big edits: refresh incrementally with
  `graphify . --update` (only changed files are re-extracted).

## Notes

- Everything runs locally (tree-sitter); no code leaves the machine.
- The graph lives at `graphify-out/graph.json` and can be committed so the
  whole team's assistants share it.
