// Provider payload formats + the OpenAI tool render. The displayed token number is only
// as honest as these payloads; the render is pinned against provider-measured counts.
import { readFileSync } from "node:fs";
import { test } from "node:test";
import assert from "node:assert/strict";

import { aggregateToolPayload, toolPayload } from "../../extensions/_lib/tool-payloads.ts";
import { internals } from "../../extensions/pi-contextimate/index.ts";
import type { ToolSummary } from "../../extensions/pi-contextimate/index.ts";

const {
  buildToolNumerator,
  buildToolDisplayEstimate,
  estimateOpenAIToolDefinitionTokens,
  estimateOpenAIFunctionToolTokens,
  resolveHeuristic,
} = internals;

const ping: ToolSummary = {
  name: "ping",
  description: "Send a ping.",
  source: "builtin",
  schema: {
    type: "object",
    properties: { host: { type: "string", description: "Target host" } },
    required: ["host"],
  },
  promptGuidelines: [],
};

const mode: ToolSummary = {
  name: "mode",
  description: "Pick a mode",
  source: "builtin",
  schema: {
    type: "object",
    properties: { level: { type: "string", enum: ["a", "bb"], description: "" } },
  },
  promptGuidelines: [],
};

test("per-provider payload formats are exact", () => {
  assert.deepEqual(toolPayload(ping, "anthropic"), {
    name: "ping",
    description: "Send a ping.",
    input_schema: ping.schema,
  });
  assert.deepEqual(toolPayload(ping, "openai-responses"), {
    type: "function",
    name: "ping",
    description: "Send a ping.",
    parameters: ping.schema,
    strict: null,
  });
  assert.deepEqual(toolPayload(ping, "openai-chat"), {
    type: "function",
    function: { name: "ping", description: "Send a ping.", parameters: ping.schema, strict: null },
  });
  assert.deepEqual(toolPayload(ping, "bedrock"), {
    toolSpec: { name: "ping", description: "Send a ping.", inputSchema: { json: ping.schema } },
  });
  assert.deepEqual(toolPayload(ping, "pi-messages"), {
    name: "ping",
    description: "Send a ping.",
    parameters: ping.schema,
  });
  // Gemini aggregates into one functionDeclarations wrapper.
  assert.deepEqual(aggregateToolPayload([ping, mode], "gemini"), {
    functionDeclarations: [
      { name: "ping", description: "Send a ping.", parametersJsonSchema: ping.schema },
      { name: "mode", description: "Pick a mode", parametersJsonSchema: mode.schema },
    ],
  });
});

test("unknown formats fall back to the OpenAI Responses payload", () => {
  assert.deepEqual(toolPayload(ping, "some-future-format"), toolPayload(ping, "openai-responses"));
});

test("OpenAI tool render tracks provider-measured counts", () => {
  const measured = JSON.parse(readFileSync(new URL("../fixtures/openai-codex-tool-counts.json", import.meta.url), "utf8"));
  const tools: Array<ToolSummary & { measuredTokens: number }> = measured.tools;
  for (const tool of tools) {
    const estimate = estimateOpenAIToolDefinitionTokens(tool);
    assert.ok(Math.abs(estimate - tool.measuredTokens) <= tool.measuredTokens * 0.15, `${tool.name}: ${estimate} vs ${tool.measuredTokens}`);
  }
  const total = measured.blockTokens + tools.reduce((sum, tool) => sum + tool.measuredTokens, 0);
  const estimate = estimateOpenAIFunctionToolTokens(tools);
  assert.ok(Math.abs(estimate - total) <= total * 0.02, `${estimate} vs ${total}`);
});

test("displayed per-tool estimates count the same payload the section total counts", () => {
  // Anthropic format: the aggregate content must be exactly the JSON array of the
  // per-tool payloads that buildToolDisplayEstimate measures.
  const heuristic = resolveHeuristic({ provider: "anthropic", id: "claude-opus-4-8", api: "anthropic-messages" }, {});
  const numerator = buildToolNumerator([ping, mode], heuristic);
  const perTool = [ping, mode].map((tool) => JSON.stringify(toolPayload(tool, "anthropic")));
  assert.equal(numerator.content, `[${perTool.join(",")}]`);
  for (const tool of [ping, mode]) {
    const estimate = buildToolDisplayEstimate(tool, heuristic);
    assert.equal(estimate.chars, JSON.stringify(toolPayload(tool, "anthropic")).length);
    assert.equal(estimate.tokens, Math.ceil(estimate.chars / heuristic.toolDenominator));
  }

  // OpenAI render: the section total is the per-tool estimates plus the tool block.
  const codexHeuristic = resolveHeuristic({ provider: "openai-codex", id: "gpt-5.5", api: "openai-codex-responses" }, {});
  const codexNumerator = buildToolNumerator([ping, mode], codexHeuristic);
  const perToolTokens = [ping, mode].map((tool) => buildToolDisplayEstimate(tool, codexHeuristic).tokens);
  assert.equal(codexNumerator.tokens, perToolTokens.reduce((a, b) => a + b, 0) + 16);
});
