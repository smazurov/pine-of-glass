// Measured attribution: tool outputs from provider-reported prompt growth, and the
// harness from the first request's prompt, each only while the prompt cache proves the
// earlier prefix was reused unchanged.
import { test } from "node:test";
import assert from "node:assert/strict";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import type { AssistantMessage } from "@earendil-works/pi-ai";

import { estimateSessionBreakdown, scanSession } from "../../extensions/pi-contextimate/session-accounting.ts";
import { assistantMessage } from "../helpers.ts";

type Step = {
  prompt: number;
  cacheRead: number;
  output: number;
  reasoning?: number;
  model?: string;
  content?: AssistantMessage["content"];
};

function codexResponse(step: Step): AssistantMessage {
  return assistantMessage(step.content ?? [{ type: "toolCall", id: "call_1", name: "bash", arguments: { command: "head bin" } }], {
    api: "openai-codex-responses",
    provider: "openai-codex",
    model: step.model ?? "gpt-6-sol",
    usage: {
      input: Math.max(0, step.prompt - step.cacheRead),
      output: step.output,
      cacheRead: step.cacheRead,
      cacheWrite: 0,
      ...(step.reasoning === undefined ? {} : { reasoning: step.reasoning }),
      totalTokens: Math.max(step.prompt, step.cacheRead) + step.output,
      cost: { total: 0, input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    },
  });
}

function toolResult(text: string) {
  return {
    role: "toolResult" as const,
    toolCallId: "call_1",
    toolName: "bash",
    content: [{ type: "text" as const, text }],
    isError: false,
    timestamp: 2,
  };
}

// 1,000 chars that the provider billed as 6,000 tokens (a binary dump, say).
const DENSE_OUTPUT = "\u0000\ufffd".repeat(500);

function denseToolSession(overrides: { next?: Partial<Step>; between?: () => Parameters<SessionManager["appendMessage"]>[0] } = {}) {
  const session = SessionManager.inMemory("/tmp/contextimate-measured-growth");
  session.appendMessage({ role: "user", content: "go", timestamp: 1 });
  session.appendMessage(codexResponse({ prompt: 10_000, cacheRead: 0, output: 40 }));
  session.appendMessage(toolResult(DENSE_OUTPUT));
  if (overrides.between) session.appendMessage(overrides.between());
  session.appendMessage(codexResponse({
    prompt: 10_000 + 40 + 6_000,
    cacheRead: 9_984,
    output: 10,
    content: [{ type: "text", text: "done" }],
    ...overrides.next,
  }));
  return session;
}

const estimateOptions = { denominator: 4, harnessTokens: 9_000, contextTokens: 16_050 };

test("dense tool output counts at its measured prompt growth, not its character count", () => {
  const breakdown = scanSession(denseToolSession()).breakdown!;
  assert.equal(breakdown.measuredToolOutputTokens, 6_000);
  assert.equal(breakdown.measuredToolOutputChars, DENSE_OUTPUT.length);

  const estimate = estimateSessionBreakdown(breakdown, estimateOptions);
  assert.equal(estimate.toolOutputTokens, 6_000, "no chars ÷ 4 guess (which would say 250)");
  assert.ok(estimate.unattributedTokens < 50, `gap ${estimate.unattributedTokens} should be framing noise`);
});

test("growth is not measured when the step cannot prove what was appended", () => {
  const cases: [string, Parameters<typeof denseToolSession>[0]][] = [
    ["prompt cache missed, so the prefix may have changed", { next: { cacheRead: 0 } }],
    ["cache reused only part of the previous prompt", { next: { cacheRead: 5_000 } }],
    ["the next response came from another model", { next: { model: "gpt-6-luna" } }],
    ["a user message joined the tool result", { between: () => ({ role: "user", content: "steer", timestamp: 3 }) }],
    ["the prompt grew by less than the replayed response", { next: { prompt: 10_020 } }],
  ];
  for (const [label, overrides] of cases) {
    const breakdown = scanSession(denseToolSession(overrides)).breakdown!;
    assert.equal(breakdown.measuredToolOutputTokens, 0, label);
    assert.equal(breakdown.measuredToolOutputChars, 0, label);
  }
});

test("a cache read beyond the previous prompt still confirms the prefix", () => {
  const breakdown = scanSession(denseToolSession({ next: { cacheRead: 16_000 } })).breakdown!;
  assert.equal(breakdown.measuredToolOutputTokens, 6_000);
});

test("reasoning counts as replayed only when the response carries a replay item", () => {
  const replayItem = JSON.stringify({ type: "reasoning", id: "rs_1", encrypted_content: "opaque" });
  for (const [label, carrier, replayedReasoning] of [
    ["encrypted carrier replays reasoning", true, 500],
    ["no carrier drops reasoning", false, 0],
  ] as const) {
    const session = SessionManager.inMemory("/tmp/contextimate-measured-reasoning");
    session.appendMessage({ role: "user", content: "go", timestamp: 1 });
    session.appendMessage(codexResponse({
      prompt: 10_000,
      cacheRead: 0,
      output: 510,
      reasoning: 500,
      content: [
        { type: "thinking", thinking: "", ...(carrier ? { thinkingSignature: replayItem } : {}) },
        { type: "toolCall", id: "call_1", name: "bash", arguments: { command: "ls" } },
      ],
    }));
    session.appendMessage(toolResult("x".repeat(400)));
    session.appendMessage(codexResponse({
      prompt: 10_000 + 10 + replayedReasoning + 2_000,
      cacheRead: 9_984,
      output: 10,
      content: [{ type: "text", text: "done" }],
    }));
    assert.equal(scanSession(session).breakdown!.measuredToolOutputTokens, 2_000, label);
  }
});

test("the harness total is the first prompt while the cache chain holds", () => {
  const breakdown = scanSession(denseToolSession()).breakdown!;
  assert.deepEqual(breakdown.firstPrompt, { tokens: 10_000, preludeChars: 2 });
  const estimate = estimateSessionBreakdown(breakdown, estimateOptions);
  assert.equal(estimate.harnessSource, "measured");
  assert.equal(estimate.harnessTokens, 10_000 - 1, "first prompt minus the estimated 2-char prelude");
  assert.equal(estimate.totalTokens, 16_050 - 9_999, "the session is what the harness does not explain");

  const withoutPiTotal = estimateSessionBreakdown(breakdown, { ...estimateOptions, contextTokens: undefined });
  assert.equal(withoutPiTotal.harnessSource, "estimate", "no provider total, nothing to anchor");
});

test("the harness falls back to its estimate when the first prompt may no longer hold", () => {
  const brokenChain = denseToolSession();
  brokenChain.appendMessage({ role: "user", content: "again", timestamp: 4 });
  brokenChain.appendMessage(codexResponse({ prompt: 16_100, cacheRead: 0, output: 10, content: [{ type: "text", text: "ok" }] }));

  const compacted = denseToolSession();
  compacted.appendCompaction("summary", null, 16_050);
  compacted.appendMessage({ role: "user", content: "after", timestamp: 5 });
  compacted.appendMessage(codexResponse({ prompt: 9_500, cacheRead: 9_000, output: 10, content: [{ type: "text", text: "ok" }] }));

  const longPrelude = SessionManager.inMemory("/tmp/contextimate-long-prelude");
  longPrelude.appendMessage({ role: "user", content: "p".repeat(4 * 4_097), timestamp: 1 });
  longPrelude.appendMessage(codexResponse({ prompt: 14_100, cacheRead: 0, output: 10, content: [{ type: "text", text: "ok" }] }));

  for (const [label, session] of [
    ["a later request missed the cache", brokenChain],
    ["compaction rewrote the prompt", compacted],
    ["the first message is too large to subtract as an estimate", longPrelude],
  ] as const) {
    const estimate = estimateSessionBreakdown(scanSession(session).breakdown!, estimateOptions);
    assert.equal(estimate.harnessSource, "estimate", label);
    assert.equal(estimate.harnessTokens, estimateOptions.harnessTokens, label);
  }
});
