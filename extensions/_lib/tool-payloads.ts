// Provider tool payload formats and token estimators shared by the family.

import { isJsonObject } from "./boundary.ts";
import { estimateCharsAsTokens, type HeuristicNumbers } from "./heuristics.ts";

/** The slice of a tool definition the estimators need; contextimate's ToolSummary
 * satisfies it structurally. */
export type ToolDefinition = {
  name: string;
  description: string;
  schema: unknown;
};

// --- JSON schema readers shared by the tool estimators -----------------------------------

function trimFinalPeriod(text: string): string {
  return text.endsWith(".") ? text.slice(0, -1) : text;
}

export function getSchemaProperties(schema: unknown): Record<string, unknown> {
  if (!isJsonObject(schema) || !isJsonObject(schema.properties)) return {};
  return schema.properties;
}

export function schemaPropertyType(property: unknown): string {
  if (!isJsonObject(property)) return "object";
  if (typeof property.type === "string") return property.type;
  if (Array.isArray(property.type)) return property.type.filter((entry): entry is string => typeof entry === "string").join("|");
  if (property.anyOf) return "anyOf";
  if (property.oneOf) return "oneOf";
  if (property.allOf) return "allOf";
  return "object";
}

export function schemaPropertyDescription(property: unknown): string {
  if (!isJsonObject(property)) return "";
  return typeof property.description === "string" ? trimFinalPeriod(property.description) : "";
}

export function schemaArrayItemProperties(property: unknown): Record<string, unknown> {
  if (!isJsonObject(property)) return {};
  return getSchemaProperties(property.items);
}

export function getSchemaRequired(schema: unknown): string[] {
  if (!isJsonObject(schema) || !Array.isArray(schema.required)) return [];
  return schema.required.filter((entry): entry is string => typeof entry === "string");
}

export function arrayItemsSchema(property: unknown): unknown {
  return isJsonObject(property) ? property.items : undefined;
}

export function safeMinifiedJson(value: unknown): string {
  try {
    return JSON.stringify(value) ?? "undefined";
  } catch (error) {
    return `[unserializable: ${error instanceof Error ? error.message : String(error)}]`;
  }
}

// --- provider tool payload formats ---------------------------------------------------------

export function openAIResponsesToolPayload(tool: ToolDefinition): unknown {
  return {
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.schema,
    strict: null,
  };
}

/** `format` is a `toolNumerator` name; `openai-cookbook` and unknown names take the
 * OpenAI Responses payload. */
export function toolPayload(tool: ToolDefinition & { promptGuidelines?: string[] }, format: string): unknown {
  switch (format) {
    case "openai-chat":
    case "openai-completions":
    case "mistral":
      return {
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.schema,
          strict: null,
        },
      };
    case "anthropic":
      return { name: tool.name, description: tool.description, input_schema: tool.schema };
    case "gemini":
    case "google":
    case "vertex":
      return { name: tool.name, description: tool.description, parametersJsonSchema: tool.schema };
    case "bedrock":
      return {
        toolSpec: {
          name: tool.name,
          description: tool.description,
          inputSchema: { json: tool.schema },
        },
      };
    case "pi-messages":
      return { name: tool.name, description: tool.description, parameters: tool.schema };
    case "raw-schema":
      return {
        name: tool.name,
        description: tool.description,
        parameters: tool.schema,
        promptGuidelines: tool.promptGuidelines ?? [],
      };
    default:
      return openAIResponsesToolPayload(tool);
  }
}

export function aggregateToolPayload(tools: Array<ToolDefinition & { promptGuidelines?: string[] }>, format: string): unknown {
  if (format === "gemini" || format === "google" || format === "vertex") {
    return {
      functionDeclarations: tools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        parametersJsonSchema: tool.schema,
      })),
    };
  }
  return tools.map((tool) => toolPayload(tool, format));
}

export function toolPayloadLabel(format: string): string {
  switch (format) {
    case "openai-responses":
    case "openai-codex-responses":
      return "OpenAI Responses tool payload";
    case "openai-chat":
    case "openai-completions":
    case "mistral":
      return "OpenAI Chat tool payload";
    case "anthropic":
      return "Anthropic tool payload";
    case "gemini":
    case "google":
    case "vertex":
      return "Gemini/Vertex tool payload";
    case "bedrock":
      return "Bedrock tool payload";
    case "pi-messages":
      return "Pi Messages tool payload";
    case "raw-schema":
      return "Raw tool schema payload";
    default:
      return `Unknown tool payload format ${format}; OpenAI Responses fallback`;
  }
}

// --- OpenAI tool formula ------------------------------------------------------------------

// OpenAI renders each function as a TypeScript-style declaration, which drops most of its
// JSON envelope. Live counts of 179 tools fit these constants; see docs/pi-contextimate.md.
const OPENAI_TOOL_ENVELOPE_CHARS = 190;
const OPENAI_SMALL_TOOL_DENOMINATOR = 8;
const OPENAI_TOOL_BLOCK_TOKENS = 16;

export function estimateOpenAIToolDefinitionTokens(tool: ToolDefinition, denominator: number): number {
  const chars = safeMinifiedJson(openAIResponsesToolPayload(tool)).length;
  return Math.ceil(Math.max(chars / OPENAI_SMALL_TOOL_DENOMINATOR, (chars - OPENAI_TOOL_ENVELOPE_CHARS) / denominator));
}

export function estimateOpenAIFunctionToolTokens(tools: ToolDefinition[], denominator: number): number {
  return tools.reduce((sum, tool) => sum + estimateOpenAIToolDefinitionTokens(tool, denominator), OPENAI_TOOL_BLOCK_TOKENS);
}

/** Total estimated tokens for a tool list under a family heuristic: the cookbook
 * formula where it applies, the payload char ratio everywhere else. */
export function estimateToolListTokens(
  tools: Array<ToolDefinition & { promptGuidelines?: string[] }>,
  heuristic: Pick<HeuristicNumbers, "toolNumerator" | "toolDenominator">,
): number {
  if (tools.length === 0) return 0;
  if (heuristic.toolNumerator === "openai-cookbook") return estimateOpenAIFunctionToolTokens(tools, heuristic.toolDenominator);
  const content = safeMinifiedJson(aggregateToolPayload(tools, heuristic.toolNumerator));
  return estimateCharsAsTokens(content.length, heuristic.toolDenominator);
}
