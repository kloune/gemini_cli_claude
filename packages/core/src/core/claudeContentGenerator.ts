/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  FinishReason,
  type GenerateContentParameters,
  type GenerateContentResponse,
  type CountTokensParameters,
  type CountTokensResponse,
  type EmbedContentParameters,
  type EmbedContentResponse,
  type Content,
  type Part,
  type Candidate,
  type GenerateContentResponseUsageMetadata,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';
import { estimateTokenCountSync } from '../utils/tokenCalculation.js';
import { debugLogger } from '../utils/debugLogger.js';
import { CLAUDE_SONNET_MODEL, CLAUDE_HAIKU_MODEL } from '../config/models.js';

// Vertex SDK for the client class
type AnthropicVertexType = import('@anthropic-ai/vertex-sdk').AnthropicVertex;

// Types from the base Anthropic SDK (transitive dependency of vertex-sdk)
type MessageCreateParamsBase =
  import('@anthropic-ai/sdk/resources/messages/messages').MessageCreateParamsBase;
type MessageCreateParamsType =
  import('@anthropic-ai/sdk/resources/messages/messages').MessageCreateParamsNonStreaming;
type MessageParamType =
  import('@anthropic-ai/sdk/resources/messages/messages').MessageParam;
type ContentBlockParamType =
  import('@anthropic-ai/sdk/resources/messages/messages').ContentBlockParam;
type ToolType = import('@anthropic-ai/sdk/resources/messages/messages').Tool;
type MessageType =
  import('@anthropic-ai/sdk/resources/messages/messages').Message;
type MessageStreamEventType =
  import('@anthropic-ai/sdk/resources/messages/messages').MessageStreamEvent;

/**
 * Maps Gemini model names used in auxiliary calls to Claude equivalents.
 */
function mapGeminiModelToClaude(
  geminiModel: string,
  userModel: string,
): string {
  if (geminiModel.startsWith('claude-')) {
    return geminiModel;
  }
  if (geminiModel.includes('flash-lite') || geminiModel.includes('flash')) {
    return CLAUDE_HAIKU_MODEL;
  }
  if (geminiModel.includes('pro')) {
    return userModel;
  }
  // Default: use the user's selected model
  return userModel;
}

/**
 * Translates Gemini Content[] to Anthropic MessageParam[].
 * Handles role mapping, message alternation merging, and part conversion.
 */
function translateContentsToMessages(contents: Content[]): {
  messages: MessageParamType[];
} {
  const messages: MessageParamType[] = [];

  for (const content of contents) {
    const role =
      content.role === 'model' ? ('assistant' as const) : ('user' as const);

    const blocks = translateParts(content.parts || []);
    if (blocks.length === 0) continue;

    // Merge consecutive same-role messages (Anthropic requires strict alternation)
    const lastMsg = messages[messages.length - 1];
    if (lastMsg && lastMsg.role === role) {
      if (Array.isArray(lastMsg.content)) {
        lastMsg.content.push(...blocks);
      } else {
        lastMsg.content = [{ type: 'text', text: lastMsg.content }, ...blocks];
      }
    } else {
      messages.push({
        role,
        content: blocks,
      });
    }
  }

  // Ensure conversation starts with a user message
  if (messages.length > 0 && messages[0].role === 'assistant') {
    messages.unshift({
      role: 'user',
      content: [{ type: 'text', text: '(conversation start)' }],
    });
  }

  return { messages };
}

/**
 * Translates Gemini Part[] to Anthropic content blocks.
 */
function translateParts(parts: Part[]): ContentBlockParamType[] {
  const blocks: ContentBlockParamType[] = [];

  for (const part of parts) {
    // Skip thought parts - Claude generates its own thinking
    if (part.thought) continue;

    // Strip thoughtSignature - Gemini-specific
    // (no action needed, just don't include it)

    if (part.text !== undefined && part.text !== null) {
      blocks.push({ type: 'text', text: part.text });
    } else if (part.functionCall) {
      blocks.push({
        type: 'tool_use',
        id: part.functionCall.id || `call_${crypto.randomUUID()}`,
        name: part.functionCall.name || 'unknown',
        input: part.functionCall.args || {},
      } as ContentBlockParamType);
    } else if (part.functionResponse) {
      // Stringify response only if it's not already a string
      const responseData = part.functionResponse.response;
      const contentStr =
        typeof responseData === 'string'
          ? responseData
          : JSON.stringify(responseData || {});
      blocks.push({
        type: 'tool_result',
        tool_use_id: part.functionResponse.id || '',
        content: contentStr,
      } as ContentBlockParamType);
    } else if (part.inlineData) {
      const mediaType = part.inlineData.mimeType || 'image/png';
      // Only pass images to Claude - skip other media types
      if (mediaType.startsWith('image/')) {
        blocks.push({
          type: 'image',
          source: {
            type: 'base64',
            // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
            media_type: mediaType as
              | 'image/jpeg'
              | 'image/png'
              | 'image/gif'
              | 'image/webp',
            data: part.inlineData.data || '',
          },
        } as ContentBlockParamType);
      } else {
        debugLogger.warn(
          `[ClaudeContentGenerator] Unsupported inline data type: ${mediaType}, skipping`,
        );
      }
    } else if (part.fileData) {
      debugLogger.warn(
        '[ClaudeContentGenerator] fileData parts are not supported by Claude, skipping',
      );
    }
  }

  return blocks;
}

/**
 * Translates Gemini Tool[] to Anthropic Tool[].
 */
function translateTools(geminiTools: unknown[] | undefined): ToolType[] {
  if (!geminiTools || !Array.isArray(geminiTools)) return [];

  const claudeTools: ToolType[] = [];

  for (const tool of geminiTools) {
    if (!tool || typeof tool !== 'object') continue;

    // Skip Gemini-specific tools
    if ('googleSearch' in tool || 'urlContext' in tool) {
      debugLogger.warn(
        '[ClaudeContentGenerator] googleSearch/urlContext tools not available for Claude, skipping',
      );
      continue;
    }

    if (
      'functionDeclarations' in tool &&
      (tool as Record<string, unknown>)['functionDeclarations']
    ) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      const declarations = (tool as Record<string, unknown>)[
        'functionDeclarations'
      ] as Array<{
        name?: string;
        description?: string;
        parameters?: Record<string, unknown>;
      }>;
      for (const func of declarations) {
        claudeTools.push({
          name: func.name || 'unknown',
          description: func.description || '',
          // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
          input_schema: (func.parameters || {
            type: 'object',
            properties: {},
          }) as ToolType['input_schema'],
        });
      }
    }
  }

  return claudeTools;
}

/**
 * Extracts system instruction text from Gemini config.
 */
function extractSystemInstruction(
  config: GenerateContentParameters['config'],
): string | undefined {
  if (!config?.systemInstruction) return undefined;

  const sysInstr = config.systemInstruction;
  if (typeof sysInstr === 'string') return sysInstr;

  // Content object with parts
  if (typeof sysInstr === 'object' && 'parts' in sysInstr) {
    const content = sysInstr;
    return (content.parts || [])
      .filter((p: Part) => p.text)
      .map((p: Part) => p.text)
      .join('\n');
  }

  // Part or Part[]
  if (Array.isArray(sysInstr)) {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    return (sysInstr as Part[])
      .filter((p: Part) => p.text)
      .map((p: Part) => p.text)
      .join('\n');
  }

  if (typeof sysInstr === 'object' && 'text' in sysInstr) {
    return sysInstr.text || undefined;
  }

  return undefined;
}

/**
 * Translates Anthropic stop_reason to Gemini FinishReason.
 */
function translateStopReason(
  stopReason: string | null | undefined,
): FinishReason {
  switch (stopReason) {
    case 'end_turn':
    case 'tool_use':
      return FinishReason.STOP;
    case 'max_tokens':
      return FinishReason.MAX_TOKENS;
    default:
      return FinishReason.STOP;
  }
}

/**
 * Translates an Anthropic Message response to a Gemini GenerateContentResponse.
 */
function translateResponseToGemini(
  message: MessageType,
): GenerateContentResponse {
  const parts: Part[] = [];

  for (const block of message.content) {
    if (block.type === 'text') {
      parts.push({ text: block.text });
    } else if (block.type === 'tool_use') {
      parts.push({
        functionCall: {
          id: block.id,
          name: block.name,
          // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
          args: block.input as Record<string, unknown>,
        },
      });
    } else if (block.type === 'thinking') {
      parts.push({
        // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
        text: (block as unknown as Record<string, string>)['thinking'],
        thought: true,
      });
    }
  }

  const candidate: Candidate = {
    content: { role: 'model', parts },
    finishReason: translateStopReason(message.stop_reason),
  };

  const usageMetadata: GenerateContentResponseUsageMetadata = {
    promptTokenCount: message.usage?.input_tokens,
    candidatesTokenCount: message.usage?.output_tokens,
    totalTokenCount:
      (message.usage?.input_tokens || 0) + (message.usage?.output_tokens || 0),
  };

  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  return {
    candidates: [candidate],
    usageMetadata,
  } as unknown as GenerateContentResponse;
}

/**
 * ContentGenerator adapter that translates Gemini-typed requests to
 * Anthropic Messages API format, calls Claude via @anthropic-ai/vertex-sdk,
 * and translates responses back to Gemini types.
 */
export class ClaudeContentGenerator implements ContentGenerator {
  private client: AnthropicVertexType | null = null;
  private readonly projectId: string;
  private readonly region: string;
  private readonly userModel: string;

  constructor(projectId: string, region: string, userModel?: string) {
    this.projectId = projectId;
    this.region = region;
    this.userModel = userModel || CLAUDE_SONNET_MODEL;
  }

  private async getClient(): Promise<AnthropicVertexType> {
    if (!this.client) {
      const { AnthropicVertex } = await import('@anthropic-ai/vertex-sdk');
      this.client = new AnthropicVertex({
        projectId: this.projectId,
        region: this.region,
      });
    }
    return this.client;
  }

  /**
   * Build the Anthropic API request parameters from a Gemini request.
   */
  private buildRequest(
    req: GenerateContentParameters,
  ): MessageCreateParamsBase {
    const model = mapGeminiModelToClaude(req.model, this.userModel);
    const contents = Array.isArray(req.contents)
      ? // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
        (req.contents as Content[])
      : [];
    const { messages } = translateContentsToMessages(contents);

    let systemText = extractSystemInstruction(req.config);
    const tools = translateTools(req.config?.tools);

    // Handle JSON mode: Claude doesn't support responseMimeType, so we add
    // explicit JSON instructions to the system prompt
     
    const responseMimeType =
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      (req.config as Record<string, unknown> | undefined)?.[
        'responseMimeType'
      ] as string | undefined;
     
    const responseJsonSchema =
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      (req.config as Record<string, unknown> | undefined)?.[
        'responseJsonSchema'
      ] as Record<string, unknown> | undefined;

    if (responseMimeType === 'application/json') {
      const jsonInstruction = responseJsonSchema
        ? `\n\nYou MUST respond with valid JSON only, no other text. Your response must conform to this JSON schema:\n${JSON.stringify(responseJsonSchema, null, 2)}`
        : '\n\nYou MUST respond with valid JSON only, no other text.';
      systemText = (systemText || '') + jsonInstruction;
    }

    const maxTokens = req.config?.maxOutputTokens || 8192;
    const temperature = req.config?.temperature;
    const topP = req.config?.topP;
    const topK = req.config?.topK;

    const params: MessageCreateParamsType = {
      model,
      messages,
      max_tokens: maxTokens,
    };

    if (systemText) {
      params.system = systemText;
    }
    if (tools.length > 0) {
      params.tools = tools;
    }
    // Claude API: temperature is mutually exclusive with top_p and top_k.
    // If temperature is set, omit top_p/top_k. Otherwise, allow top_p/top_k.
    if (temperature !== undefined && temperature !== null) {
      params.temperature = temperature;
    } else {
      if (topP !== undefined && topP !== null) {
        params.top_p = topP;
      }
      if (topK !== undefined && topK !== null) {
        params.top_k = topK;
      }
    }

    // Map thinking config
    // thinkingBudget: 0 means "no thinking" in Gemini - skip for Claude
    // thinkingBudget > 0 or thinkingLevel/includeThoughts enables thinking
    const thinkingConfig = req.config?.thinkingConfig;
    if (thinkingConfig) {
      const budget = thinkingConfig.thinkingBudget;
      const hasThinking =
        (budget !== undefined && budget !== null && budget > 0) ||
        thinkingConfig.includeThoughts ||
        thinkingConfig.thinkingLevel;

      if (hasThinking) {
        const MINIMUM_THINKING_BUDGET = 1024;
        let budgetTokens = budget && budget > 0 ? budget : 4096;
        // Claude requires budget_tokens >= 1024 and < max_tokens
        budgetTokens = Math.max(MINIMUM_THINKING_BUDGET, budgetTokens);
        if (budgetTokens >= maxTokens) {
          // Increase max_tokens to accommodate thinking budget
          params.max_tokens = budgetTokens + maxTokens;
        }
        // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
        (params as unknown as Record<string, unknown>)['thinking'] = {
          type: 'enabled',
          budget_tokens: budgetTokens,
        };
      }
    }

    return params;
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const client = await this.getClient();
    const baseParams = this.buildRequest(request);
    const params = {
      ...baseParams,
      stream: false as const,
    } as MessageCreateParamsType;

    // Pass abort signal to the Anthropic SDK via request options
    const abortSignal = request.config?.abortSignal;
    const requestOptions = abortSignal ? { signal: abortSignal } : undefined;

    const message = await client.messages.create(params, requestOptions);

    return translateResponseToGemini(message as MessageType);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const client = await this.getClient();
    const params = this.buildRequest(request);

    // Pass abort signal to the Anthropic SDK via request options
    const abortSignal = request.config?.abortSignal;
    const requestOptions = abortSignal ? { signal: abortSignal } : undefined;

    // Use the streaming API
    const stream = client.messages.stream(params, requestOptions);

    return this.processStream(stream);
  }

  private async *processStream(
    stream: ReturnType<AnthropicVertexType['messages']['stream']>,
  ): AsyncGenerator<GenerateContentResponse> {
    // Buffer for accumulating tool_use input JSON deltas
    const toolUseBuffers = new Map<
      number,
      { id: string; name: string; inputJson: string }
    >();

    let currentBlockIndex = -1;
    let inputTokens = 0;
    let outputTokens = 0;

    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    for await (const event of stream as AsyncIterable<MessageStreamEventType>) {
      switch (event.type) {
        case 'message_start': {
          if (event.message?.usage) {
            inputTokens = event.message.usage.input_tokens || 0;
          }
          break;
        }

        case 'content_block_start': {
          currentBlockIndex = event.index;
          const block = event.content_block;

          if (block.type === 'tool_use') {
            // Start buffering tool use
            toolUseBuffers.set(currentBlockIndex, {
              id: block.id,
              name: block.name,
              inputJson: '',
            });
          }
          // Note: thinking blocks are handled via thinking_delta events,
          // not at content_block_start (which has no initial text).
          break;
        }

        case 'content_block_delta': {
          const delta = event.delta;
          if (delta.type === 'text_delta') {
            // Emit text deltas immediately
            yield this.makeChunk([{ text: delta.text }]);
          } else if (delta.type === 'input_json_delta') {
            // Buffer tool use input JSON
            const buffer = toolUseBuffers.get(event.index);
            if (buffer) {
              buffer.inputJson += delta.partial_json;
            }
          } else if (delta.type === 'thinking_delta') {
            // Emit thinking deltas
            // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
            const thinkingText = (delta as unknown as Record<string, string>)[
              'thinking'
            ];
            yield this.makeChunk([{ text: thinkingText, thought: true }]);
          }
          break;
        }

        case 'content_block_stop': {
          // If this was a tool_use block, emit the complete function call
          const buffer = toolUseBuffers.get(event.index);
          if (buffer) {
            let parsedInput: Record<string, unknown> = {};
            try {
              if (buffer.inputJson) {
                // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
                parsedInput = JSON.parse(buffer.inputJson) as Record<
                  string,
                  unknown
                >;
              }
            } catch {
              debugLogger.warn(
                `[ClaudeContentGenerator] Failed to parse tool_use input JSON: ${buffer.inputJson}`,
              );
            }

            yield this.makeChunk([
              {
                functionCall: {
                  id: buffer.id,
                  name: buffer.name,
                  args: parsedInput,
                },
              },
            ]);
            toolUseBuffers.delete(event.index);
          }
          break;
        }

        case 'message_delta': {
          if (event.usage) {
            outputTokens = event.usage.output_tokens || 0;
          }

          // Emit final chunk with finish reason and usage
          const stopReason = event.delta?.stop_reason;
          // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
          yield {
            candidates: [
              {
                content: { role: 'model', parts: [] },
                finishReason: translateStopReason(stopReason),
              },
            ],
            usageMetadata: {
              promptTokenCount: inputTokens,
              candidatesTokenCount: outputTokens,
              totalTokenCount: inputTokens + outputTokens,
            },
          } as unknown as GenerateContentResponse;
          break;
        }

        case 'message_stop': {
          // Stream complete
          break;
        }

        default:
          break;
      }
    }
  }

  private makeChunk(parts: Part[]): GenerateContentResponse {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    return {
      candidates: [
        {
          content: { role: 'model', parts },
        },
      ],
    } as unknown as GenerateContentResponse;
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Claude has no countTokens API on Vertex.
    // Use character-based estimation as fallback.
    const contents = Array.isArray(request.contents)
      ? // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
        (request.contents as Content[])
      : [];
    const allParts = contents.flatMap((c) => c.parts || []);
    const tokenCount = estimateTokenCountSync(allParts);
    return {
      totalTokens: tokenCount,
    } as CountTokensResponse;
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error(
      'Embedding is not supported with Claude models. ' +
        'Memory features requiring embeddings are not available when using Claude.',
    );
  }
}
