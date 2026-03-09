/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { FinishReason } from '@google/genai';
import type { GenerateContentParameters, Content } from '@google/genai';

const mockCreate = vi.fn();
const mockStream = vi.fn();
vi.mock('@anthropic-ai/vertex-sdk', () => ({
  AnthropicVertex: vi.fn().mockImplementation(() => ({
    messages: { create: mockCreate, stream: mockStream },
  })),
}));

import { ClaudeContentGenerator } from './claudeContentGenerator.js';
import { LlmRole } from '../telemetry/llmRole.js';

function makeGenerator() {
  return new ClaudeContentGenerator(
    'test-project',
    'us-east5',
    'claude-sonnet-4-6',
  );
}

function makeRequest(
  overrides: Partial<GenerateContentParameters> = {},
): GenerateContentParameters {
  return {
    model: 'claude-sonnet-4-6',
    contents: [
      {
        role: 'user',
        parts: [{ text: 'hello' }],
      },
    ] as Content[],
    config: {},
    ...overrides,
  } as GenerateContentParameters;
}

const defaultResponse = {
  content: [{ type: 'text', text: 'response' }],
  stop_reason: 'end_turn',
  usage: { input_tokens: 10, output_tokens: 5 },
};

describe('ClaudeContentGenerator', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCreate.mockResolvedValue(defaultResponse);
  });

  const generator = makeGenerator();

  async function callAndGetParams(req: GenerateContentParameters) {
    await generator.generateContent(req, 'test-prompt', LlmRole.UTILITY_TOOL);
    return mockCreate.mock.calls[0][0];
  }

  describe('Request Translation', () => {
    it('maps user role to user and model role to assistant', async () => {
      const req = makeRequest({
        contents: [
          { role: 'user', parts: [{ text: 'hi' }] },
          { role: 'model', parts: [{ text: 'hello' }] },
        ] as Content[],
      });
      const params = await callAndGetParams(req);
      expect(params.messages[0].role).toBe('user');
      expect(params.messages[1].role).toBe('assistant');
    });

    it('translates text parts correctly', async () => {
      const req = makeRequest({
        contents: [
          { role: 'user', parts: [{ text: 'test message' }] },
        ] as Content[],
      });
      const params = await callAndGetParams(req);
      expect(params.messages[0].content).toEqual([
        { type: 'text', text: 'test message' },
      ]);
    });

    it('translates functionCall to tool_use', async () => {
      const req = makeRequest({
        contents: [
          {
            role: 'model',
            parts: [
              {
                functionCall: {
                  id: 'call-1',
                  name: 'read_file',
                  args: { path: '/tmp/test' },
                },
              },
            ],
          },
        ] as Content[],
      });
      // Prepend user message since model can't be first

      req.contents = [
        { role: 'user', parts: [{ text: 'start' }] },
        ...(req.contents as Content[]),
      ] as Content[];
      const params = await callAndGetParams(req);
      const assistantContent = params.messages[1].content;
      expect(assistantContent[0].type).toBe('tool_use');
      expect(assistantContent[0].id).toBe('call-1');
      expect(assistantContent[0].name).toBe('read_file');
      expect(assistantContent[0].input).toEqual({ path: '/tmp/test' });
    });

    it('translates functionResponse to tool_result with stringified object', async () => {
      const responseObj = { result: 'ok' };
      const req = makeRequest({
        contents: [
          {
            role: 'user',
            parts: [
              {
                functionResponse: {
                  id: 'call-1',
                  name: 'read_file',
                  response: responseObj,
                },
              },
            ],
          },
        ] as Content[],
      });
      const params = await callAndGetParams(req);
      const block = params.messages[0].content[0];
      expect(block.type).toBe('tool_result');
      expect(block.tool_use_id).toBe('call-1');
      expect(block.content).toBe(JSON.stringify(responseObj));
    });

    it('translates functionResponse with string response as-is', async () => {
      const req = makeRequest({
        contents: [
          {
            role: 'user',
            parts: [
              {
                functionResponse: {
                  id: 'call-2',
                  name: 'run_cmd',
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  response: 'done' as any,
                },
              },
            ],
          },
        ] as Content[],
      });
      const params = await callAndGetParams(req);
      expect(params.messages[0].content[0].content).toBe('done');
    });

    it('strips thought parts from messages', async () => {
      const req = makeRequest({
        contents: [
          {
            role: 'model',
            parts: [
              { text: 'thinking...', thought: true },
              { text: 'visible answer' },
            ],
          },
        ] as Content[],
      });

      req.contents = [
        { role: 'user', parts: [{ text: 'q' }] },
        ...(req.contents as Content[]),
      ] as Content[];
      const params = await callAndGetParams(req);
      const assistantBlocks = params.messages[1].content;
      expect(assistantBlocks).toHaveLength(1);
      expect(assistantBlocks[0].text).toBe('visible answer');
    });

    it('passes system instruction as system param', async () => {
      const req = makeRequest({
        config: {
          systemInstruction: {
            role: 'user',
            parts: [{ text: 'You are helpful.' }],
          },
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.system).toBe('You are helpful.');
    });

    it('merges consecutive same-role messages', async () => {
      const req = makeRequest({
        contents: [
          { role: 'user', parts: [{ text: 'first' }] },
          { role: 'user', parts: [{ text: 'second' }] },
        ] as Content[],
      });
      const params = await callAndGetParams(req);
      expect(params.messages).toHaveLength(1);
      expect(params.messages[0].content).toHaveLength(2);
      expect(params.messages[0].content[0].text).toBe('first');
      expect(params.messages[0].content[1].text).toBe('second');
    });
  });

  describe('Tool Translation', () => {
    it('translates functionDeclarations to tools with input_schema', async () => {
      const req = makeRequest({
        config: {
          tools: [
            {
              functionDeclarations: [
                {
                  name: 'read_file',
                  description: 'Reads a file',
                  parameters: {
                    type: 'object',
                    properties: { path: { type: 'string' } },
                  },
                },
              ],
            },
          ],
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.tools).toHaveLength(1);
      expect(params.tools[0].name).toBe('read_file');
      expect(params.tools[0].description).toBe('Reads a file');
      expect(params.tools[0].input_schema).toEqual({
        type: 'object',
        properties: { path: { type: 'string' } },
      });
    });

    it('filters out googleSearch tools', async () => {
      const req = makeRequest({
        config: {
          tools: [
            { googleSearch: {} },
            {
              functionDeclarations: [{ name: 'keep_me', description: 'stays' }],
            },
          ],
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.tools).toHaveLength(1);
      expect(params.tools[0].name).toBe('keep_me');
    });

    it('filters out urlContext tools', async () => {
      const req = makeRequest({
        config: {
          tools: [{ urlContext: {} }],
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.tools).toBeUndefined();
    });
  });

  describe('Response Translation', () => {
    it('translates text content block to Part with text', async () => {
      mockCreate.mockResolvedValue({
        content: [{ type: 'text', text: 'hello world' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 5, output_tokens: 3 },
      });
      const result = await generator.generateContent(
        makeRequest(),
        'test-prompt',
        LlmRole.UTILITY_TOOL,
      );
      const parts = result.candidates![0].content!.parts!;
      expect(parts[0].text).toBe('hello world');
    });

    it('translates tool_use block to functionCall Part', async () => {
      mockCreate.mockResolvedValue({
        content: [
          {
            type: 'tool_use',
            id: 'tu-1',
            name: 'write_file',
            input: { path: '/tmp/f', content: 'data' },
          },
        ],
        stop_reason: 'tool_use',
        usage: { input_tokens: 8, output_tokens: 4 },
      });
      const result = await generator.generateContent(
        makeRequest(),
        'test-prompt',
        LlmRole.UTILITY_TOOL,
      );
      const fc = result.candidates![0].content!.parts![0].functionCall!;
      expect(fc.id).toBe('tu-1');
      expect(fc.name).toBe('write_file');
      expect(fc.args).toEqual({ path: '/tmp/f', content: 'data' });
    });

    it('maps stop_reason end_turn to STOP', async () => {
      mockCreate.mockResolvedValue({
        ...defaultResponse,
        stop_reason: 'end_turn',
      });
      const result = await generator.generateContent(
        makeRequest(),
        'test-prompt',
        LlmRole.UTILITY_TOOL,
      );
      expect(result.candidates![0].finishReason).toBe(FinishReason.STOP);
    });

    it('maps stop_reason max_tokens to MAX_TOKENS', async () => {
      mockCreate.mockResolvedValue({
        ...defaultResponse,
        stop_reason: 'max_tokens',
      });
      const result = await generator.generateContent(
        makeRequest(),
        'test-prompt',
        LlmRole.UTILITY_TOOL,
      );
      expect(result.candidates![0].finishReason).toBe(FinishReason.MAX_TOKENS);
    });

    it('populates usage metadata', async () => {
      const result = await generator.generateContent(
        makeRequest(),
        'test-prompt',
        LlmRole.UTILITY_TOOL,
      );
      expect(result.usageMetadata!.promptTokenCount).toBe(10);
      expect(result.usageMetadata!.candidatesTokenCount).toBe(5);
      expect(result.usageMetadata!.totalTokenCount).toBe(15);
    });
  });

  describe('JSON Mode', () => {
    it('adds JSON instruction to system when responseMimeType is application/json', async () => {
      const req = makeRequest({
        config: {
          systemInstruction: 'Base instruction.',
          responseMimeType: 'application/json',
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.system).toContain('Base instruction.');
      expect(params.system).toContain('You MUST respond with valid JSON only');
    });
  });

  describe('Thinking Config', () => {
    it('enables thinking when thinkingBudget > 0', async () => {
      const req = makeRequest({
        config: {
          thinkingConfig: { thinkingBudget: 2048 },
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.thinking).toEqual({
        type: 'enabled',
        budget_tokens: 2048,
      });
    });

    it('does NOT enable thinking when thinkingBudget is 0', async () => {
      const req = makeRequest({
        config: {
          thinkingConfig: { thinkingBudget: 0 },
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.thinking).toBeUndefined();
    });

    it('enables thinking with default budget when includeThoughts is set', async () => {
      const req = makeRequest({
        config: {
          thinkingConfig: { includeThoughts: true },
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.thinking).toEqual({
        type: 'enabled',
        budget_tokens: 4096,
      });
    });
  });

  describe('Model Mapping', () => {
    it('maps gemini-2.5-flash to claude-haiku', async () => {
      const req = makeRequest({ model: 'gemini-2.5-flash' });
      const params = await callAndGetParams(req);
      expect(params.model).toContain('haiku');
    });

    it('maps gemini-3-pro-preview to user model', async () => {
      const req = makeRequest({ model: 'gemini-3-pro-preview' });
      const params = await callAndGetParams(req);
      expect(params.model).toBe('claude-sonnet-4-6');
    });

    it('passes through claude model names', async () => {
      const req = makeRequest({ model: 'claude-sonnet-4-6' });
      const params = await callAndGetParams(req);
      expect(params.model).toBe('claude-sonnet-4-6');
    });
  });

  describe('countTokens', () => {
    it('returns an estimate without error', async () => {
      const gen = makeGenerator();
      const result = await gen.countTokens({
        model: 'claude-sonnet-4-6',
        contents: [
          { role: 'user', parts: [{ text: 'hello world' }] },
        ] as Content[],
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      expect(result.totalTokens).toBeGreaterThanOrEqual(0);
    });
  });

  describe('embedContent', () => {
    it('throws a descriptive error', async () => {
      const gen = makeGenerator();
      await expect(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        gen.embedContent({} as any),
      ).rejects.toThrow('Embedding is not supported with Claude models');
    });
  });

  describe('Parameter Exclusivity', () => {
    it('omits top_p and top_k when temperature is set', async () => {
      const req = makeRequest({
        config: {
          temperature: 0.7,
          topP: 0.9,
          topK: 40,
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.temperature).toBe(0.7);
      expect(params.top_p).toBeUndefined();
      expect(params.top_k).toBeUndefined();
    });

    it('includes top_p when temperature is not set', async () => {
      const req = makeRequest({
        config: {
          topP: 0.9,
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.temperature).toBeUndefined();
      expect(params.top_p).toBe(0.9);
    });

    it('includes top_k when temperature is not set', async () => {
      const req = makeRequest({
        config: {
          topK: 40,
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.temperature).toBeUndefined();
      expect(params.top_k).toBe(40);
    });
  });

  describe('Thinking Budget Validation', () => {
    it('clamps budget_tokens to minimum 1024', async () => {
      const req = makeRequest({
        config: {
          thinkingConfig: { thinkingBudget: 500 },
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.thinking.budget_tokens).toBe(1024);
    });

    it('increases max_tokens when budget_tokens >= max_tokens', async () => {
      const req = makeRequest({
        config: {
          maxOutputTokens: 2000,
          thinkingConfig: { thinkingBudget: 4096 },
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      const params = await callAndGetParams(req);
      expect(params.thinking.budget_tokens).toBe(4096);
      expect(params.max_tokens).toBe(6096); // 4096 + 2000
    });
  });

  describe('AbortSignal', () => {
    it('forwards abort signal to create call', async () => {
      const controller = new AbortController();
      const req = makeRequest({
        config: {
          abortSignal: controller.signal,
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any);
      await generator.generateContent(req, 'test-prompt', LlmRole.UTILITY_TOOL);
      const requestOptions = mockCreate.mock.calls[0][1];
      expect(requestOptions).toBeDefined();
      expect(requestOptions.signal).toBe(controller.signal);
    });
  });
});
