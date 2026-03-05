/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { MessageBus } from '../confirmation-bus/message-bus.js';
import { WEB_SEARCH_TOOL_NAME } from './tool-names.js';
import type { GroundingMetadata } from '@google/genai';
import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  type ToolInvocation,
  type ToolResult,
} from './tools.js';
import { ToolErrorType } from './tool-error.js';

import { getErrorMessage } from '../utils/errors.js';
import { type Config } from '../config/config.js';
import { getResponseText } from '../utils/partUtils.js';
import { debugLogger } from '../utils/debugLogger.js';
import { WEB_SEARCH_DEFINITION } from './definitions/coreTools.js';
import { resolveToolDeclaration } from './definitions/resolver.js';
import { LlmRole } from '../telemetry/llmRole.js';
import { isClaudeModel } from '../config/models.js';
import { fetchWithTimeout } from '../utils/fetch.js';

interface GroundingChunkWeb {
  uri?: string;
  title?: string;
}

interface GroundingChunkItem {
  web?: GroundingChunkWeb;
  // Other properties might exist if needed in the future
}

interface GroundingSupportSegment {
  startIndex: number;
  endIndex: number;
  text?: string; // text is optional as per the example
}

interface GroundingSupportItem {
  segment?: GroundingSupportSegment;
  groundingChunkIndices?: number[];
  confidenceScores?: number[]; // Optional as per example
}

/**
 * Parameters for the WebSearchTool.
 */
export interface WebSearchToolParams {
  /**
   * The search query.
   */

  query: string;
}

/**
 * Extends ToolResult to include sources for web search.
 */
export interface WebSearchToolResult extends ToolResult {
  sources?: GroundingMetadata extends { groundingChunks: GroundingChunkItem[] }
    ? GroundingMetadata['groundingChunks']
    : GroundingChunkItem[];
}

class WebSearchToolInvocation extends BaseToolInvocation<
  WebSearchToolParams,
  WebSearchToolResult
> {
  constructor(
    private readonly config: Config,
    params: WebSearchToolParams,
    messageBus: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ) {
    super(params, messageBus, _toolName, _toolDisplayName);
  }

  override getDescription(): string {
    return `Searching the web for: "${this.params.query}"`;
  }

  private async executeWithExternalSearch(
    signal: AbortSignal,
  ): Promise<WebSearchToolResult> {
    const apiKey = process.env['GOOGLE_CSE_API_KEY'];
    const cseId = process.env['GOOGLE_CSE_CX'];

    if (!apiKey || !cseId) {
      return {
        llmContent:
          'Error: Web search with Claude requires GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX environment variables. ' +
          'Set up a Google Custom Search Engine at https://programmablesearchengine.google.com/ ' +
          'and enable the Custom Search JSON API in your Google Cloud project.',
        returnDisplay: 'Error: Missing Google Custom Search API credentials.',
        error: {
          message:
            'Web search with Claude requires GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX environment variables.',
          type: ToolErrorType.WEB_SEARCH_FAILED,
        },
      };
    }

    try {
      const searchUrl = `https://www.googleapis.com/customsearch/v1?key=${encodeURIComponent(apiKey)}&cx=${encodeURIComponent(cseId)}&q=${encodeURIComponent(this.params.query)}&num=5`;

      const response = await fetchWithTimeout(searchUrl, 10000, { signal });
      if (!response.ok) {
        throw new Error(
          `Custom Search API returned status ${response.status}`,
        );
      }

      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      const data = (await response.json()) as {
        items?: Array<{
          title?: string;
          link?: string;
          snippet?: string;
        }>;
      };

      if (!data.items || data.items.length === 0) {
        return {
          llmContent: `No search results found for query: "${this.params.query}"`,
          returnDisplay: 'No results found.',
        };
      }

      // Format search results
      const formattedResults = data.items
        .map(
          (item, i) =>
            `[${i + 1}] ${item.title || 'Untitled'}\n${item.link || ''}\n${item.snippet || ''}`,
        )
        .join('\n\n');

      // Pass results to LLM for synthesis (use web-fetch-fallback config
      // which doesn't include googleSearch tool that Claude can't use)
      const geminiClient = this.config.getGeminiClient();
      const synthesisPrompt = `Synthesize the following search results for the query "${this.params.query}" into a clear, accurate answer. Include inline citation numbers (e.g., [1], [2]) referencing the sources below. Focus on directly answering the query.\n\nSearch Results:\n${formattedResults}`;

      const llmResponse = await geminiClient.generateContent(
        { model: 'web-fetch-fallback' },
        [{ role: 'user', parts: [{ text: synthesisPrompt }] }],
        signal,
        LlmRole.UTILITY_TOOL,
      );

      const responseText = getResponseText(llmResponse) || formattedResults;

      const sources = data.items.map((item) => ({
        web: { uri: item.link, title: item.title },
      }));

      const sourceListFormatted = data.items.map(
        (item, i) =>
          `[${i + 1}] ${item.title || 'Untitled'} (${item.link || 'No URI'})`,
      );

      const fullResponse =
        responseText + '\n\nSources:\n' + sourceListFormatted.join('\n');

      return {
        llmContent: `Web search results for "${this.params.query}":\n\n${fullResponse}`,
        returnDisplay: `Search results for "${this.params.query}" returned.`,
        sources,
      };
    } catch (error: unknown) {
      const errorMessage = `Error during web search for query "${this.params.query}": ${getErrorMessage(error)}`;
      debugLogger.warn(errorMessage, error);
      return {
        llmContent: `Error: ${errorMessage}`,
        returnDisplay: `Error performing web search.`,
        error: {
          message: errorMessage,
          type: ToolErrorType.WEB_SEARCH_FAILED,
        },
      };
    }
  }

  async execute(signal: AbortSignal): Promise<WebSearchToolResult> {
    // Claude models don't support Gemini's googleSearch - use external search API
    if (isClaudeModel(this.config.getModel())) {
      return this.executeWithExternalSearch(signal);
    }

    const geminiClient = this.config.getGeminiClient();

    try {
      const response = await geminiClient.generateContent(
        { model: 'web-search' },
        [{ role: 'user', parts: [{ text: this.params.query }] }],
        signal,
        LlmRole.UTILITY_TOOL,
      );

      const responseText = getResponseText(response);
      const groundingMetadata = response.candidates?.[0]?.groundingMetadata;
      const sources = groundingMetadata?.groundingChunks as
        | GroundingChunkItem[]
        | undefined;
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      const groundingSupports = groundingMetadata?.groundingSupports as
        | GroundingSupportItem[]
        | undefined;

      if (!responseText || !responseText.trim()) {
        return {
          llmContent: `No search results or information found for query: "${this.params.query}"`,
          returnDisplay: 'No information found.',
        };
      }

      let modifiedResponseText = responseText;
      const sourceListFormatted: string[] = [];

      if (sources && sources.length > 0) {
        sources.forEach((source: GroundingChunkItem, index: number) => {
          const title = source.web?.title || 'Untitled';
          const uri = source.web?.uri || 'No URI';
          sourceListFormatted.push(`[${index + 1}] ${title} (${uri})`);
        });

        if (groundingSupports && groundingSupports.length > 0) {
          const insertions: Array<{ index: number; marker: string }> = [];
          groundingSupports.forEach((support: GroundingSupportItem) => {
            if (support.segment && support.groundingChunkIndices) {
              const citationMarker = support.groundingChunkIndices
                .map((chunkIndex: number) => `[${chunkIndex + 1}]`)
                .join('');
              insertions.push({
                index: support.segment.endIndex,
                marker: citationMarker,
              });
            }
          });

          // Sort insertions by index in descending order to avoid shifting subsequent indices
          insertions.sort((a, b) => b.index - a.index);

          // Use TextEncoder/TextDecoder since segment indices are UTF-8 byte positions
          const encoder = new TextEncoder();
          const responseBytes = encoder.encode(modifiedResponseText);
          const parts: Uint8Array[] = [];
          let lastIndex = responseBytes.length;
          for (const ins of insertions) {
            const pos = Math.min(ins.index, lastIndex);
            parts.unshift(responseBytes.subarray(pos, lastIndex));
            parts.unshift(encoder.encode(ins.marker));
            lastIndex = pos;
          }
          parts.unshift(responseBytes.subarray(0, lastIndex));

          // Concatenate all parts into a single buffer
          const totalLength = parts.reduce((sum, part) => sum + part.length, 0);
          const finalBytes = new Uint8Array(totalLength);
          let offset = 0;
          for (const part of parts) {
            finalBytes.set(part, offset);
            offset += part.length;
          }
          modifiedResponseText = new TextDecoder().decode(finalBytes);
        }

        if (sourceListFormatted.length > 0) {
          modifiedResponseText +=
            '\n\nSources:\n' + sourceListFormatted.join('\n');
        }
      }

      return {
        llmContent: `Web search results for "${this.params.query}":\n\n${modifiedResponseText}`,
        returnDisplay: `Search results for "${this.params.query}" returned.`,
        sources,
      };
    } catch (error: unknown) {
      const errorMessage = `Error during web search for query "${
        this.params.query
      }": ${getErrorMessage(error)}`;
      debugLogger.warn(errorMessage, error);
      return {
        llmContent: `Error: ${errorMessage}`,
        returnDisplay: `Error performing web search.`,
        error: {
          message: errorMessage,
          type: ToolErrorType.WEB_SEARCH_FAILED,
        },
      };
    }
  }
}

/**
 * A tool to perform web searches using Google Search via the Gemini API.
 */
export class WebSearchTool extends BaseDeclarativeTool<
  WebSearchToolParams,
  WebSearchToolResult
> {
  static readonly Name = WEB_SEARCH_TOOL_NAME;

  constructor(
    private readonly config: Config,
    messageBus: MessageBus,
  ) {
    super(
      WebSearchTool.Name,
      'GoogleSearch',
      WEB_SEARCH_DEFINITION.base.description!,
      Kind.Search,
      WEB_SEARCH_DEFINITION.base.parametersJsonSchema,
      messageBus,
      true, // isOutputMarkdown
      false, // canUpdateOutput
    );
  }

  /**
   * Validates the parameters for the WebSearchTool.
   * @param params The parameters to validate
   * @returns An error message string if validation fails, null if valid
   */
  protected override validateToolParamValues(
    params: WebSearchToolParams,
  ): string | null {
    if (!params.query || params.query.trim() === '') {
      return "The 'query' parameter cannot be empty.";
    }
    return null;
  }

  protected createInvocation(
    params: WebSearchToolParams,
    messageBus: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ): ToolInvocation<WebSearchToolParams, WebSearchToolResult> {
    return new WebSearchToolInvocation(
      this.config,
      params,
      messageBus ?? this.messageBus,
      _toolName,
      _toolDisplayName,
    );
  }

  override getSchema(modelId?: string) {
    return resolveToolDeclaration(WEB_SEARCH_DEFINITION, modelId);
  }
}
