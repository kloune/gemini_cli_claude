================================================================================
  Claude Model Support via Vertex AI - Patch for Gemini CLI
================================================================================

PATCH FILE:   claude-support.patch
BASE COMMIT:  717660997d652d62c89868272dff293aaa621965
              feat(sandbox): add experimental LXC container sandbox support (#20735)
BRANCH:       main
DATE:         2026-03-05


PREREQUISITES
-------------

1. Clone or checkout the gemini-cli repository at the base commit:

     git clone https://github.com/google-gemini/gemini-cli.git
     cd gemini-cli
     git checkout 717660997d652d62c89868272dff293aaa621965

2. Ensure Node.js >= 20 is installed.


HOW TO APPLY THE PATCH
----------------------

From the repository root (gemini-cli/):

     git apply patch/claude-support.patch

If there are whitespace warnings, use:

     git apply --whitespace=nowarn patch/claude-support.patch

To preview what the patch will do without applying:

     git apply --stat patch/claude-support.patch
     git apply --check patch/claude-support.patch


AFTER APPLYING
--------------

1. Install dependencies (the patch adds @anthropic-ai/vertex-sdk):

     npm install

2. Build:

     npm run build

3. Run tests:

     cd packages/core
     npx vitest run


HOW TO REVERT THE PATCH
------------------------

     git apply -R patch/claude-support.patch
     npm install


WHAT THE PATCH CONTAINS
-----------------------

25 files changed (23 modified, 2 new), ~2400 lines.

New files:
  - packages/core/src/core/claudeContentGenerator.ts     (~580 lines)
    Adapter implementing ContentGenerator interface for Claude on Vertex AI.
    Translates Gemini API types <-> Anthropic Messages API types.

  - packages/core/src/core/claudeContentGenerator.test.ts (~350 lines)
    26 unit tests for the adapter.

Modified files - Implementation:
  - packages/core/package.json                  - Added @anthropic-ai/vertex-sdk dependency
  - packages/core/src/config/models.ts          - Claude model constants, aliases, isClaudeModel()
  - packages/core/src/config/defaultModelConfigs.ts - Claude model configs (opus, sonnet, haiku)
  - packages/core/src/core/contentGenerator.ts  - Claude routing in createContentGenerator()
  - packages/core/src/core/geminiChat.ts        - Skip thoughtSignatures for Claude
  - packages/core/src/core/tokenLimits.ts       - 200k context window for Claude
  - packages/core/src/config/config.ts          - Provider switch detection in setModel()
  - packages/core/src/prompts/snippets.ts       - Model-aware preamble identity
  - packages/core/src/prompts/promptProvider.ts - Pass model name to preamble
  - packages/core/src/tools/web-search.ts       - External search API path for Claude
  - packages/core/src/tools/web-fetch.ts        - Fallback/experimental path for Claude
  - packages/core/src/utils/retry.ts            - Anthropic error retryability comment+guard
  - packages/cli/src/ui/components/ModelDialog.tsx - Claude models in Vertex AI picker
  - package-lock.json                           - Updated lockfile

Modified files - Tests:
  - packages/core/src/config/models.test.ts          - 17 new Claude model tests
  - packages/core/src/core/contentGenerator.test.ts  - 4 new Claude auth tests
  - packages/core/src/config/config.test.ts          - 3 new provider switching tests
  - packages/core/src/utils/retry.test.ts            - 4 new Anthropic error tests
  - packages/core/src/prompts/promptProvider.test.ts - 2 new preamble tests
  - packages/core/src/tools/web-search.test.ts       - 2 new Claude search tests
  - packages/core/src/tools/web-fetch.test.ts        - 2 new Claude fetch tests

Modified files - Test infrastructure:
  - packages/core/src/services/test-data/resolved-aliases.golden.json
  - packages/core/src/services/test-data/resolved-aliases-retry.golden.json


GCP SETUP — VERTEX AI AUTHENTICATION
-------------------------------------

Claude models on Vertex AI use the same GCP credentials as Gemini on Vertex AI.
The Anthropic Vertex SDK authenticates via Google Application Default Credentials
(ADC), which are set up through the gcloud CLI.

### Step 1: Install the gcloud CLI

  If not already installed, follow:
  https://cloud.google.com/sdk/docs/install

### Step 2: Authenticate with GCP

  Log in to your Google Cloud account:

    gcloud auth login

  Set up Application Default Credentials (ADC) — this is what the SDK uses:

    gcloud auth application-default login

  Both commands open a browser for OAuth. The ADC credentials are stored at
  ~/.config/gcloud/application_default_credentials.json and are automatically
  picked up by the Anthropic Vertex SDK (via google-auth-library).

### Step 3: Select or create a GCP project

  List your projects:

    gcloud projects list

  Set your active project:

    gcloud config set project YOUR_PROJECT_ID

### Step 4: Enable the Vertex AI API

  The Vertex AI API must be enabled on your project:

    gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID

### Step 5: Verify Claude model access

  Claude models on Vertex AI require access through the Anthropic model garden.
  Go to the Google Cloud Console:
    https://console.cloud.google.com/vertex-ai/publishers/anthropic/model-garden

  Enable the specific Claude model(s) you want to use. This grants your project
  permission to call Claude via the Vertex AI API.

### Step 6: Set environment variables

  These are REQUIRED for Claude on Vertex AI:

    export GOOGLE_GENAI_USE_VERTEXAI=true
    export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
    export GOOGLE_CLOUD_LOCATION=REGION

  GOOGLE_CLOUD_LOCATION determines the Vertex AI regional endpoint. Claude is
  available in these regions (check for latest availability):
    - global              (routes to nearest available region)
    - us-east5            (Columbus, Ohio)
    - us-central1         (Iowa)
    - europe-west1        (Belgium)
    - europe-west4        (Netherlands)
    - asia-southeast1     (Singapore)

  Example with all variables:

    export GOOGLE_GENAI_USE_VERTEXAI=true
    export GOOGLE_CLOUD_PROJECT=my-ai-project-123
    export GOOGLE_CLOUD_LOCATION=us-east5

  You can also add these to a .env file in your project root — gemini_cli
  loads .env files automatically via dotenv.

### Quick start (all commands together)

    # One-time setup
    gcloud auth login
    gcloud auth application-default login
    gcloud config set project YOUR_PROJECT_ID
    gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID

    # Set env vars (add to .bashrc or .env for persistence)
    export GOOGLE_GENAI_USE_VERTEXAI=true
    export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
    export GOOGLE_CLOUD_LOCATION=us-east5

    # Run gemini_cli with Claude
    gemini --model claude-sonnet


USAGE
-----

After applying the patch and configuring GCP auth (see above), Claude models
are available:

  gemini --model claude-sonnet
  gemini --model claude-opus
  gemini --model claude-haiku

Available models and aliases:
  claude-opus    -> claude-opus-4-6
  claude-sonnet  -> claude-sonnet-4-6
  claude-haiku   -> claude-haiku-4-5@20251001

Models can also be selected interactively via the /model command in the UI.
Claude options appear when Vertex AI auth is active.

You can switch models mid-session:
  /model set claude-sonnet
  /model set gemini-2.5-pro

When switching between Gemini and Claude, the content generator is automatically
recreated and conversation history is adapted.


WEB SEARCH WITH CLAUDE
-----------------------

Claude does not support Gemini's built-in googleSearch tool. Instead, when
Claude is active, web search uses the Google Custom Search JSON API.

To enable web search with Claude:

1. Create a Custom Search Engine at:
   https://programmablesearchengine.google.com/

2. Enable the Custom Search JSON API in your GCP project:
   https://console.cloud.google.com/apis/api/customsearch.googleapis.com

3. Create an API key for the Custom Search API:
   https://console.cloud.google.com/apis/credentials

4. Set environment variables:

     export GOOGLE_CSE_API_KEY=your-custom-search-api-key
     export GOOGLE_CSE_CX=your-custom-search-engine-id

Without these variables, web search will return an error message explaining
what needs to be configured.

Web fetch (URL content retrieval) works without additional setup — it uses
direct HTTP fetching when Claude is active, bypassing Gemini's urlContext tool.


LIMITATIONS
-----------

- Claude models are ONLY available via Vertex AI auth (GOOGLE_GENAI_USE_VERTEXAI=true).
  They cannot be used with a GEMINI_API_KEY.
- Embedding features (memory) are not available with Claude. The embedContent
  API will return a descriptive error.
- Token counting uses character-based estimation (Claude has no countTokens API
  on Vertex AI).
- The fallback/model-switching system (for quota exhaustion) only operates with
  Google OAuth auth, not Vertex AI. Claude quota errors will surface directly.
- Cost: All auxiliary LLM calls (classifier, summarizer, compression, etc.)
  also route through Claude when it is the active model. Claude Haiku is used
  for lightweight calls, but may be more expensive than Gemini Flash Lite.
