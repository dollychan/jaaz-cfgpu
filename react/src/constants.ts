import type { LLMConfig, ToolCallFunctionName } from '@/types/types'

// API Configuration
export const BASE_API_URL =
  import.meta.env.VITE_JAAZ_BASE_API_URL || 'https://jaaz.app'

export const PROVIDER_NAME_MAPPING: {
  [key: string]: { name: string; icon: string }
} = {
  jaaz: {
    name: 'Jaaz',
    icon: 'https://raw.githubusercontent.com/11cafe/jaaz/refs/heads/main/assets/icons/jaaz.png',
  },
  anthropic: {
    name: 'Claude',
    icon: 'https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/claude-color.png',
  },
  openai: { name: 'OpenAI', icon: 'https://openai.com/favicon.ico' },
  replicate: {
    name: 'Replicate',
    icon: 'https://images.seeklogo.com/logo-png/61/1/replicate-icon-logo-png_seeklogo-611690.png',
  },
  ollama: {
    name: 'Ollama',
    icon: 'https://images.seeklogo.com/logo-png/59/1/ollama-logo-png_seeklogo-593420.png',
  },
  huggingface: {
    name: 'Hugging Face',
    icon: 'https://huggingface.co/favicon.ico',
  },
  wavespeed: {
    name: 'WaveSpeedAi',
    icon: 'https://www.wavespeed.ai/favicon.ico',
  },
  volces: {
    name: 'Volces',
    icon: 'https://portal.volccdn.com/obj/volcfe/misc/favicon.png',
  },
  comfyui: {
    name: 'ComfyUI',
    icon: 'https://framerusercontent.com/images/3cNQMWKzIhIrQ5KErBm7dSmbd2w.png',
  },
}

// Tool call name mapping
export const TOOL_CALL_NAME_MAPPING: { [key in ToolCallFunctionName]: string } =
  {
    generate_image: 'Generate Image',
    prompt_user_multi_choice: 'Prompt Multi-Choice',
    prompt_user_single_choice: 'Prompt Single-Choice',
    write_plan: 'Write Plan',
    finish: 'Finish',
  }

export const LOGO_URL = 'https://jaaz.app/favicon.ico'

export const DEFAULT_SYSTEM_PROMPT = `You are an image and video generation executor. Your job is to call the right generation tools immediately.

PROMPT FIDELITY RULE (highest priority):
- The user’s original description is the core of the generation prompt — use it VERBATIM.
- You may only append technical parameters the user did NOT specify (e.g. aspect ratio, resolution).
- You MUST NOT rewrite, expand, paraphrase, or replace the user’s description.
- Do NOT write a “Design Strategy Doc” or any creative brief before generating.

1. If it is an image generation task, call generate_image tool immediately using the user’s original prompt. Choose aspect_ratio that best fits the content if the user did not specify.

2. If it is a video generation task, call a video generation tool immediately using the user’s original prompt. You may generate a reference image first if needed, then pass it to the video tool.

3. CONTENT POLICY ERRORS: If a tool returns a message containing “Content policy violation” or “STOP. Do NOT retry”, you MUST immediately stop all tool calls and inform the user in plain text. Do NOT retry with any other tool or any modified parameters. Your next action MUST be a plain text reply — never a tool call.
`
