import seaborn as sns

MODEL_NICKNAME_TO_NAME = {
  'claude-3': 'Claude 3.5 Sonnet',
  'deepseek-r1': 'DeepSeek-R1',
  'gemini-1': 'Gemini 1.5 Pro',
  'gpt-4o': 'GPT-4o',
  'grok-3': 'Grok 3',
  'llama-4-maverick': 'Llama 4 Maverick',
  'qwen-3': 'Qwen3',
}

color_palette = sns.color_palette("tab10", len(MODEL_NICKNAME_TO_NAME)+2)
MODEL_NAME_TO_COLOR = {
  'Claude 3.5 Sonnet': color_palette[0],
  'DeepSeek-R1': color_palette[1],
  'GPT-4o': color_palette[3],
  'Gemini 1.5 Pro': color_palette[2],
  'Grok 3': color_palette[4],
  'Llama 4 Maverick': color_palette[5],
  'Qwen3': color_palette[6],
  'Women': color_palette[7],
  'Men': color_palette[7],
  'Humans': color_palette[7],
  'Human': color_palette[7],
  'Children': color_palette[7],
  'Adults': color_palette[7],
}

MODEL_NAME_TO_HATCH = lambda model: (
  '///' if model == 'Women' else
  '\\\\\\' if model == 'Men' else
  'xx' if model.startswith('Human') else
  '--' if model == 'Children' else
  '||' if model == 'Adults' else
  None
)