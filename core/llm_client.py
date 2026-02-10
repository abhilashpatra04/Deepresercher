"""
LLM Client — Unified interface for Groq and Gemini.
Supports free-tier models for zero-cost demo.
"""

import os
import json
from groq import Groq

# Try new SDK first, fall back to deprecated one
try:
    from google import genai as google_genai
    GEMINI_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as google_genai
        GEMINI_NEW_SDK = False
    except ImportError:
        google_genai = None
        GEMINI_NEW_SDK = False


class LLMClient:
    """Unified LLM client supporting Groq (free) and Gemini (free)."""

    def __init__(self, provider="groq", model=None):
        self.provider = provider

        if provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
            self.client = Groq(api_key=api_key)
            self.model = model or "llama-3.3-70b-versatile"
        elif provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
            self.model = model or "gemini-2.0-flash"
            if GEMINI_NEW_SDK:
                self.client = google_genai.Client(api_key=api_key)
            else:
                google_genai.configure(api_key=api_key)
                self.client = google_genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        """Generate text from prompt. Returns string response."""
        try:
            if self.provider == "groq":
                return self._groq_generate(prompt, system_prompt, temperature)
            elif self.provider == "gemini":
                return self._gemini_generate(prompt, system_prompt, temperature)
        except Exception as e:
            return f"[LLM Error: {str(e)}]"

    def _groq_generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    def _gemini_generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        if GEMINI_NEW_SDK:
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config={"temperature": temperature, "max_output_tokens": 4096},
            )
            return response.text
        else:
            response = self.client.generate_content(
                full_prompt,
                generation_config=google_genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=4096,
                ),
            )
            return response.text

    def generate_json(self, prompt: str, system_prompt: str = None) -> dict:
        """Generate and parse JSON response."""
        json_prompt = prompt + "\n\nRespond ONLY with valid JSON, no markdown formatting."
        response = self.generate(json_prompt, system_prompt, temperature=0.3)

        # Clean response — strip markdown code fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]  # Remove opening ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove closing ```
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw": response}
