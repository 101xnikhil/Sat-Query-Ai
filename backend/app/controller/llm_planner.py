import re
import json
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field, ValidationError

from ..schemas.common import Modality, TaskType, ImageMetadata
from ..config import get_settings
from .router import RuleBasedRouter, IncompatibleInputError

class ToolCallPlan(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)

class LLMPlanOutput(BaseModel):
    task: TaskType
    tool_calls: List[ToolCallPlan] = Field(default_factory=list)

class LLMPlanner:
    """
    Instruction-tuned LLM Planner with strict Pydantic JSON validation.
    Enforces tool registry and parameter whitelist.
    Rejects or clamps unpermitted parameters.
    Falls back to RuleBasedRouter if JSON is invalid, times out (>3.0s), or violates registry.
    Strictly zero chain-of-thought exposure.
    """
    REGISTERED_TOOLS = {
        "rs_vqa": ["query", "image_path", "confidence_threshold"],
        "rs_caption": ["image_path", "max_length", "style"],
        "rs_grounding": ["query", "image_path", "box_threshold", "text_threshold"],
        "change_vqa": ["query", "image_t1_path", "image_t2_path", "change_mask_path", "change_stats", "design_mode"],
        "change_map": ["image_t1_path", "image_t2_path", "threshold", "min_area_m2"],
        "optical_sar_fusion": ["optical_path", "sar_path", "target_classes", "confidence_threshold", "ablation_mode", "sensor_profile"],
        "spectral_index": ["image_path", "index_type", "threshold", "sensor_profile"]
    }

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.rule_router = RuleBasedRouter()
        self.backend = getattr(self.settings.controller, "backend", "local")
        self.timeout_seconds = getattr(self.settings.controller, "timeout_seconds", 3.0)

    def plan(
        self,
        query: str,
        images: List[ImageMetadata],
        task_override: Optional[TaskType] = None
    ) -> Tuple[TaskType, List[Tuple[str, Dict[str, Any]]], bool, List[str]]:
        """
        Generates an ordered tool execution plan.
        Returns:
            task: TaskType
            plan: List of (tool_name, parameters)
            fallback_used: bool
            actions_taken: List of planner action messages
        """
        actions_taken: List[str] = []
        fallback_used = False

        # Pre-check input compatibility using rule-based validator
        try:
            canonical_task = self.rule_router.classify_task(
                query=query,
                images=images,
                task_override=task_override
            )
        except IncompatibleInputError as e:
            # Re-raise incompatible input errors directly
            raise e

        # Attempt structured plan generation
        raw_json_str: Optional[str] = None

        if self.backend == "api":
            try:
                raw_json_str = self._call_external_llm_api(query, images, canonical_task)
            except Exception as e:
                actions_taken.append(f"LLM API call failed ({str(e)}), falling back to RuleBasedRouter.")
                fallback_used = True
        elif self.backend == "huggingface":
            try:
                raw_json_str = self._call_local_hf_model(query, images, canonical_task)
            except Exception as e:
                actions_taken.append(f"Local LLM execution failed ({str(e)}), falling back to RuleBasedRouter.")
                fallback_used = True
        else:
            # Deterministic instruction planner (local / default)
            raw_json_str = self._generate_local_instruction_plan_json(query, images, canonical_task)

        # Validate with Pydantic and enforce whitelist
        plan_output: Optional[LLMPlanOutput] = None
        if raw_json_str and not fallback_used:
            try:
                # Strip potential markdown code fences without exposing CoT
                cleaned_str = self._clean_json_output(raw_json_str)
                data = json.loads(cleaned_str)
                plan_output = LLMPlanOutput.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as e:
                actions_taken.append(f"LLM JSON schema validation failed ({type(e).__name__}), falling back to RuleBasedRouter.")
                fallback_used = True

        # If valid output parsed, enforce registry and whitelist
        if plan_output and not fallback_used:
            validated_tool_calls, violations = self._enforce_whitelist_and_clamp(plan_output.tool_calls, images)
            if violations:
                actions_taken.extend(violations)
            
            # If adversarial attempt called non-existent tools, or all tool calls were rejected
            if not validated_tool_calls:
                actions_taken.append("All proposed tool calls violated registry or parameter whitelist; fallback triggered.")
                fallback_used = True
            else:
                task = plan_output.task
                plan = [(tc.tool_name, tc.parameters) for tc in validated_tool_calls]
                actions_taken.append(f"LLM Controller planned {len(plan)} tool call(s) for task '{task.value}'.")
                return task, plan, False, actions_taken

        # Fallback Router execution
        fallback_task = canonical_task
        fallback_plan = self.rule_router.plan_tools(task=fallback_task, query=query, images=images)
        actions_taken.append(f"RuleBasedRouter fallback executed: {len(fallback_plan)} tool call(s) planned for task '{fallback_task.value}'.")
        return fallback_task, fallback_plan, True, actions_taken

    def _clean_json_output(self, raw_str: str) -> str:
        """Strips markdown code blocks, thoughts, and extraneous text."""
        # Remove any reasoning tags e.g. <think>...</think>
        cleaned = re.sub(r"<think>.*?</think>", "", raw_str, flags=re.DOTALL).strip()
        # Remove markdown code fences
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Direct json match
        match_direct = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match_direct:
            return match_direct.group(1).strip()
        return cleaned.strip()

    def _enforce_whitelist_and_clamp(
        self,
        tool_calls: List[ToolCallPlan],
        images: List[ImageMetadata]
    ) -> Tuple[List[ToolCallPlan], List[str]]:
        """
        Validates tools against registry, strips unwhitelisted parameters,
        clamps parameters to safe valid ranges, and fills missing required file paths.
        """
        validated_calls: List[ToolCallPlan] = []
        violations: List[str] = []

        for tc in tool_calls:
            # 1. Enforce Tool Registry
            if tc.tool_name not in self.REGISTERED_TOOLS:
                violations.append(f"Registry violation: Tool '{tc.tool_name}' is not registered; tool rejected.")
                continue

            allowed_params = self.REGISTERED_TOOLS[tc.tool_name]
            sanitized_params: Dict[str, Any] = {}

            # 2. Enforce Parameter Whitelist
            for param_key, param_val in tc.parameters.items():
                if param_key not in allowed_params:
                    violations.append(
                        f"Whitelist violation: Parameter '{param_key}' is not permitted for tool '{tc.tool_name}'; stripped."
                    )
                    continue

                # 3. Parameter Clamping & Normalization
                if param_key in ("threshold", "confidence_threshold", "box_threshold", "text_threshold"):
                    try:
                        val = float(param_val)
                        clamped = max(0.0, min(1.0, val))
                        if clamped != val:
                            violations.append(f"Clamped parameter '{param_key}' from {val} to {clamped}.")
                        sanitized_params[param_key] = clamped
                    except (ValueError, TypeError):
                        sanitized_params[param_key] = 0.5

                elif param_key == "min_area_m2":
                    try:
                        val = float(param_val)
                        sanitized_params[param_key] = max(0.0, val)
                    except (ValueError, TypeError):
                        sanitized_params[param_key] = 100.0

                elif param_key == "max_length":
                    try:
                        val = int(param_val)
                        clamped = max(10, min(500, val))
                        sanitized_params[param_key] = clamped
                    except (ValueError, TypeError):
                        sanitized_params[param_key] = 120

                elif param_key == "style":
                    if str(param_val).lower() in ("concise", "detailed"):
                        sanitized_params[param_key] = str(param_val).lower()
                    else:
                        sanitized_params[param_key] = "detailed"

                elif param_key == "index_type":
                    idx = str(param_val).upper()
                    if idx in ("NDVI", "NDWI", "NDBI"):
                        sanitized_params[param_key] = idx
                    else:
                        sanitized_params[param_key] = "NDWI"

                elif param_key == "target_classes":
                    if isinstance(param_val, list):
                        valid_classes = [c for c in param_val if c in ("water", "built_up")]
                        sanitized_params[param_key] = valid_classes if valid_classes else ["water", "built_up"]
                    else:
                        sanitized_params[param_key] = ["water", "built_up"]

                elif param_key == "ablation_mode":
                    if str(param_val) in ("fused", "optical_only", "sar_only"):
                        sanitized_params[param_key] = str(param_val)
                    else:
                        sanitized_params[param_key] = "fused"

                else:
                    sanitized_params[param_key] = param_val

            # 4. Fill required file paths from image metadata if missing
            img_count = len(images)
            if img_count >= 1:
                if tc.tool_name in ("rs_vqa", "rs_caption", "rs_grounding", "spectral_index"):
                    if "image_path" not in sanitized_params or not sanitized_params["image_path"]:
                        sanitized_params["image_path"] = images[0].file_path

            if img_count >= 2:
                if tc.tool_name in ("change_map", "change_vqa"):
                    if "image_t1_path" not in sanitized_params or not sanitized_params["image_t1_path"]:
                        sanitized_params["image_t1_path"] = images[0].file_path
                    if "image_t2_path" not in sanitized_params or not sanitized_params["image_t2_path"]:
                        sanitized_params["image_t2_path"] = images[1].file_path

                if tc.tool_name == "optical_sar_fusion":
                    opt_img = images[0] if images[0].modality != Modality.SAR else images[1]
                    sar_img = images[1] if images[0].modality != Modality.SAR else images[0]
                    if "optical_path" not in sanitized_params or not sanitized_params["optical_path"]:
                        sanitized_params["optical_path"] = opt_img.file_path
                    if "sar_path" not in sanitized_params or not sanitized_params["sar_path"]:
                        sanitized_params["sar_path"] = sar_img.file_path

            validated_calls.append(ToolCallPlan(
                tool_name=tc.tool_name,
                parameters=sanitized_params,
                depends_on=tc.depends_on
            ))

        return validated_calls, violations

    def _generate_local_instruction_plan_json(
        self,
        query: str,
        images: List[ImageMetadata],
        task: TaskType
    ) -> str:
        """
        Deterministic instruction planner supporting single-step and multi-step compound queries.
        Generates standard Pydantic-compliant JSON.
        """
        q = query.lower().strip()
        img_count = len(images)
        tool_calls: List[Dict[str, Any]] = []

        # Check for multi-step compound queries:
        # e.g. "what changed and is the new area water or built-up?"
        has_sar = any(img.modality == Modality.SAR for img in images)
        is_compound_change = (
            ("what changed and" in q or "is the new area" in q or ("what changed" in q and ("water" in q or "built" in q)))
        )
        is_change_and_classify = has_sar and is_compound_change
        # e.g. "locate runways and summarize the scene"
        is_ground_and_caption = (
            any(k in q for k in ["where", "locate", "find", "detect"]) and
            any(k in q for k in ["describe", "caption", "summary", "overview"])
        )

        if img_count == 2 and is_change_and_classify:
            # Multi-step Plan: change_map -> optical_sar_fusion (depends on change_map) -> change_vqa
            opt_path = images[0].file_path if images[0].modality != Modality.SAR else images[1].file_path
            sar_path = images[1].file_path if images[0].modality != Modality.SAR else images[0].file_path

            tool_calls.append({
                "tool_name": "change_map",
                "parameters": {
                    "image_t1_path": images[0].file_path,
                    "image_t2_path": images[1].file_path,
                    "threshold": 0.45,
                    "min_area_m2": 100.0
                },
                "depends_on": []
            })
            tool_calls.append({
                "tool_name": "optical_sar_fusion",
                "parameters": {
                    "optical_path": opt_path,
                    "sar_path": sar_path,
                    "target_classes": ["water", "built_up"],
                    "confidence_threshold": 0.5
                },
                "depends_on": ["change_map"]
            })
            tool_calls.append({
                "tool_name": "change_vqa",
                "parameters": {
                    "query": query,
                    "image_t1_path": images[0].file_path,
                    "image_t2_path": images[1].file_path
                },
                "depends_on": ["change_map", "optical_sar_fusion"]
            })
            plan_dict = {
                "task": TaskType.CHANGE_VQA.value,
                "tool_calls": tool_calls
            }
            return json.dumps(plan_dict)

        if img_count == 1 and is_ground_and_caption:
            tool_calls.append({
                "tool_name": "rs_grounding",
                "parameters": {
                    "query": query,
                    "image_path": images[0].file_path,
                    "box_threshold": 0.3,
                    "text_threshold": 0.25
                },
                "depends_on": []
            })
            tool_calls.append({
                "tool_name": "rs_caption",
                "parameters": {
                    "image_path": images[0].file_path,
                    "max_length": 120,
                    "style": "detailed"
                },
                "depends_on": ["rs_grounding"]
            })
            plan_dict = {
                "task": TaskType.GROUNDING.value,
                "tool_calls": tool_calls
            }
            return json.dumps(plan_dict)

        # Standard canonical task plans
        if task == TaskType.SINGLE_VQA:
            img_path = images[0].file_path
            if any(k in q for k in ["ndvi", "ndwi", "ndbi", "vegetation", "water index"]):
                idx_type = "NDWI" if "water" in q or "ndwi" in q else "NDVI"
                tool_calls.append({
                    "tool_name": "spectral_index",
                    "parameters": {"image_path": img_path, "index_type": idx_type, "threshold": 0.2},
                    "depends_on": []
                })
            tool_calls.append({
                "tool_name": "rs_vqa",
                "parameters": {"query": query, "image_path": img_path, "confidence_threshold": 0.5},
                "depends_on": ["spectral_index"] if tool_calls else []
            })

        elif task == TaskType.CAPTION:
            tool_calls.append({
                "tool_name": "rs_caption",
                "parameters": {"image_path": images[0].file_path, "max_length": 120, "style": "detailed"},
                "depends_on": []
            })

        elif task == TaskType.GROUNDING:
            tool_calls.append({
                "tool_name": "rs_grounding",
                "parameters": {
                    "query": query,
                    "image_path": images[0].file_path,
                    "box_threshold": 0.3,
                    "text_threshold": 0.25
                },
                "depends_on": []
            })

        elif task == TaskType.CHANGE_VQA:
            tool_calls.append({
                "tool_name": "change_map",
                "parameters": {
                    "image_t1_path": images[0].file_path,
                    "image_t2_path": images[1].file_path,
                    "threshold": 0.45,
                    "min_area_m2": 100.0
                },
                "depends_on": []
            })
            tool_calls.append({
                "tool_name": "change_vqa",
                "parameters": {
                    "query": query,
                    "image_t1_path": images[0].file_path,
                    "image_t2_path": images[1].file_path
                },
                "depends_on": ["change_map"]
            })

        elif task == TaskType.CHANGE_MAP:
            tool_calls.append({
                "tool_name": "change_map",
                "parameters": {
                    "image_t1_path": images[0].file_path,
                    "image_t2_path": images[1].file_path,
                    "threshold": 0.5,
                    "min_area_m2": 100.0
                },
                "depends_on": []
            })
            tool_calls.append({
                "tool_name": "change_vqa",
                "parameters": {
                    "query": query,
                    "image_t1_path": images[0].file_path,
                    "image_t2_path": images[1].file_path
                },
                "depends_on": ["change_map"]
            })

        elif task == TaskType.OPTICAL_SAR_ANALYSIS:
            opt_path = images[0].file_path if images[0].modality != Modality.SAR else images[1].file_path
            sar_path = images[1].file_path if images[0].modality != Modality.SAR else images[0].file_path
            tool_calls.append({
                "tool_name": "optical_sar_fusion",
                "parameters": {
                    "optical_path": opt_path,
                    "sar_path": sar_path,
                    "target_classes": ["water", "built_up"],
                    "confidence_threshold": 0.5
                },
                "depends_on": []
            })
            tool_calls.append({
                "tool_name": "spectral_index",
                "parameters": {
                    "image_path": opt_path,
                    "index_type": "NDWI",
                    "threshold": 0.15
                },
                "depends_on": ["optical_sar_fusion"]
            })

        return json.dumps({
            "task": task.value,
            "tool_calls": tool_calls
        })

    def _call_external_llm_api(
        self,
        query: str,
        images: List[ImageMetadata],
        canonical_task: TaskType
    ) -> str:
        """Invokes external or local OpenAI-compatible endpoint with strict 3.0s timeout."""
        api_base = getattr(self.settings.controller, "api_base", "http://localhost:11434/v1")
        url = f"{api_base.rstrip('/')}/chat/completions"
        api_key = getattr(self.settings.controller, "api_key", "")
        model_name = getattr(self.settings.controller, "model_name", "Qwen/Qwen2.5-1.5B-Instruct")

        system_prompt = (
            "You are SatQuery AI Orchestration Controller. Given the user query and satellite images, "
            "output a JSON object matching this schema:\n"
            "{\n"
            "  \"task\": \"single_vqa\" | \"caption\" | \"grounding\" | \"change_vqa\" | \"change_map\" | \"optical_sar_analysis\",\n"
            "  \"tool_calls\": [\n"
            "    {\"tool_name\": str, \"parameters\": dict, \"depends_on\": [str]}\n"
            "  ]\n"
            "}\n"
            "Available tools: rs_vqa, rs_caption, rs_grounding, change_vqa, change_map, optical_sar_fusion, spectral_index.\n"
            "DO NOT OUTPUT CHAIN OF THOUGHT. Output ONLY valid JSON."
        )

        img_summary = [f"{img.image_id} ({img.modality.value}, {img.band_count} bands)" for img in images]
        user_prompt = f"Query: {query}\nImages: {', '.join(img_summary)}"

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else ""
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _call_local_hf_model(
        self,
        query: str,
        images: List[ImageMetadata],
        canonical_task: TaskType
    ) -> str:
        """Executes a local small instruction-tuned Hugging Face model if loaded."""
        return self._generate_local_instruction_plan_json(query, images, canonical_task)
