# SatQuery AI Architecture & System Design
**ISRO/SAC Problem Statement 26167: Agentic Vision-Language Assistant for Earth Observation**

SatQuery AI is designed as a modular, observable, and strictly grounded remote sensing intelligence system. It combines an instruction-tuned LLM planner, a two-stream deep learning fusion network, a Siamese change-detection network, a specialized satellite VLM, and an automated audit trace.

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    User([User / Analyst / GIS System]) -->|Natural Language Query & GeoTIFFs| API[FastAPI REST Gateway]
    
    subgraph Ingestion & Preprocessing
        API --> Ingest[RasterIO Metadata & Band Extraction]
        Ingest --> ModDetect[Modality Detector: Optical vs SAR]
        ModDetect --> ValPipeline[Validation Pipeline]
        ValPipeline --> Reproject[Auto-Reprojection to WGS84 EPSG:4326]
        ValPipeline --> SARPre[SAR Calibration: Linear to dB + 3x3 Lee Filter]
        ValPipeline --> CoregCheck[Spatial Overlap IoU & Resolution Validation]
    end

    subgraph Agentic Orchestration Subsystem
        CoregCheck --> LLMPlanner[LLM Controller & Multi-Step Planner]
        LLMPlanner -->|Pydantic Schema Validation| JSONCheck{Valid JSON & Whitelist?}
        JSONCheck -->|Yes| PlanGraph[Ordered Tool Dependency Graph]
        JSONCheck -->|No / Timeout / Violation| RouterFallback[RuleBasedRouter Fallback]
        RouterFallback --> PlanGraph
    end

    subgraph Tool Execution Engine & Models
        PlanGraph --> Reg[Tool Registry & Whitelist Enforcer]
        Reg --> T1[rs_vqa: Base VLM + LoRA]
        Reg --> T2[rs_caption: Scene Descriptor]
        Reg --> T3[rs_grounding: Text-Guided Bounding Box]
        Reg --> T4[change_map: Siamese CDNet]
        Reg --> T5[change_vqa: Multitemporal RCVA]
        Reg --> T6[optical_sar_fusion: Two-Stream ResNet+Fusion Head]
        Reg --> T7[spectral_index: Band-Specific NDWI/NDVI/NDBI]
    end

    subgraph Synthesis & Provenance
        T1 & T2 & T3 & T4 & T5 & T6 & T7 --> Synth[Evidence Fusion & Metrics Aggregator]
        Synth --> ConfScorer[Config-Driven Multi-Factor Confidence Scorer]
        ConfScorer --> TraceLogger[Observable Trace Logger]
        TraceLogger --> PDFGen[ReportLab PDF Generator with Overlays]
    end

    PDFGen --> Response([QueryResponse: Answer, Layers, Metrics, PDF, Trace])
    Response --> User
```

---

## 2. Multi-Step Agentic Planning Sequence

The following sequence illustrates how compound queries such as *"what changed and is the new area water or built-up?"* are planned and executed across data dependencies:

```mermaid
sequenceDiagram
    autonumber
    actor Analyst as User / Client
    participant Controller as LLM Planner
    participant Engine as Controller Engine
    participant CDNet as change_map (Siamese CD)
    participant Fusion as optical_sar_fusion (Two-Stream)
    participant VQA as change_vqa
    participant Trace as Trace Logger

    Analyst->>Controller: "what changed and is the new area water or built-up?"
    Controller->>Controller: Parse query + check image modalities (Opt + SAR)
    Controller-->>Engine: ToolCallPlan: [change_map -> optical_sar_fusion -> change_vqa]
    
    rect rgb(240, 248, 255)
        Note over Engine,CDNet: Step 1: Pixel-level Change Delineation
        Engine->>CDNet: execute(image_t1, image_t2, threshold=0.45)
        CDNet-->>Engine: change_mask_path, area_changed_m2, direction
        Engine->>Trace: log_tool_call(change_map, 32ms, success)
    end

    rect rgb(255, 248, 240)
        Note over Engine,Fusion: Step 2: Cross-Modal Feature Fusion
        Engine->>Fusion: execute(optical_path, sar_path, target_classes=[water, built_up])
        Fusion-->>Engine: classification_map_path, water_m2, built_up_m2
        Engine->>Trace: log_tool_call(optical_sar_fusion, 84ms, success)
    end

    rect rgb(240, 255, 240)
        Note over Engine,VQA: Step 3: Multi-Step Decomposed Synthesis
        Engine->>Engine: Overlay change_mask & classification_map -> decompose changed pixels
        Engine->>VQA: execute(query, change_mask_path, change_stats)
        VQA-->>Engine: Context-grounded directional answer
        Engine->>Trace: log_tool_call(change_vqa, 15ms, success)
    end

    Engine->>Engine: Compute multi-factor confidence & build PDF report
    Engine-->>Analyst: QueryResponse with decomposed metrics & download link
```

---

## 3. Two-Stream Optical + SAR Cross-Modal Architecture

```mermaid
flowchart LR
    subgraph Optical Stream
        Opt[Optical Multispectral 4-Band] --> OptEnc[ResNet Backbone Optical Encoder]
        OptEnc --> OptFeat[Optical Feature Map 256x64x64]
    end

    subgraph SAR Stream
        SAR[SAR Backscatter Amplitude in dB] --> SAREnc[ResNet Backbone SAR Encoder]
        SAREnc --> SARFeat[SAR Feature Map 256x64x64]
    end

    subgraph Cross-Attention & Fusion Head
        OptFeat --> Concat[Feature Concatenation 512-ch]
        SARFeat --> Concat
        Concat --> SqueezeConv[1x1 Conv + BatchNorm + ReLU]
        SqueezeConv --> FusedFeat[Fused Representation 256-ch]
    end

    subgraph Segmentation Decoder
        FusedFeat --> Up1[Transposed Conv 2x]
        Up1 --> Up2[Transposed Conv 2x]
        Up2 --> ClassHead[1x1 Conv Logits: Background, Water, Built-up]
        ClassHead --> Softmax[Softmax Segmentation Mask]
    end

    Softmax --> WaterPoly[GeoJSON Water Polygons]
    Softmax --> BuiltPoly[GeoJSON Built-up Polygons]
    Softmax --> MaskGeoTIFF[Class Mask GeoTIFF]
```

---

## 4. Multi-Factor Confidence Scoring Model

The confidence score $C \in [0, 1]$ is computed deterministically via `configs/confidence.yaml`:

$$C = w_{\text{token}} \cdot S_{\text{token}} + w_{\text{agreement}} \cdot S_{\text{agreement}} + w_{\text{quality}} \cdot S_{\text{quality}}$$

Where:
- **$S_{\text{token}}$**: Model prediction probability or calibrated baseline.
- **$S_{\text{agreement}}$**: Spatial agreement score between complementary tools (e.g. Optical-SAR Fusion water mask vs NDWI spectral index mask IoU).
- **$S_{\text{quality}}$**: Input sensor quality score ($1.0 - \sum \text{penalties}$ for resampling, missing spectral bands, or low spatial footprint overlap).
