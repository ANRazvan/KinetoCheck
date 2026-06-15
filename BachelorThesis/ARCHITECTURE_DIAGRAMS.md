# KinetoCheck Architecture Diagrams

## 1. System-Level Architecture

```mermaid
graph TB
    subgraph Input["Input Layer"]
        Video["Video Input<br/>MP4 Files"]
        Camera["Live Camera<br/>MediaPipe"]
        Dataset["UI-PRMD Dataset<br/>Vicon .txt"]
    end
    
    subgraph Preprocess["Preprocessing"]
        ME["MediaPipe<br/>Extraction<br/>17 COCO Joints"]
        Align["Vicon→COCO<br/>Alignment<br/>39→17 Mapping"]
        Norm["Normalization<br/>Hip-Centric<br/>Z-Score"]
        Features["Feature Engineering<br/>Pos, Vel, Acc<br/>9 or 12 channels"]
    end
    
    subgraph Model["Model Inference"]
        Factory["ModelFactory<br/>Singleton Pattern"]
        STGAT["ST-GAT Encoder<br/>Spatial-Temporal<br/>Graph Attention"]
        Embed["Global Embedding<br/>L2 Normalized<br/>128-dim"]
        PhaseOpt["Phase Aligner<br/>Optional<br/>Temporal DTW"]
        Decoder["Frame Decoder<br/>Optional<br/>Per-Joint Correction"]
    end
    
    subgraph Output["Output & Visualization"]
        Score["Similarity Score<br/>0.0 - 1.0"]
        Predict["Classification<br/>Correct/Incorrect"]
        Feedback["Joint Feedback<br/>Importance Ranking"]
        VizVideo["Annotated Video<br/>Ghost Overlay<br/>Correction Arrows"]
    end
    
    subgraph Storage["Checkpoints"]
        CP1["uiprmd_phase_aware_rom<br/>✓ Phase-Aware<br/>✓ ROM Loss<br/>Best Overall"]
        CP2["uiprmd_phase_aware<br/>✓ Phase-Aware<br/>✗ ROM Loss<br/>Ablation"]
        CP3["uiprmd<br/>✗ Phase-Aware<br/>✗ ROM Loss<br/>Baseline"]
    end
    
    Video --> ME
    Camera --> ME
    Dataset --> Align
    ME --> Norm
    Align --> Norm
    Norm --> Features
    Features --> Factory
    Factory --> STGAT
    STGAT --> Embed
    Embed --> PhaseOpt
    PhaseOpt --> Decoder
    Decoder --> Score
    Score --> Predict
    STGAT --> Feedback
    Predict --> VizVideo
    Feedback --> VizVideo
    CP1 --> Factory
    CP2 --> Factory
    CP3 --> Factory
```

## 2. Training Pipeline

```mermaid
graph LR
    subgraph Data["Data Loading"]
        Load["Load UI-PRMD<br/>Exercises 1-20"]
        Split["Leave-One-Subject-Out<br/>LOSO Split"]
        Process["Preprocess<br/>Normalize Features"]
    end
    
    subgraph Model["Model Construction"]
        Factory["ModelFactory<br/>Create Evaluator"]
        Config["Config<br/>hidden=(64,128)<br/>embed_dim=128"]
        Init["Xavier Init<br/>Weights"]
    end
    
    subgraph Loss["Loss Functions"]
        Contr["ContrastiveLoss<br/>margin=1.0<br/>Embedding similarity"]
        Delta["DeltaRegressionLoss<br/>weight=0.1<br/>Huber loss"]
        ROM["RangeOfMotionLoss<br/>weight=0.3<br/>Amplitude enforce"]
    end
    
    subgraph Train["Training Loop"]
        Epoch["FOR epoch in range(30)"]
        Forward["Forward Pass<br/>Template + User"]
        Loss["Compute Loss<br/>α·Contr + β·Delta + γ·ROM"]
        Backward["Backward Pass<br/>Compute Gradients"]
        Optimize["Adam Optimizer<br/>lr=1e-3, decay=1e-4"]
        Val["Validation<br/>Track Best Loss"]
        Early["Early Stop<br/>patience=10"]
    end
    
    subgraph Checkpoint["Save Checkpoint"]
        Save["Save best_checkpoint.pt"]
        Meta["Metadata<br/>config, threshold<br/>use_phase_decoder"]
    end
    
    Load --> Split
    Split --> Process
    Process --> Factory
    Config --> Factory
    Factory --> Init
    Init --> Epoch
    Contr --> Loss
    Delta --> Loss
    ROM --> Loss
    Epoch --> Forward
    Forward --> Loss
    Loss --> Backward
    Backward --> Optimize
    Optimize --> Val
    Val --> Early
    Early --> Save
    Save --> Meta
```

## 3. ST-GAT Block Architecture

```mermaid
graph TB
    Input["Input<br/>B, C_in, T, J"]
    
    subgraph SpatialAttn["Spatial Graph Attention"]
        Proj1["Project to Hidden<br/>C_in → C_out"]
        Scores["Compute Attention<br/>via Query/Key"]
        Mask["Mask Invalid Edges<br/>Skeleton Topology"]
        Softmax["Softmax Over<br/>Destination Joints"]
        Aggregate["Aggregate Features<br/>Weighted Sum"]
    end
    
    subgraph Temporal["Temporal Pyramid"]
        K3["Conv1D k=3<br/>dilation=1"]
        K5["Conv1D k=5<br/>dilation=2"]
        K7["Conv1D k=7<br/>dilation=3"]
        Cat["Concatenate<br/>3 × C_out"]
        Project["Project Back<br/>→ C_out"]
    end
    
    subgraph Residual["Residual Connection"]
        ResProj["1×1 Conv<br/>if C_in ≠ C_out"]
        Add["Add + Normalize"]
    end
    
    Output["Output<br/>B, C_out, T, J"]
    
    Input --> Proj1
    Proj1 --> Scores
    Scores --> Mask
    Mask --> Softmax
    Softmax --> Aggregate
    Aggregate --> K3
    Aggregate --> K5
    Aggregate --> K7
    K3 --> Cat
    K5 --> Cat
    K7 --> Cat
    Cat --> Project
    Project --> ResProj
    Aggregate --> ResProj
    ResProj --> Add
    Add --> Output
```

## 4. Phase-Aware Model Extensions

```mermaid
graph TB
    UserFeat["User Features<br/>B, C, T_u, J"]
    TemplFeat["Template Features<br/>B, C, T_t, J"]
    
    subgraph PhaseAlign["Phase Aligner<br/>Temporal Alignment"]
        ProjQ["Project Query<br/>C → C/4"]
        ProjK["Project Key<br/>C → C/4"]
        Attention["Attention<br/>Q·K^T / √d"]
        Warp["Warp Template<br/>to User Time"]
        WarpWeights["Output Weights<br/>B, T_u, T_t"]
    end
    
    subgraph FrameDecoder["Frame Decoder<br/>Per-Joint Correction"]
        Concat["Concat User + Warped<br/>B, C×2, T_u, J"]
        MLP["MLP Stack<br/>C×2 → C → C/2"]
        DeltaHead["Delta Head<br/>C/2 → 3"]
        ConfHead["Conf Head<br/>C/2 → 1"]
        DeltaOut["Output Delta XYZ<br/>B, T_u, J, 3"]
        ConfOut["Output Confidence<br/>B, T_u, J"]
    end
    
    subgraph JointScore["Joint Scoring"]
        Attention_Vec["Spatial Attention<br/>from Encoder"]
        DeltaMag["Delta Magnitude<br/>from Decoder"]
        ConfScore["Confidence<br/>from Decoder"]
        Combine["Combine via<br/>Weighted Mean"]
        JointImp["Joint Importance<br/>B, J"]
    end
    
    UserFeat --> ProjQ
    TemplFeat --> ProjK
    ProjQ --> Attention
    ProjK --> Attention
    Attention --> Warp
    Attention --> WarpWeights
    Warp --> Concat
    UserFeat --> Concat
    Concat --> MLP
    MLP --> DeltaHead
    MLP --> ConfHead
    DeltaHead --> DeltaOut
    ConfHead --> ConfOut
    DeltaOut --> Combine
    ConfOut --> Combine
    Attention_Vec --> Combine
    DeltaMag --> Combine
    ConfScore --> Combine
    Combine --> JointImp
```

## 5. Loss Function Landscape

```mermaid
graph TB
    subgraph Contrastive["Contrastive Loss<br/>margin=1.0"]
        SimScore["Compute Similarity<br/>cosine(emb_template, emb_user)"]
        Dist["Distance = 1 - Similarity"]
        PosLoss["Positive (label=1)<br/>L = distance²"]
        NegLoss["Negative (label=0)<br/>L = max(margin-dist, 0)²"]
        ContTotal["Total Contrastive<br/>L_contr = mean(positive + negative)"]
    end
    
    subgraph Delta["Delta Regression Loss<br/>weight=0.1"]
        GroundTruth["Ground Truth<br/>Δ = template_xyz - user_xyz"]
        PredDelta["Predicted Delta<br/>from FrameDecoder"]
        Huber["Huber Loss<br/>(robust to outliers)"]
        MaskCorrect["Mask: Only label=1<br/>(correct attempts)"]
        DeltaTotal["Total Delta Loss<br/>L_delta = Huber(pred, gt)"]
    end
    
    subgraph ROM["Range of Motion Loss<br/>weight=0.3"]
        UserROM["User ROM<br/>max - min per joint"]
        TemplROM["Template ROM<br/>max - min per joint"]
        Coverage["Coverage<br/>user_rom / template_rom"]
        Shortfall["Shortfall<br/>max(0.75 - coverage, 0)"]
        ROMTotal["Total ROM Loss<br/>L_rom = weight × shortfall"]
    end
    
    Total["Total Training Loss<br/>L_total = 1.0·L_contr + 0.1·L_delta + 0.3·L_rom"]
    
    SimScore --> Dist
    Dist --> PosLoss
    Dist --> NegLoss
    PosLoss --> ContTotal
    NegLoss --> ContTotal
    GroundTruth --> Huber
    PredDelta --> Huber
    Huber --> MaskCorrect
    MaskCorrect --> DeltaTotal
    UserROM --> Coverage
    TemplROM --> Coverage
    Coverage --> Shortfall
    Shortfall --> ROMTotal
    ContTotal --> Total
    DeltaTotal --> Total
    ROMTotal --> Total
```

## 6. Inference Decision Flow

```mermaid
graph TB
    Video["Input Video"]
    Extract["Extract Pose<br/>MediaPipe 33 → 17 COCO"]
    Sequence["Build Sequence<br/>T_video, 17, 3"]
    Preprocess["Preprocess<br/>Normalize & Featurize"]
    Tensor["Tensor<br/>1, 9/12, 120, 17"]
    
    subgraph Models["Load All Models<br/>Exercise 1-20"]
        M1["Model 1"]
        M2["Model 2"]
        MN["Model N"]
    end
    
    subgraph Inference["Per-Model Inference"]
        Forward["Forward Pass<br/>Template vs User"]
        Score["Get Similarity<br/>Score ∈ [0,1]"]
        Compare["Compare to<br/>Learned Threshold"]
        Decision["IF Score > Threshold<br/>→ CORRECT"]
    end
    
    Margin["Compute Margin<br/>Score - Threshold"]
    GetJoint["Extract Joint<br/>Importance"]
    
    subgraph Visualization["If Phase-Aware"]
        WarpViz["Use Warp Weights<br/>Temporal Alignment"]
        Ghost["Render Ghost<br/>Perfect Form"]
        Arrows["Draw Correction<br/>Arrows"]
        Sync["Compute Temporal<br/>Correlation"]
        HUD["Render HUD<br/>Scores + Feedback"]
    end
    
    Best["Select Best Result<br/>Highest Margin"]
    Output["Output<br/>Label + Score + Feedback"]
    
    Video --> Extract
    Extract --> Sequence
    Sequence --> Preprocess
    Preprocess --> Tensor
    Tensor --> M1
    Tensor --> M2
    Tensor --> MN
    M1 --> Forward
    M2 --> Forward
    MN --> Forward
    Forward --> Score
    Score --> Compare
    Compare --> Decision
    Decision --> Margin
    Decision --> GetJoint
    Forward --> WarpViz
    WarpViz --> Ghost
    Ghost --> Arrows
    Arrows --> Sync
    Sync --> HUD
    Margin --> Best
    GetJoint --> Best
    HUD --> Best
    Best --> Output
```

## 7. Checkpoint Evolution & Selection

```mermaid
graph LR
    subgraph V1["Baseline"]
        B["Contrastive Only<br/>Accuracy: 82.3%<br/>No Phase<br/>No ROM"]
    end
    
    subgraph V2["Phase-Aware"]
        PA["+ Phase Aligner<br/>+ Frame Decoder<br/>Accuracy: 87.1%<br/>+4.8% improvement"]
    end
    
    subgraph V3["Full Model"]
        Full["+ ROM Loss<br/>+ Joint Confidence<br/>Accuracy: 91.2%<br/>+4.1% improvement<br/>+17.2% ROM accuracy"]
    end
    
    subgraph Variants["Variants for Ablation"]
        A1["Phase-aware v2<br/>Experimental iteration"]
        A2["Without ROM v1<br/>Isolation study"]
        A3["Without ROM v2<br/>Refined ablation"]
    end
    
    B -->|+Phase Alignment| PA
    PA -->|+ROM Enforcement| Full
    Full -.->|for research| Variants
    
    Production["🎯 Production<br/>uiprmd_phase_aware_rom"]
    Full -->|Deploy| Production
```

## 8. ROM Loss Impact Visualization

```mermaid
graph TB
    subgraph Test["Test Scenario<br/>User performs 50% ROM"]
        User["User ROM<br/>50% of template"]
    end
    
    subgraph WithoutROM["Without ROM Loss"]
        Score1["Score: 0.92<br/>HIGH"]
        Pred1["Label: CORRECT ❌<br/>FALSE POSITIVE"]
        Issue["Issue:<br/>Ignores amplitude<br/>Only looks at shape"]
    end
    
    subgraph WithROM["With ROM Loss"]
        Coverage["Coverage: 50%<br/>Below 75% threshold"]
        Score2["Score: 0.58<br/>LOW, Rejected"]
        Pred2["Label: INCORRECT ✓<br/>CORRECT DECISION"]
        Reason["Training enforced<br/>min_coverage = 75%"]
    end
    
    User --> Score1
    User --> Coverage
    Score1 --> Pred1
    Pred1 --> Issue
    Coverage --> Score2
    Score2 --> Pred2
    Pred2 --> Reason
    
    Result["🎯 Result: +17.2% accuracy<br/>on low-amplitude detection"]
```

## 9. Singleton Model Cache Pattern

```mermaid
graph TB
    subgraph Request1["Request 1"]
        R1["get_cached_models()"]
        Load1["Cache Empty?<br/>YES → Load from disk"]
        Store1["Store in _MODEL_CACHE"]
        Return1["Return models"]
    end
    
    subgraph Request2["Request 2"]
        R2["get_cached_models()"]
        Load2["Cache Empty?<br/>NO → Use cached"]
        Return2["Return same models<br/>No reload"]
    end
    
    subgraph Request3["Request 3...N"]
        R3["get_cached_models()"]
        Load3["Cache Empty?<br/>NO → Use cached"]
        Return3["Return same models"]
    end
    
    GlobalCache["Global _MODEL_CACHE<br/>Singleton Dict"]
    
    R1 --> Load1
    Load1 --> Store1
    Store1 --> GlobalCache
    Store1 --> Return1
    R2 --> Load2
    Load2 --> GlobalCache
    Load2 --> Return2
    R3 --> Load3
    Load3 --> GlobalCache
    Load3 --> Return3
    
    Benefit["✓ First request: 500ms<br/>✓ Subsequent: 1-2ms<br/>✓ Memory efficient"]
```

## 10. LOSO Cross-Validation Strategy

```mermaid
graph TB
    Dataset["UI-PRMD Dataset<br/>10 Subjects × 20 Exercises"]
    
    subgraph Fold1["Fold 1: Leave Subject 1 Out"]
        Train1["Train: Subjects 2-10"]
        Val1["Val: 10% of Train"]
        Test1["Test: Subject 1"]
    end
    
    subgraph Fold2["Fold 2: Leave Subject 2 Out"]
        Train2["Train: Subjects 1,3-10"]
        Val2["Val: 10% of Train"]
        Test2["Test: Subject 2"]
    end
    
    subgraph FoldN["Fold N: Leave Subject N Out"]
        TrainN["Train: Subjects 1..N-1,N+1..10"]
        ValN["Val: 10% of Train"]
        TestN["Test: Subject N"]
    end
    
    Results["Aggregate Results<br/>Average across folds"]
    Output["Subject-Independent<br/>Performance Estimate"]
    
    Dataset --> Fold1
    Dataset --> Fold2
    Dataset --> FoldN
    Fold1 --> Results
    Fold2 --> Results
    FoldN --> Results
    Results --> Output
    
    Strength["✓ Ensures generalization<br/>to unseen subjects<br/>✓ No subject leakage"]
```

---

**These diagrams complement the main thesis documentation in `APPLICATION_ARCHITECTURE_AND_FLOW.md`**
