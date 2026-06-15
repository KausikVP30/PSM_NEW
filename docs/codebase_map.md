# Codebase Map

This snapshot covers the active main project in the workspace and excludes Kaggle mirrors, generated outputs, and the nested `repo/` copy.

## Structure

```text
PSM_NEW/
├── README.md
├── config.yaml
├── config_gpu_run.yaml
├── config_triviaqa_full.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-py311.txt
├── run_experiment.py
├── run_experiment_gpu.py
├── run_triviaqa_full.py
├── test_generator.py
├── test_ollama_generator.py
├── data/
│   ├── triviaqa_full.jsonl
│   ├── triviaqa_sample.jsonl
├── scripts/
│   ├── build_codebase_map.py
│   ├── monitor_experiment.py
│   ├── ollama_shared_gpu.py
│   ├── prepare_triviaqa_full.py
│   ├── run_with_ollama.py
│   ├── watch_progress.sh
├── src/
│   ├── __init__.py
│   ├── config/
│   │   └── __init__.py
│   │   └── settings.py
│   ├── data/
│   │   └── __init__.py
│   │   └── dataset.py
│   │   └── schemas.py
│   ├── evaluation/
│   │   └── __init__.py
│   │   └── metrics.py
│   ├── generation/
│   │   └── __init__.py
│   │   └── generator.py
│   │   └── generator_ollama.py
│   │   └── generator_t5.py
│   ├── logging_utils/
│   │   └── __init__.py
│   │   └── logger.py
│   ├── memory/
│   │   └── __init__.py
│   │   └── embedding_memory_store.py
│   │   └── embedding_utils.py
│   │   └── memory_store.py
│   ├── pipeline/
│   │   └── __init__.py
│   │   └── rag_pipeline.py
│   ├── prompt/
│   │   └── __init__.py
│   │   └── assembler.py
│   ├── retrieval/
│   │   └── __init__.py
│   │   └── bm25_retriever.py
│   │   └── dense_hnsw.py
│   │   └── hybrid.py
│   │   └── reranker.py
│   │   └── utils.py
│   ├── router/
│   │   └── __init__.py
│   │   └── confidence_gate.py
│   ├── utils/
│   │   └── __init__.py
│   │   └── io.py
│   │   └── paths.py
│   │   └── runtime.py
│   │   └── seed.py
├── tests/
│   ├── test_smoke.py
└── (excluded: kaggle_*, repo/, outputs/, logs/, .venv/)
```

## Core Packages

- src.config: YAML-backed settings loader.
- src.data: dataset loading and schema objects.
- src.evaluation: answer scoring and metrics.
- src.generation: generator backends for T5 and Ollama.
- src.logging_utils: structured logging helpers.
- src.memory: semantic memory store and persistence helpers.
- src.pipeline: end-to-end RAG orchestration.
- src.prompt: prompt assembly.
- src.retrieval: BM25, dense HNSW, hybrid retrieval, reranking.
- src.router: confidence gate for memory-vs-retrieval routing.
- src.utils: IO, runtime, path, and seeding helpers.

## Entry Points

- run_experiment.py: primary main-project runner.
- run_experiment_gpu.py: GPU-oriented variant.
- run_triviaqa_full.py: full TriviaQA runner.
- scripts/*.py: local utilities for monitoring, Ollama, and dataset prep.

## Import Graph

```mermaid
flowchart LR
  run_experiment["run_experiment"]
  run_triviaqa_full["run_triviaqa_full"]
  scripts_ollama_shared_gpu["scripts.ollama_shared_gpu"]
  scripts_run_with_ollama["scripts.run_with_ollama"]
  src_config["src.config"]
  src_config_settings["src.config.settings"]
  src_data["src.data"]
  src_data_dataset["src.data.dataset"]
  src_data_schemas["src.data.schemas"]
  src_evaluation["src.evaluation"]
  src_evaluation_metrics["src.evaluation.metrics"]
  src_generation["src.generation"]
  src_generation_generator["src.generation.generator"]
  src_generation_generator_ollama["src.generation.generator_ollama"]
  src_generation_generator_t5["src.generation.generator_t5"]
  src_logging_utils["src.logging_utils"]
  src_logging_utils_logger["src.logging_utils.logger"]
  src_memory["src.memory"]
  src_memory_embedding_memory_store["src.memory.embedding_memory_store"]
  src_memory_embedding_utils["src.memory.embedding_utils"]
  src_memory_memory_store["src.memory.memory_store"]
  src_pipeline["src.pipeline"]
  src_pipeline_rag_pipeline["src.pipeline.rag_pipeline"]
  src_prompt["src.prompt"]
  src_prompt_assembler["src.prompt.assembler"]
  src_retrieval["src.retrieval"]
  src_retrieval_bm25_retriever["src.retrieval.bm25_retriever"]
  src_retrieval_dense_hnsw["src.retrieval.dense_hnsw"]
  src_retrieval_hybrid["src.retrieval.hybrid"]
  src_retrieval_reranker["src.retrieval.reranker"]
  src_retrieval_utils["src.retrieval.utils"]
  src_router["src.router"]
  src_router_confidence_gate["src.router.confidence_gate"]
  src_utils_io["src.utils.io"]
  src_utils_paths["src.utils.paths"]
  src_utils_runtime["src.utils.runtime"]
  src_utils_seed["src.utils.seed"]
  test_generator["test_generator"]
  test_ollama_generator["test_ollama_generator"]
  tests_test_smoke["tests.test_smoke"]

  run_experiment --> src_config
  run_experiment --> src_data
  run_experiment --> src_generation
  run_experiment --> src_logging_utils
  run_experiment --> src_memory_embedding_memory_store
  run_experiment --> src_pipeline
  run_experiment --> src_prompt
  run_experiment --> src_retrieval
  run_experiment --> src_router
  run_experiment --> src_utils_io
  run_experiment --> src_utils_paths
  run_experiment --> src_utils_runtime
  run_experiment --> src_utils_seed
  run_triviaqa_full --> src_data_dataset
  run_triviaqa_full --> src_pipeline_rag_pipeline
  scripts_run_with_ollama --> scripts_ollama_shared_gpu
  src_config --> src_config_settings
  src_data --> src_data_dataset
  src_data --> src_data_schemas
  src_data_dataset --> src_data_schemas
  src_data_dataset --> src_utils_runtime
  src_evaluation --> src_evaluation_metrics
  src_evaluation_metrics --> src_data_schemas
  src_evaluation_metrics --> src_retrieval_utils
  src_generation --> src_generation_generator
  src_generation --> src_generation_generator_ollama
  src_generation_generator --> src_data_schemas
  src_generation_generator_ollama --> src_data_schemas
  src_generation_generator_t5 --> src_data_schemas
  src_logging_utils --> src_logging_utils_logger
  src_memory --> src_memory_embedding_memory_store
  src_memory --> src_memory_memory_store
  src_memory_embedding_memory_store --> src_memory_embedding_utils
  src_memory_embedding_memory_store --> src_retrieval_utils
  src_memory_memory_store --> src_retrieval_utils
  src_pipeline --> src_pipeline_rag_pipeline
  src_pipeline_rag_pipeline --> src_data_schemas
  src_pipeline_rag_pipeline --> src_evaluation_metrics
  src_pipeline_rag_pipeline --> src_generation_generator
  src_pipeline_rag_pipeline --> src_memory_embedding_memory_store
  src_pipeline_rag_pipeline --> src_prompt_assembler
  src_pipeline_rag_pipeline --> src_retrieval_hybrid
  src_pipeline_rag_pipeline --> src_retrieval_reranker
  src_pipeline_rag_pipeline --> src_router_confidence_gate
  src_pipeline_rag_pipeline --> src_utils_io
  src_prompt --> src_prompt_assembler
  src_prompt_assembler --> src_data_schemas
  src_retrieval --> src_retrieval_bm25_retriever
  src_retrieval --> src_retrieval_dense_hnsw
  src_retrieval --> src_retrieval_hybrid
  src_retrieval --> src_retrieval_reranker
  src_retrieval_bm25_retriever --> src_data_schemas
  src_retrieval_bm25_retriever --> src_retrieval_utils
  src_retrieval_dense_hnsw --> src_data_schemas
  src_retrieval_hybrid --> src_data_schemas
  src_retrieval_hybrid --> src_retrieval_bm25_retriever
  src_retrieval_hybrid --> src_retrieval_dense_hnsw
  src_retrieval_reranker --> src_data_schemas
  src_retrieval_reranker --> src_retrieval_utils
  src_router --> src_router_confidence_gate
  src_utils_io --> src_data_schemas
  src_utils_io --> src_utils_paths
  test_generator --> src_data_schemas
  test_generator --> src_generation_generator
  test_ollama_generator --> src_data_schemas
  test_ollama_generator --> src_generation_generator_ollama
  tests_test_smoke --> src_router_confidence_gate
```

## Refresh

Run `source .venv/bin/activate && python scripts/build_codebase_map.py` from the project root to regenerate this file.
