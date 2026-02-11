# Refatoração da Base de Dados — Log incremental

## Etapa 1 — Contrato e inventário central (implementada)

### O que mudou

- Criado `utils/ifdata_cache/metric_registry.py` com:
  - registro central de métricas (`METRIC_REGISTRY`)
  - contratos de datasets (`DATASET_CONTRACTS`)
  - funções utilitárias de consulta e validação de contrato.
- Exportações públicas do novo módulo adicionadas em `utils/ifdata_cache/__init__.py`.
- Criada documentação técnica `docs/data_pipeline.md` com visão de pipeline e guia de manutenção.

### Como testar

1. Rodar testes de contrato/registry.
2. Rodar testes de métricas derivadas já existentes para garantir não regressão.

### Risco

- Baixo: mudança aditiva (sem alterar fórmulas ativas nas abas).


## Ajuste pós-review

- Removida dependência de `pandas` em `metric_registry.py` (validação agora aceita interface estrutural `empty/columns`).
- Testes de `tests/test_metric_registry.py` reescritos para usar `FakeDataFrame`, permitindo execução em ambiente mínimo.


## Integração inicial do registry ao cálculo derivado

- `derived_metrics.py` passou a consumir labels/formatação/fórmulas de métricas derivadas a partir de `metric_registry.py`, removendo duplicação de catálogo.
- Mantido fallback local para preservar retrocompatibilidade se o registry estiver incompleto.
- Testes do registry ampliados para validar consistência dos helpers de métricas derivadas.


## Otimização urgente peers (performance)

- Substituído lookup O(n) por célula (`df[(inst)&(periodo)]`) por índice em memória `(Instituição, Período) -> row` com cache em `df.attrs`.
- Adicionado recorte prévio dos caches extras por bancos/períodos selecionados antes de montar a tabela peers.
- Objetivo: reduzir drasticamente tempo da aba com múltiplos bancos (ex.: 15 bancos).


## Leitura filtrada no cache (Parquet)

- Adicionada função `_carregar_cache_relatorio_slice` para carregar recortes por período/instituição direto do arquivo de cache (com `pyarrow.dataset` quando disponível).
- A aba **Peers (Tabela)** passou a carregar `ativo/passivo/carteiras/dre/capital` já filtrados, reduzindo I/O e memória antes do cálculo.
- Fallback preservado: se filtro parquet falhar, carrega e filtra em pandas.


## Cache agressivo do pré-processamento de Rankings

- Extraído pré-processamento pesado da aba Rankings para `_get_rankings_base_df` com `@st.cache_data` (normalização de lucro, CET1 e merge complementar).
- Cache invalidado por token de versão dos caches `principal/capital`, flag de mescla de capital e assinatura de aliases.
- Objetivo: remover recomputação completa a cada clique no multiselect de bancos.


## Hotfix NameError na aba Peers

- Reintroduzido `_slice_cache_for_peers` como shim de compatibilidade para evitar quebra em deployments com código parcial (erro `NameError`).
- Mantida lógica de recorte por `Instituição`/`Período` sem impacto na semântica dos indicadores.
