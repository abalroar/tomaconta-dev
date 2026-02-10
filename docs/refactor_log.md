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
