import pandas as pd

from utils.ifdata_cache.metric_registry import (
    METRIC_REGISTRY,
    DATASET_CONTRACTS,
    get_metric_definition,
    validate_dataframe_by_contract_name,
)


def test_registry_has_core_metrics():
    assert "indice_basileia" in METRIC_REGISTRY
    assert "roe_ac_ytd_an" in METRIC_REGISTRY
    assert "desp_captacao_captacao" in METRIC_REGISTRY


def test_metric_definition_scale_is_decimal_for_pct_metrics():
    metric = get_metric_definition("desp_captacao_captacao")
    assert metric is not None
    assert metric.unit == "%"
    assert metric.internal_scale == "decimal_0_1"


def test_dataset_contract_validation_ok():
    contract = DATASET_CONTRACTS["derived_metrics_long"]
    df = pd.DataFrame(
        {
            "Instituição": ["Banco A"],
            "Período": ["1/2025"],
            "Métrica": ["Desp Captação / Captação"],
            "Valor": [0.2],
            "Unidade": ["pct"],
        }
    )
    errors = validate_dataframe_by_contract_name(df, contract.name)
    assert errors == []


def test_dataset_contract_validation_missing_column():
    df = pd.DataFrame(
        {
            "Instituição": ["Banco A"],
            "Período": ["1/2025"],
            "Valor": [0.2],
        }
    )
    errors = validate_dataframe_by_contract_name(df, "derived_metrics_long")
    assert any("coluna obrigatória ausente" in e for e in errors)
