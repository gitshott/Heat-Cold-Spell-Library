from heat_cold_spell import (
    analyze_history,
    compare_models,
    predict_future,
    predict_spells,
    prepare_data,
    train_models,
)


def test_public_api_is_available():
    assert callable(prepare_data)
    assert callable(analyze_history)
    assert callable(compare_models)
    assert callable(train_models)
    assert callable(predict_future)
    assert callable(predict_spells)