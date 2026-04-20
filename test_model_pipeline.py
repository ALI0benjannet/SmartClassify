from pathlib import Path

from model_pipeline import (
    evaluate_model,
    prepare_data,
    save_model,
    train_model,
    load_model,
)


def test_pipeline_train_save_load_and_evaluate(tmp_path):
    data_path = Path(__file__).resolve().parent / "archive (1)" / "Obesity_Dataset.arff"
    model_path = tmp_path / "obesity_model.joblib"

    X_train, X_test, y_train, y_test = prepare_data(
        data_path, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 4

    saved_path = save_model(model, model_path)
    assert saved_path.exists()

    loaded_model = load_model(saved_path)
    loaded_metrics = evaluate_model(loaded_model, X_test, y_test)
    assert loaded_metrics["accuracy"] == metrics["accuracy"]
