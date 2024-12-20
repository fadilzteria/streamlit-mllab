# Binary Classification with Advanced Ensemble Tree Models
from streamlit.testing.v1 import AppTest

# Model Training
def test_training():
    """A user train the models with the cleaned dataset"""
    at = AppTest.from_file("codes/modeling/training.py", default_timeout=150).run()

    assert "Model Training" == at.title[0].value

    # Experiment
    at.text_input("exp_name").set_value("Classif Tree Baseline").run()
    at.selectbox("dp_name").set_value("Classif Baseline").run()
    at.selectbox("ml_task").set_value("Classification").run()
    at.number_input("folds").set_value(5).run()

    # Feature Engineering
    at.toggle("fe_drop_many_cat_unique").set_value(False).run()
    at.toggle("fe_cat_encoding").set_value(True).run()
    at.selectbox("fe_cat_type").set_value("Categorical").run()
    at.toggle("fe_scaling").set_value(False).run()

    # Model
    model_names = ["XGBoost", "LightGBM", "CatBoost"]
    for model_name in model_names:
        at.multiselect("model_names").select(model_name).run()

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Avg Precision"]
    for metric in metrics:
        at.multiselect("metric_names").select(metric).run()

    at.selectbox("best_metric").set_value("ROC AUC").run()

    # Submit
    at.button[0].click().run()

# Model Prediction
def test_prediction():
    """A user predict targets from the test dataset using trained models"""
    at = AppTest.from_file("codes/home/prediction.py", default_timeout=60).run()

    assert "Prediction" == at.title[0].value

    csv_files = ["classif_test_labels.csv", "classif_test_no_labels.csv"]
    test_names = ["Test with Labels", "Test No Labels"]
    for (csv_file, test_name) in zip(csv_files, test_names):
        # Dataset Input and Preprocessing
        at.selectbox("data_file").set_value(csv_file).run()
        at.selectbox("dp_name").set_value("Classif Baseline").run()
        at.button[0].click().run()

        # Testing Configuration
        at.text_input("test_name").set_value(test_name).run()
        at.selectbox("exp_name").set_value("Classif Tree Baseline").run()
        at.button[1].click().run()

        # Local Explainability
        at.button[2].click().run()
        for i in range(5):
            at.number_input("idx").set_value(i).run()
