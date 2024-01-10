from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_cv_splits, train_models_on_cv_folds, evaluate_model_performance


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_cv_splits,
                inputs=["model_input_table_research_paper", "params:model_options_stunted"],
                outputs="cv_data_splits_stunted",
                name="generate_cv_splits_stunted_node",
            ),
            node(
                func=train_models_on_cv_folds,
                inputs=["cv_data_splits_stunted", "params:model_options_stunted"],
                outputs="independent_rf_models_stunted",
                name="training_indpendent_rf_models_stunted_node",
            ),
            node(
                func=evaluate_model_performance,
                inputs=["cv_data_splits_stunted", "independent_rf_models_stunted"],
                outputs="metrics_stunted",
                name="evaluate_model_performance_stunted_node",
            ),
            node(
                func=generate_cv_splits,
                inputs=["model_input_table_research_paper", "params:model_options_wasted"],
                outputs="cv_data_splits_wasted",
                name="generate_cv_splits_wasted_node",
            ),
            node(
                func=train_models_on_cv_folds,
                inputs=["cv_data_splits_wasted", "params:model_options_wasted"],
                outputs="independent_rf_models_wasted",
                name="training_indpendent_rf_models_wasted_node",
            ),
            node(
                func=evaluate_model_performance,
                inputs=["cv_data_splits_wasted", "independent_rf_models_wasted"],
                outputs="metrics_wasted",
                name="evaluate_model_performance_wasted_node",
            ),
            node(
                func=generate_cv_splits,
                inputs=["model_input_table_research_paper", "params:model_options_healthy"],
                outputs="cv_data_splits_healthy",
                name="generate_cv_splits_healthy_node",
            ),
            node(
                func=train_models_on_cv_folds,
                inputs=["cv_data_splits_healthy", "params:model_options_healthy"],
                outputs="independent_rf_models_healthy",
                name="training_indpendent_rf_models_healthy_node",
            ),
            node(
                func=evaluate_model_performance,
                inputs=["cv_data_splits_healthy", "independent_rf_models_healthy"],
                outputs="metrics_healthy",
                name="evaluate_model_performance_healthy_node",
            ),
            node(
                func=generate_cv_splits,
                inputs=["model_input_table_research_paper", "params:model_options_poorest"],
                outputs="cv_data_splits_poorest",
                name="generate_cv_splits_poorest_node",
            ),
            node(
                func=train_models_on_cv_folds,
                inputs=["cv_data_splits_poorest", "params:model_options_poorest"],
                outputs="independent_rf_models_poorest",
                name="training_indpendent_rf_models_poorest_node",
            ),
            node(
                func=evaluate_model_performance,
                inputs=["cv_data_splits_poorest", "independent_rf_models_poorest"],
                outputs="metrics_poorest",
                name="evaluate_model_performance_poorest_node",
            ),
            node(
                func=generate_cv_splits,
                inputs=["model_input_table_research_paper", "params:model_options_underweight_bmi"],
                outputs="cv_data_splits_underweight_bmi",
                name="generate_cv_splits_underweight_bmi_node",
            ),
            node(
                func=train_models_on_cv_folds,
                inputs=["cv_data_splits_underweight_bmi", "params:model_options_underweight_bmi"],
                outputs="independent_rf_models_underweight_bmi",
                name="training_indpendent_rf_models_underweight_bmi_node",
            ),
            node(
                func=evaluate_model_performance,
                inputs=["cv_data_splits_underweight_bmi", "independent_rf_models_underweight_bmi"],
                outputs="metrics_underweight_bmi",
                name="evaluate_model_performance_underweight_bmi_node",
            ),
        ]
    )
