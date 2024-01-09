from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model, generate_cv_splits


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=split_data,
            #     inputs=["model_input_table", "params:model_options"],
            #     outputs=["X_train", "X_test", "y_train", "y_test"],
            #     name="split_data_node",
            # ),
            # node(
            #     func=train_model,
            #     inputs=["X_train", "y_train"],
            #     outputs="regressor",
            #     name="train_model_node",
            # ),
            # node(
            #     func=evaluate_model,
            #     inputs=["regressor", "X_test", "y_test"],
            #     name="evaluate_model_node",
            #     outputs="metrics",
            # ),
            node(
                func=generate_cv_splits,
                inputs=["model_input_table_research_paper", "params:model_options_stunted"],
                outputs="cv_data_splits_stunted",
                name="generate_cv_splits_node",
            ),
        ]
    )
