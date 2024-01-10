from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_raw_data,
                inputs=["data", "params:variables"],
                outputs="model_input_table_research_paper",
                name="create_model_input_table_node_research_paper",
            ),
        ]
    )
