"""
This is a boilerplate pipeline
generated using Kedro 0.18.1
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from pyspark.sql import DataFrame, types
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from flight_delay.utils import mathematics


def select_cols(df: DataFrame, params: Dict):
    cols = params["columns_of_interest"]
    return df.select(cols).na.drop()


def feature_engineering(df: DataFrame, params: Dict):
    target = params["target_column"]
    sample_fraction = params["sample_fraction"]
    df2 = df.withColumn("DepHour", (df["DepTime"] / 100).cast("int")).withColumn(
        target, (df["DepDelay"] > 15).cast("int")
    )
    return df2.drop("DepDelay").sample(sample_fraction)


def split_data(data: DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    # Split to training and testing data

    comp = mathematics.complementary(parameters["train_fraction"])

    data_train, data_test = data.randomSplit(
        weights=[parameters["train_fraction"], comp]
         #weights=[parameters["train_fraction"], 1 - parameters["train_fraction"]]

    )
    # X_train = data_train.drop(parameters["target_column"])
    # X_test = data_test.drop(parameters["target_column"])
    # y_train = data_train.select(parameters["target_column"])
    # y_test = data_test.select(parameters["target_column"])

    return data_train, data_test


def make_pyspark_pipeline(data_train: DataFrame, params: Dict):
    string_fields = params["string_fields"]
    categorical_fields = params["categorical_fields"]
    continuous_fields = params["continuous_fields"]

    output_cols = [f"{column_name}Index" for column_name in string_fields]
    indexers = StringIndexer(
        inputCols=string_fields, outputCols=output_cols, handleInvalid="keep"
    )

    input_to_ohe = output_cols + [
        col for col in categorical_fields if col not in string_fields
    ]
    output_from_ohe = [f"{col}OneHot" for col in input_to_ohe]
    encoders = OneHotEncoder(
        inputCols=input_to_ohe, outputCols=output_from_ohe, handleInvalid="keep"
    )

    assembler = VectorAssembler(
        inputCols=output_from_ohe + continuous_fields, outputCol="features"
    )
    forest = RandomForestClassifier(
        featuresCol="features",
        labelCol="Delayed",
        numTrees=params.get("numTrees"),
        seed=params.get("random_state"),
    )

    pipe = Pipeline(stages=[indexers, encoders, assembler, forest])

    pipeline = pipe.fit(data_train)

    return pipeline


def predict(data_test: DataFrame, pipe: PipelineModel, params):
    target = params["target_column"]

    return pipe.transform(data_test).select([target, "probability", "prediction","rawPrediction"])


def report_evaluator(data_test: DataFrame, params: Dict):
    target = params["target_column"]
    evaluator = BinaryClassificationEvaluator(labelCol=target)
    areaUnderROC = evaluator.evaluate(data_test)
    logger = logging.getLogger(__name__)
    logger.info("Model has an areaUnderROC of %.3f on test data.", areaUnderROC)
