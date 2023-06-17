package org.apache.spark

import breeze.linalg.DenseVector
import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}


trait EstimatorParams extends Params with HasFeaturesCol with HasLabelCol {

  var numIterations: Int = 100
  var learningRate: Double = 0.1

  def setNumIterations(value: Integer): this.type = {
    numIterations = value
    this
  }

  def setLearningRate(value: Double): this.type = {
    learningRate = value
    this
  }

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT())
    schema
  }
}


class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with EstimatorParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    val data = dataset.select(col($(featuresCol)), col($(labelCol))).rdd.map(row => (DenseVector(row.getAs[Vector](0).toArray), row.getDouble(1)))
    val weights = DenseVector.rand[Double](data.first()._1.length)

    for (_ <- 1 to numIterations) {
      val gradient = data.map {
        case (features, label) =>
          val prediction = features.dot(weights)
          val error = prediction - label
          features * error
      }.reduce(_ + _)
      weights -= gradient * learningRate
    }

    new LinearRegressionModel(weights)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = copyValues(new LinearRegression(uid), extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}


class LinearRegressionModel(override val uid: String,
                            val weights: DenseVector[Double]
                           ) extends Model[LinearRegressionModel] with EstimatorParams {
  def this(weights: DenseVector[Double]) = this(Identifiable.randomUID("linearRegressionModel"), weights)

  def getWeights: DenseVector[Double] = weights

  def transform(dataset: Dataset[_]): DataFrame = {
    val predictUDF = udf((features: Vector) => {
      val prediction = new DenseVector(features.toArray) dot weights
      prediction
    })

    dataset.withColumn("prediction", predictUDF(col($(featuresCol))))
  }

  def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  def copy(extra: org.apache.spark.ml.param.ParamMap): LinearRegressionModel =
    copyValues(new LinearRegressionModel(uid, weights.copy), extra)
}
