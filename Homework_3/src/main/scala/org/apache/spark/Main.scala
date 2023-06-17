package org.apache.spark

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Gaussian
import breeze.storage.Zero.DoubleZero
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}


object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("LinearRegression")
      .getOrCreate()

    val num = 100000
    val features_num = 3
    val X = DenseMatrix.rand(num, features_num) * 10.0
    val noise = DenseVector(Gaussian(0, 1).sample(num).toArray)
    val y = X * DenseVector(1.5, 0.3, -0.7) + noise
    val data = DenseMatrix.horzcat(X, y.asDenseMatrix.t)
    val rows = data.t.toArray.grouped(data.cols).toSeq.map(row => Row.fromSeq(row))

    val schema = StructType(Seq(
      StructField("x1", DoubleType),
      StructField("x2", DoubleType),
      StructField("x3", DoubleType),
      StructField("y", DoubleType)
    ))

    val df = spark.createDataFrame(spark.sparkContext.parallelize(rows), schema)

    val pipeline = new Pipeline().setStages(
      Array(
        new VectorAssembler()
          .setInputCols(Array("x1", "x2", "x3"))
          .setOutputCol("features"),
        new LinearRegression()
          .setLabelCol("y")
          .setFeaturesCol("features")
          .setLearningRate(1e-7)
          .setNumIterations(100)
      )
    )

    val lrModel = pipeline.fit(df)
    val pred = lrModel.transform(df)
    pred.show(10)

    val lr = lrModel.stages.last.asInstanceOf[LinearRegressionModel]
    println(s"Coefficients: ${lr.getWeights}")
  }
}
