package org.apache.spark

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.storage.Zero.DoubleZero
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  "Assembler" should "vectorize input arguments" in {
    val X = new DenseMatrix(3, 2, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val y = DenseVector(1.0, 2.0, 3.0)
    val data = DenseMatrix.horzcat(X, y.asDenseMatrix.t)
    val rows = data.t.toArray.grouped(data.cols).toSeq.map(row => Row.fromSeq(row))
    val schema = StructType(Seq(
      StructField("x1", DoubleType),
      StructField("x2", DoubleType),
      StructField("y", DoubleType)
    ))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(rows), schema)
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2"))
      .setOutputCol("features")


    val result = vectorAssembler.transform(df)


    result.schema("features").dataType.typeName should be("vector")
 }

  "Assembler" should "vectorize input arguments correctly" in {
    val X = new DenseMatrix(3, 2, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val y = DenseVector(1.0, 2.0, 3.0)
    val data = DenseMatrix.horzcat(X, y.asDenseMatrix.t)
    val rows = data.t.toArray.grouped(data.cols).toSeq.map(row => Row.fromSeq(row))
    val schema = StructType(Seq(
      StructField("x1", DoubleType),
      StructField("x2", DoubleType),
      StructField("y", DoubleType)
    ))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(rows), schema)
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2"))
      .setOutputCol("features")


    val result = vectorAssembler.transform(df)


    val row = result.collect()(1)
    val value = row.getAs[Vector]("features").toArray
    value.length should be(2)
    value.apply(0) should be(2.0)
    value.apply(1) should be(5.0)
 }

  "Estimator" should "return model with correct args in delta" in {
    val X = DenseMatrix.rand(1000, 2) * 10.0
    val y = X * DenseVector(1.0, 2.0)
    val data = DenseMatrix.horzcat(X, y.asDenseMatrix.t)
    val rows = data.t.toArray.grouped(data.cols).toSeq.map(row => Row.fromSeq(row))
    val schema = StructType(Seq(
      StructField("x1", DoubleType),
      StructField("x2", DoubleType),
      StructField("y", DoubleType)
    ))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(rows), schema)
    val pipeline = new Pipeline().setStages(
      Array(
        new VectorAssembler()
          .setInputCols(Array("x1", "x2"))
          .setOutputCol("features"),
        new LinearRegression()
          .setLabelCol("y")
          .setFeaturesCol("features")
          .setLearningRate(1e-5)
          .setNumIterations(100)
      )
    )
    val delta = 0.001
    spark.sparkContext.setLogLevel("error")


    val lrModel = pipeline.fit(df)
    val lr = lrModel.stages.last.asInstanceOf[LinearRegressionModel]
    val coefficients = lr.getWeights.toArray



    coefficients should have length 2
    coefficients(0) should be(1.0 +- delta)
    coefficients(1) should be(2.0 +- delta)
 }
}
