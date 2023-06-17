package org.apache.spark

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute, UnresolvedAttribute}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCol}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.sql.functions.{col, struct, udf}
import org.apache.spark.sql.types.{DoubleType, NumericType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.immutable.Seq
import scala.collection.mutable


trait TransformerParams extends Params with HasInputCols with HasOutputCol {

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val incorrectColumns = $(inputCols).flatMap { name =>
      schema(name).dataType match {
        case _: NumericType => None
        case t if t.isInstanceOf[VectorUDT] => None
        case other => Some(s"Data type ${other.catalogString} of column $name is not supported.")
      }
    }
    if (incorrectColumns.nonEmpty) {
      throw new IllegalArgumentException(incorrectColumns.mkString("\n"))
    }
    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      StructType(schema.fields :+ StructField(getOutputCol, new VectorUDT, nullable = true))
    }
  }
}


class VectorAssembler(override val uid: String) extends Transformer with TransformerParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("vectorAssembler"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val schema = dataset.schema

    val featureAttributesMap = $(inputCols).map { c =>
      val field = schema(c)
      field.dataType match {
        case DoubleType =>
          val attribute = Attribute.fromStructField(field)
          attribute match {
            case UnresolvedAttribute =>
              Seq(NumericAttribute.defaultAttr.withName(c))
            case _ =>
              Seq(attribute.withName(c))
          }
        case _: NumericType =>
          Seq(NumericAttribute.defaultAttr.withName(c))
        case otherType =>
          throw new SparkException(s"VectorAssembler does not support the $otherType type")
      }
    }
    val featureAttributes = featureAttributesMap.flatten[Attribute]
    val metadata = new AttributeGroup($(outputCol), featureAttributes).toMetadata()

    val assembleFunc = udf { r: Row =>
      assemble(r.toSeq: _*)
    }.asNondeterministic()
    val args = $(inputCols).map { c =>
      schema(c).dataType match {
        case DoubleType => dataset(c)
        case _: VectorUDT => dataset(c)
        case _: NumericType => dataset(c).cast(DoubleType).as(s"${c}_double_$uid")
      }
    }

    dataset.select(col("*"), assembleFunc(struct(args: _*)).as($(outputCol), metadata))
  }

  override def copy(extra: ParamMap): Transformer = copyValues(new VectorAssembler(uid), extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  private def assemble(vv: Any*): Vector = {
    val indices = mutable.ArrayBuilder.make[Int]
    val values = mutable.ArrayBuilder.make[Double]
    var featureIndex = 0

    var inputColumnIndex = 0
    vv.foreach {
      case v: Double =>
        if (v.isNaN) {
          throw new SparkException(
            s"""Encountered NaN while assembling a row with handleInvalid = "error". Consider
               |removing NaNs from dataset or using handleInvalid = "keep" or "skip"."""
              .stripMargin)
        } else if (v != 0.0) {
          indices += featureIndex
          values += v
        }
        inputColumnIndex += 1
        featureIndex += 1
      case vec: Vector =>
        vec.foreachNonZero { case (i, v) =>
          indices += featureIndex + i
          values += v
        }
        inputColumnIndex += 1
        featureIndex += vec.size
      case null =>
        throw new SparkException(
          s"""Encountered null while assembling a row with handleInvalid = "error". Consider
             |removing nulls from dataset or using handleInvalid = "keep" or "skip"."""
            .stripMargin)
      case o =>
        throw new SparkException(s"$o of type ${o.getClass.getName} is not supported.")
    }
    Vectors.sparse(featureIndex, indices.result(), values.result()).compressed
  }
}
