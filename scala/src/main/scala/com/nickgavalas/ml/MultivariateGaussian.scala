package com.nickgavalas.ml

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration

import scala.collection.mutable.ArrayBuffer

class MultivariateGaussian(numFeats: Int) {
  private val sums: ArrayBuffer[Double] = ArrayBuffer.fill(numFeats)(0)
  private val sumsSquared: ArrayBuffer[Double] = ArrayBuffer.fill(numFeats)(0)
  private var count: Int = 0

  private def updateModel(row: Vector[Double]): Unit = {
    count += 1
    for (i <- sums.indices)
      sums(i) += row(i)
    for (i <- sumsSquared.indices)
      sumsSquared(i) += math.pow(row(i), 2.0)
  }

  def gaussPDF(value: Double, featureIndex: Int): Double = {
    val mu = sums(featureIndex) / count
    val sigma2 = (sumsSquared(featureIndex) / count) - math.pow(mu, 2.0)
    math.exp(-math.pow(value - mu, 2.0) / (2.0 * sigma2)) / math.sqrt(2.0 * math.Pi * sigma2)
  }

  // apply the model to a single instance
  def score(row: Vector[Double]): Double = {
    updateModel(row)

    var score = 1.0
    for (i <- row.indices)
      score *= gaussPDF(row(i), i)
    score
  }
}

class MultivariateGaussianMapper(numFeats: Int) extends RichMapFunction[Vector[Double], Double] {
  var model: MultivariateGaussian = _

  override def open(parameters: Configuration): Unit = {
    model = new MultivariateGaussian(numFeats)
  }

  override def map(value: Vector[Double]): Double = {
    model.score(value)
  }
}
