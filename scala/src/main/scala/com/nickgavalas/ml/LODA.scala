package com.nickgavalas.ml

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

// TODO: write tests

class LODA(numDimensions: Int, numVectors: Int = 100, bucketWidth: Double, seed: Long = 42) {
  private var count = 0
  private val rng = Random

  private var histograms = ArrayBuffer[mutable.Map[Int, Int]]()
  private var projectionVectors = ArrayBuffer[ArrayBuffer[Double]]()

  rng.setSeed(seed)

  // generate projection vectors
  for (_ <- 1 to numVectors) {
    val vec: ArrayBuffer[Double] = ArrayBuffer.fill(numDimensions)(0)
    // indices have to be chosen without replacement, so we use a set to keep track of the used ones.
    var usedIndices = Set[Int]()
    val range = math.floor(math.sqrt(numDimensions)).toInt
    for (_ <- 1 to range) {
      var index = rng.nextInt(range)
      while (usedIndices.contains(index)) {
        index = rng.nextInt(range)
      }
      vec(index) = rng.nextGaussian()
      usedIndices += index
    }
    projectionVectors += vec
  }

  // create equi-width histograms
  histograms ++= ArrayBuffer.fill(numDimensions)(mutable.Map[Int, Int]())

  def score(instance: Vector[Double]): Double = {
    var score: Double = 0
    for ((vector, histogram) <- projectionVectors zip histograms) {
      val dot: Double = (instance, vector).zipped.map((x, y) => x * y).sum
      val bucketIndex: Int = math.floor(dot / bucketWidth).toInt
      if (!histogram.contains(bucketIndex)) {
        histogram += (bucketIndex -> 0)
      } else {
        score += math.log(histogram(bucketIndex).toDouble)
      }
      histogram(bucketIndex) += 1
    }
    score * (-1.0 / numVectors)
  }
}

class LODAMapper(numDimensions: Int, numVectors: Int = 100, bucketWidth: Double, seed: Long = 42)
  extends RichMapFunction[Vector[Double], Double] {
  var model: LODA = _

  override def open(parameters: Configuration): Unit = {
    model = new LODA(numDimensions, numVectors, bucketWidth, seed)
  }

  override def map(value: Vector[Double]): Double = {
    model.score(value)
  }
}
