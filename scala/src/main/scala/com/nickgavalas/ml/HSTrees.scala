package com.nickgavalas.ml

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

// TODO: write tests.

class HSNode(var l: Int, val level: Int) {
  var r: Int = Int.MaxValue
}

class HSInternalNode(val leftNode: HSNode, val rightNode: HSNode, val splitFeat: Int,
                     val splitVal: Double, level: Int) extends HSNode(0, level)

class HSTrees (rangeMin: Vector[Double], rangeMax: Vector[Double], numTrees: Int = 50, sizeLimit: Int = 25,
               maxDepth: Int = 15, seed: Int = 42, windowSize: Int = 256) {
  private val rng = Random
  private val model = ArrayBuffer[HSNode]()

  private var numFeats: Int = rangeMin.length
  private var windowCount = 0

  assert(rangeMin.length == rangeMax.length)
  rng.setSeed(42)

  for (i <- 1 to numTrees) {
    val (min, max) = initWorkSpace(rangeMin, rangeMax)
    model += buildSingleTree(min, max, 0)
  }

  private def initWorkSpace(rangeMin: Vector[Double], rangeMax: Vector[Double]): (Vector[Double], Vector[Double]) = {
    var min = ArrayBuffer[Double]()
    var max = ArrayBuffer[Double]()

    for ((ma, mi) <- rangeMax zip rangeMin) {
      val s = rng.nextDouble() * (ma - mi)
      val sigma = 2 * math.max(s - mi, ma - s)
      min += s - sigma
      max += s + sigma
    }

    (Vector() ++ min, Vector() ++ max)
  }

  private def buildSingleTree(min: Vector[Double], max: Vector[Double],
                              currDepth: Int): HSNode = {
    if (currDepth == maxDepth)
      new HSNode(0, currDepth)
    else {
      val splitFeat: Int = rng.nextInt(numFeats)
      val splitValue: Double = (min(splitFeat) + max(splitFeat)) / 2

      val newMax = max.updated(splitFeat, splitValue)
      val left = buildSingleTree(min, newMax, currDepth + 1)

      val newMin = min.updated(splitFeat, splitValue)
      val right = buildSingleTree(newMin, max, currDepth + 1)

      new HSInternalNode(left, right, splitFeat, splitValue, currDepth)
    }
  }

  @scala.annotation.tailrec
  private def scoreTree(instance: Vector[Double], node: HSNode): Int = {
    if (node.level == maxDepth || node.r < sizeLimit) {
      if (node.r == Int.MaxValue)
        Int.MaxValue
      else
        node.r * math.pow(2, node.level).toInt
    } else {
      if (instance(node.asInstanceOf[HSInternalNode].splitFeat) < node.asInstanceOf[HSInternalNode].splitVal)
        scoreTree(instance, node.asInstanceOf[HSInternalNode].leftNode)
      else
        scoreTree(instance, node.asInstanceOf[HSInternalNode].rightNode)
    }
  }

  @scala.annotation.tailrec
  private def updateMass(instance: Vector[Double], node: HSNode): Unit = {
    node.l += 1
    if (node.level < maxDepth) {
      if (instance(node.asInstanceOf[HSInternalNode].splitFeat) < node.asInstanceOf[HSInternalNode].splitVal)
        updateMass(instance, node.asInstanceOf[HSInternalNode].leftNode)
      else
        updateMass(instance, node.asInstanceOf[HSInternalNode].rightNode)
    }
  }

  private def updateTree(node: HSNode): Unit = {
    if (node.level == maxDepth || node.r < sizeLimit) {
      node.r = node.l
      node.l = 0
    } else {
      updateTree(node.asInstanceOf[HSInternalNode].leftNode)
      updateTree(node.asInstanceOf[HSInternalNode].rightNode)
    }
  }

  def score(instance: Vector[Double]): Double = {
    var score: Int = 0

    for (tree <- model) {
      val tempScore = scoreTree(instance, tree)
      if (tempScore == Int.MaxValue) {
        score = Int.MaxValue
      } else {
        score += tempScore
      }
      updateMass(instance, tree)
    }

    windowCount += 1
    if (windowCount == windowSize) {
      for (tree <- model)
        updateTree(tree)
      windowCount = 0
    }

    score.toDouble
  }
}

class HSTreesMapper(rangeMin: Vector[Double], rangeMax: Vector[Double],
                    numTrees: Int = 50, sizeLimit: Int = 25, maxDepth: Int = 15,
                    seed: Int = 42, windowSize: Int = 256) extends RichMapFunction[Vector[Double], Double] {
  var model: HSTrees = _

  override def open(parameters: Configuration): Unit = {
    model = new HSTrees(rangeMin, rangeMax, numTrees, sizeLimit, maxDepth, seed, windowSize)
  }

  override def map(value: Vector[Double]): Double = {
    model.score(value)
  }
}
