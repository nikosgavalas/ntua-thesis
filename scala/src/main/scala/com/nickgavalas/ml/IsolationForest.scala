package com.nickgavalas.ml

import java.nio.file.{Files, Paths}

import org.json4s._
import org.json4s.native.JsonMethods._
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration

import scala.collection.mutable.ArrayBuffer

abstract class Node() {}

case class InternalNode(var left: Node, var right: Node, splitFeat: Int, splitValue: Double) extends Node

case class ExternalNode(size: Int) extends Node

class IsolationForest(path: String) {
  private val forest: ArrayBuffer[Node] = ArrayBuffer[Node]()
  private val EULERS_CONST: Double = 0.5772156649

  // default value. The actual one is loaded from the serialized model.
  private var subSampleSize: Int = 256

  loadModel(path)

  def loadModel(path: String): Unit = {
    // TODO: handle exceptions
    val serializedModel = new String(Files.readAllBytes(Paths.get(path)))
    val deserializedModel = parse(serializedModel).values.asInstanceOf[Map[String, Any]]

    val forest = deserializedModel("model").asInstanceOf[List[List[Map[String, Any]]]]
    this.subSampleSize = deserializedModel("size").asInstanceOf[BigInt].toInt
    forest.foreach { tree => this.forest.append(deSerializeTree(tree.to[ArrayBuffer])) }
  }

  private def deSerializeTree(treeArray: ArrayBuffer[Map[String, Any]]): Node = {
    val curr = treeArray.head
    if (curr("t").equals("e"))
      ExternalNode(curr("s").asInstanceOf[BigInt].toInt)
    else {
      val newNode = InternalNode(null, null, curr("f").asInstanceOf[BigInt].toInt, curr("v").asInstanceOf[Double])
      treeArray.remove(0)
      newNode.left = deSerializeTree(treeArray)
      treeArray.remove(0)
      newNode.right = deSerializeTree(treeArray)
      newNode
    }
  }

  private def averagePathLength(n: Int): Double = {
    if (n < 2.0)
      1.0
    else
      2.0 * (math.log(n - 1.0) + EULERS_CONST) - (2.0 * (n - 1.0) / n)
  }

  @scala.annotation.tailrec
  private def pathLength(instance: Vector[Double], node: Node, i: Int): Double = {
    node match {
      case extNode: ExternalNode => i.toDouble + averagePathLength(extNode.size)
      case intNode: InternalNode =>
        if (instance(intNode.splitFeat) < intNode.splitValue)
          pathLength(instance, intNode.left, i + 1)
        else
          pathLength(instance, intNode.right, i + 1)
      case _ => 0.0 // <=== see https://stackoverflow.com/questions/5000376/scala-match-error
    }
  }

  private def anomalyScoreForest(mean: Double): Double= {
    math.pow(2, -mean / averagePathLength(subSampleSize))
  }

  // typically one would flag all instances with score > 0.5
  def score(instance: Vector[Double]): Double = {
    anomalyScoreForest(forest.map(pathLength(instance, _, 0)).sum / forest.length)
  }
}

class IsolationForestMapper(modelPath: String) extends RichMapFunction[Vector[Double], Double] {
  var model: IsolationForest = _

  override def open(parameters: Configuration): Unit = {
    model = new IsolationForest(modelPath)
  }

  override def map(value: Vector[Double]): Double = {
    model.score(value)
  }
}
