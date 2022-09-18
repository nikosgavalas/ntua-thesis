/*
  Latency & Throughput
 */

package com.nickgavalas

import com.nickgavalas.ml.MultivariateGaussianMapper
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.api.scala._
import org.apache.flink.streaming.api.scala.{DataStream, StreamExecutionEnvironment}

object Launcher {
  def main(args: Array[String]): Unit = {
    val params = ParameterTool.fromArgs(args)

    val algo: String = params.get("algo", "gauss")
    val dataset: String = params.get("dataset", "/home/nick/ntua/thesis/data/synthetic/set_255_45_2_l.csv")

    val senv: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment

    val data: DataStream[Vector[Double]] = senv
      .readTextFile(dataset)
      .map {
        row =>
          val nums = row.split(",").map(_.toDouble)
          nums.slice(0, nums.length - 1).toVector
      }

    algo match {
      case "gauss" =>
        val scores = data.map(new MultivariateGaussianMapper(2))
        scores.print()
        senv.execute()
      case "rrcf" => throw new NotImplementedError()
      case "hstrees" => throw new NotImplementedError()
      case "loda" => throw new NotImplementedError()
      case "iforest" => throw new NotImplementedError()
      case _ => println("Invalid algo.")
    }
  }
}
