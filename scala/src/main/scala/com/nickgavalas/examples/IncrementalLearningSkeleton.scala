/* This example is taken from the official Flink examples (https://github.com/apache/flink/blob/master/flink-examples/flink-examples-streaming/src/main/scala/org/apache/flink/streaming/scala/examples/ml/IncrementalLearningSkeleton.scala)
 *
 * Other useful links and relevant examples:
 * https://ci.apache.org/projects/flink/flink-docs-stable/dev/stream/state/state.html
 * https://github.com/dataArtisans/flink-training-exercises/blob/master/src/main/java/com/dataartisans/flinktraining/solutions/datastream_java/state/RidesAndFaresSolution.java
 */

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nickgavalas.examples

import java.util.concurrent.TimeUnit

import org.apache.flink.streaming.api.TimeCharacteristic
import org.apache.flink.streaming.api.functions.source.SourceFunction
import org.apache.flink.streaming.api.functions.source.SourceFunction.SourceContext
import org.apache.flink.streaming.api.scala.{DataStream, StreamExecutionEnvironment}
import org.apache.flink.api.scala._
import org.apache.flink.streaming.api.functions.AssignerWithPunctuatedWatermarks
import org.apache.flink.streaming.api.functions.co.CoMapFunction
import org.apache.flink.streaming.api.scala.function.AllWindowFunction
import org.apache.flink.streaming.api.watermark.Watermark
import org.apache.flink.streaming.api.windowing.time.Time
import org.apache.flink.streaming.api.windowing.windows.TimeWindow
import org.apache.flink.util.Collector

object IncrementalLearningSkeleton {
  def main(args: Array[String]): Unit = {
    val senv = StreamExecutionEnvironment.getExecutionEnvironment
    senv.setStreamTimeCharacteristic(TimeCharacteristic.EventTime)

    val trainingData: DataStream[Int] = senv.addSource(new FiniteTrainingDataSource)
    val newData: DataStream[Int] = senv.addSource(new FiniteNewDataSource)

    val model: DataStream[Array[Double]] = trainingData
      .assignTimestampsAndWatermarks(new LinearTimestamp)
      .timeWindowAll(Time.of(5000, TimeUnit.MILLISECONDS))
      .apply(new PartialModelBuilder)

    val prediction: DataStream[Int] = newData.connect(model).map(new Predictor)

    prediction.print()
    senv.execute()
  }

  private class FiniteNewDataSource extends SourceFunction[Int] {
    override def run(ctx: SourceContext[Int]) = {
      Thread.sleep(10)
      (0 until 100).foreach{ _ =>
        Thread.sleep(10)
        ctx.collect(1)
      }
    }

    override def cancel() = {
      // No cleanup needed
    }
  }

  /**
    * Feeds new training data for the partial model builder. By default it is
    * implemented as constantly emitting the Integer 1 in a loop.
    */
  private class FiniteTrainingDataSource extends SourceFunction[Int] {
    override def run(ctx: SourceContext[Int]) = List(-1, -1, -1, -1, -1).foreach( _ => ctx.collect(1))

    override def cancel() = {
      // No cleanup needed
    }
  }

  private class LinearTimestamp extends AssignerWithPunctuatedWatermarks[Int] {
    var counter = 0L

    override def extractTimestamp(element: Int, previousElementTimestamp: Long): Long = {
      counter += 10L
      counter
    }

    override def checkAndGetNextWatermark(lastElement: Int, extractedTimestamp: Long) = {
      new Watermark(counter - 1)
    }
  }

  private class PartialModelBuilder extends AllWindowFunction[Int, Array[Double], TimeWindow] {

    protected def buildPartialModel(values: Iterable[Int]): Array[Double] = {
      Array[Double](1.11)
    }

    override def apply(window: TimeWindow,
                       values: Iterable[Int],
                       out: Collector[Array[Double]]): Unit = {
      out.collect(buildPartialModel(values))
    }
  }

  private class Predictor extends CoMapFunction[Int, Array[Double], Int] {

    var batchModel: Array[Double] = _
    var partialModel: Array[Double] = _

    override def map1(value: Int): Int = {
      // Return newData
      predict(value)
    }

    override def map2(value: Array[Double]): Int = {
      // Update model
      partialModel = value
      batchModel = getBatchModel()
      1
    }

    // pulls model built with batch-job on the old training data
    protected def getBatchModel(): Array[Double] = Array[Double](0)

    // performs newData using the two models
    protected def predict(inTuple: Int): Int = 0
  }
}
