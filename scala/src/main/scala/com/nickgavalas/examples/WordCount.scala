package com.nickgavalas.examples

// import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.scala._
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.api.scala.ExecutionEnvironment
// import org.apache.flink.configuration.Configuration
// import org.apache.flink.dropwizard.metrics.DropwizardMeterWrapper
// import org.apache.flink.metrics.Meter

object WordCount {

  def main(args: Array[String]) {

    val params = ParameterTool.fromArgs(args)

    val env: ExecutionEnvironment = ExecutionEnvironment.getExecutionEnvironment

    val text = env.fromElements(
      "sentence",
      "this is a sentence",
      "another sentence",
      "and another",
      "...",
      "yet one more",
      "ok I am done",
      "nop here's more lel"
    )

    val counts = text
        //.map(new ThroughputMeasuringMapper())
        .flatMap(_.toLowerCase().split("\\W+"))
        .filter(_.nonEmpty)
        .map((_, 1))
        .groupBy(0)
        .sum(1)

    counts.print()
  }
}

//class ThroughputMeasuringMapper() extends RichMapFunction[String, String] {
//  @transient private var meter: Meter = _
//
//  override def open(parameters: Configuration): Unit = {
//    val dropwizardMeter: com.codahale.metrics.Meter  = new com.codahale.metrics.Meter()
//
//    meter = getRuntimeContext
//      .getMetricGroup
//      .meter("MyMeter", new DropwizardMeterWrapper(dropwizardMeter))
//  }
//
//  override def map(value: String): String = {
//    meter.markEvent()
//    value
//  }
//}
