ThisBuild / resolvers ++= Seq(
  "Apache Development Snapshot Repository" at "https://repository.apache.org/content/repositories/snapshots/",
  Resolver.mavenLocal
)

name := "flinkML"
version := "0.1"
scalaVersion := "2.11.12"

val flinkVersion = "1.6.1"
val vegasVersion = "0.3.11"
val scalatestVersion = "3.0.5"
val json4sVersion = "3.4.0"

lazy val root = (project in file(".")).
  settings(
    libraryDependencies ++= Seq(
      // flink
      "org.apache.flink" %% "flink-scala" % flinkVersion % "provided",
      "org.apache.flink" %% "flink-streaming-scala" % flinkVersion % "provided",
      "org.apache.flink" %% "flink-clients" % flinkVersion,
      //"org.apache.flink" % "flink-metrics-dropwizard" % flinkVersion,

      // visualize
      //"org.vegas-viz" %% "vegas" % vegasVersion,

      // scalatest
      "org.scalatest" %% "scalatest" % scalatestVersion % "test",

      // JSON
      "org.json4s" %% "json4s-native" % json4sVersion
    )
  )

// Change this accordingly
Compile / mainClass := Some("com.nickgavalas.Launcher")

// make run command include the provided dependencies
Compile / run  := Defaults.runTask(Compile / fullClasspath,
  Compile / run / mainClass,
  Compile / run / runner
).evaluated

// stays inside the sbt console when we press "ctrl-c" while a Flink programme executes with "run" or "runMain"
//Compile / run / fork := true
//Global / cancelable := true

assemblyMergeStrategy in assembly ~= { old =>
  {
    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
    case x => MergeStrategy.first
  }
}

// exclude Scala library from assembly
assembly / assemblyOption  := (assembly / assemblyOption).value.copy(includeScala = false)
