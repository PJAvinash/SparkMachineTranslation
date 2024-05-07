name := "SparkMachineTranslation"
version := "0.1"
organization := "cs6320"
scalaVersion := "2.12.18"

// Spark dependencies
val sparkVersion = "3.3.2"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.hadoop" % "hadoop-client" % sparkVersion,
  "org.tensorflow" % "tensorflow" % "1.15.0",
  "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "5.3.3",
  "com.johnsnowlabs.nlp" %% "spark-nlp-aarch64" % "5.3.3",
  "com.johnsnowlabs.nlp" %% "spark-nlp" % "5.3.3",

)
// Package configuration
// addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "1.1.0")
// assembly / assemblyJarName := "SparkMachineTranslation.jar"
// assembly / assemblyMergeStrategy := {
//   case PathList("META-INF", xs @ _*) => MergeStrategy.discard
//   case x => MergeStrategy.first
// }
// See https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html for instructions on how to publish to Sonatype.
