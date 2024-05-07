package cs6320

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.{tensorflow=>tf}
import org.tensorflow.SavedModelBundle
import scala.collection.JavaConverters._
import scala.collection.mutable.WrappedArray
import org.apache.spark.SparkConf

object MachineTranslation {
  val conf = new SparkConf().setAppName("LanguageTranslation").setMaster("local[*]").set("spark.ui.retainedJobs", "1000").set("spark.ui.retainedStages", "1000") // Adjust as needed
  val spark = SparkSession.builder().appName("MachineTranslation").config(conf).config("spark.rapids.sql.enabled", "true").getOrCreate()

  def main(args: Array[String]):Unit = {
    val trainData = spark.read.parquet("SparkMachineTranslation/project/data/wili-2018/langdetection_train.parquet").repartition(8).localCheckpoint().cache
    val model = LanguageDetection.trainMLP(trainData,"Sentence","Label",2048,Array(256))
    LanguageDetection.save(model,"SparkMachineTranslation/project/models/langdetect")
   
    val loadedModel = LanguageDetection.load("SparkMachineTranslation/project/models/langdetect")
    val testData = spark.read.parquet("SparkMachineTranslation/project/data/wili-2018/langdetection_test.parquet").repartition(8).localCheckpoint().cache
    val accuracy = LanguageDetection.evaluateMLP(loadedModel,testData,"Sentence","Label")
    println(accuracy)
  }
}
