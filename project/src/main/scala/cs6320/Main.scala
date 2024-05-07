package cs6320

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover,RegexTokenizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import scala.collection.JavaConverters._
import scala.collection.mutable.WrappedArray
import org.apache.spark.SparkConf



import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.annotators.seq2seq.MarianTransformer
import org.apache.spark.ml.Pipeline

import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Normalizer,CountVectorizer}

/**
* * In testing 
* */
object MachineTranslation {
  val conf = new SparkConf().setAppName("LanguageTranslation").setMaster("local[*]").set("spark.ui.retainedJobs", "1000").set("spark.ui.retainedStages", "1000") // Adjust as needed
  val spark = SparkSession.builder().appName("MachineTranslation").config(conf).config("spark.rapids.sql.enabled", "true").getOrCreate()
  import spark.implicits._
  def main(args: Array[String]):Unit = {
    val trainData = spark.read.parquet("SparkMachineTranslation/project/data/wili-2018/langdetection_train.parquet").repartition(8).localCheckpoint().cache
    val model = LanguageDetection.trainMLP(trainData,"Sentence","Label",2048,Array(256))
    LanguageDetection.save(model,"SparkMachineTranslation/project/models/langdetect")
   
    val loadedModel = LanguageDetection.load("SparkMachineTranslation/project/models/langdetect")
    val testData = spark.read.parquet("SparkMachineTranslation/project/data/wili-2018/langdetection_test.parquet").repartition(8).localCheckpoint().cache
    val accuracy = LanguageDetection.evaluateMLP(loadedModel,testData,"Sentence","Label")
    // val langTransData = spark.read.option("header",true).csv("/Users/pjavinash/Documents/Avinash/UTD_MS/4th_sem/CS6320_NLP/Project_git_issue/SparkMachineTranslation/project/data/spa_eng_data.csv")
    // langTransData.show

    // val tokenizer = new RegexTokenizer().setInputCol("Spanish").setOutputCol("tokens").setPattern("\\W")
    // val remover = new StopWordsRemover().setInputCol("tokens").setOutputCol("clean_tokens")
    // val countVectorizer = new CountVectorizer().setInputCol("clean_tokens").setOutputCol("features")
    // val translationModel = MarianTransformer.pretrained("marian-mt-es-en")
    //Define pipeline


    // val documentAssembler = new DocumentAssembler().setInputCol("Spanish").setOutputCol("document")
    // val sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx").setInputCols("document").setOutputCol("sentence")
    // val marian = MarianTransformer.pretrained().setInputCols("sentence").setOutputCol("translation").setMaxInputLength(30)
    // val pipeline = new Pipeline().setStages(Array(documentAssembler,sentence,marian))


    // val pipeline = new Pipeline().setStages(Array(tokenizer, remover, countVectorizer, translationModel))
    // val translatedDF= pipeline.fit(langTransData).transform(langTransData)
    // translatedDF.show
    //println(accuracy)
  }
}
