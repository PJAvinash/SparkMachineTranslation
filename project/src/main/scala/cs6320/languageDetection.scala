package cs6320

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Normalizer,CountVectorizer}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier,DecisionTreeClassifier,MultilayerPerceptronClassificationModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.{col, lit, udf, when,trim,upper,regexp_replace,lower,length}
import scala.util.Try

object LanguageDetection extends Serializable {
  /* 
  **  Add more languages if you want to support 
  **  below list is lexicographically ordered
  */
  val supportedLanguages = Seq("ace", "afr", "als", "amh", "ang", "ara", "arg", "arz", "asm", "ast", "ava", "aym", "azb", "aze", "bak", "bar", "bcl", "be-tarask", "bel", "ben", "bho", "bjn", "bod", "bos", "bpy", "bre", "bul", "bxr", "cat", "cbk", "cdo", "ceb", "ces", "che", "chr", "chv", "ckb", "cor", "cos", "crh", "csb", "cym", "dan", "deu", "diq", "div", "dsb", "dty", "egl", "ell", "eng", "epo", "est", "eus", "ext", "fao", "fas", "fin", "fra", "frp", "fry", "fur", "gag", "gla", "gle", "glg", "glk", "glv", "grn", "guj", "hak", "hat", "hau", "hbs", "heb", "hif", "hin", "hrv", "hsb", "hun", "hye", "ibo", "ido", "ile", "ilo", "ina", "ind", "isl", "ita", "jam", "jav", "jbo", "jpn", "kaa", "kab", "kan", "kat", "kaz", "kbd", "khm", "kin", "kir", "koi", "kok", "kom", "kor", "krc", "ksh", "kur", "lad", "lao", "lat", "lav", "lez", "lij", "lim", "lin", "lit", "lmo", "lrc", "ltg", "ltz", "lug", "lzh", "mai", "mal", "map-bms", "mar", "mdf", "mhr", "min", "mkd", "mlg", "mlt", "mon", "mri", "mrj", "msa", "mwl", "mya", "myv", "mzn", "nan", "nap", "nav", "nci", "nds", "nds-nl", "nep", "new", "nld", "nno", "nob", "nrm", "nso", "oci", "olo", "ori", "orm", "oss", "pag", "pam", "pan", "pap", "pcd", "pdc", "pfl", "pnb", "pol", "por", "pus", "que", "roa-tara", "roh", "ron", "rue", "rup", "rus", "sah", "san", "scn", "sco", "sgs", "sin", "slk", "slv", "sme", "sna", "snd", "som", "spa", "sqi", "srd", "srn", "srp", "stq", "sun", "swa", "swe", "szl", "tam", "tat", "tcy", "tel", "tet", "tgk", "tgl", "tha", "ton", "tsn", "tuk", "tur", "tyv", "udm", "uig", "ukr", "urd", "uzb", "vec", "vep", "vie", "vls", "vol", "vro", "war", "wln", "wol", "wuu", "xho", "xmf", "yid", "yor", "zea", "zh-yue", "zho")
  val unknown_lang_code = "UNKOWN"
  val allLangs = Seq(unknown_lang_code) ++ supportedLanguages
  val supportedLanguagesIndexMap:Map[String,Int] = Range(0,allLangs.size).map(t => (allLangs(t),t)).toMap
  val unknown_lang_index:Int = supportedLanguagesIndexMap.getOrElse(unknown_lang_code,0)

  /* 
  ** categorical feature -> integer ( will be casted to double in Spark ML lib ) and integer ->  categorical feature UDFs
  */
  val langcodeToIndexUDF = udf((inputCode:String)=> {supportedLanguagesIndexMap.getOrElse(inputCode.trim.toLowerCase,unknown_lang_index)})
  val indexToLangCodeUDF = udf((inputCode:Int)=> {Try(allLangs(inputCode)).getOrElse(unknown_lang_code)})

 
  /* 
  ** Internal temporary columns , should not have any collison with input Dataset columns 
  */
  val sentence_mi = "SENTENCE"
  val language_code_mi = "LANGUAGE_CODE"
  val lang_index_mi = "LANGUAGE_INDEX"
  val tokens_mi = "TOKENS"
  val rawfeatures_mi = "RAW_FEATURES"
  val normfeatures_mi = "NORMALIZED_FEATURES"
  val langprediction_mi = "LANGUAGE_PREDICTION"

  /* 
  ** Custom tokenization functions 
  ** Preferred windowSize = 6 , stepSize = 3
  */
  def getTokens(inputString: String, windowSize: Int,stepSize:Int): Seq[String] = {
    return Range(0,inputString.size,stepSize).map(t => inputString.slice(t,t+windowSize))
  }
  def tokenizeDF(inputCol:String,outputCol:String,windowSize: Int,stepSize:Int)(inputDF:DataFrame):DataFrame ={
    val getTokensUDF = udf((inputString: String) => Try(getTokens(inputString,windowSize,stepSize)).getOrElse(Seq.empty[String]))
    return inputDF.withColumn(outputCol,getTokensUDF(col(inputCol)))
  }

  def renameInputsTrain(inputSentenceCol:String,inputTargetCol:String)(inputDF:DataFrame):DataFrame ={
    val x = Seq(inputSentenceCol,inputTargetCol)
    val newSelectExpr = inputDF.columns.filter(t => !x.contains(t)).map(t => col(t)) ++ Seq(col(inputSentenceCol).as(sentence_mi),col(inputTargetCol).as(language_code_mi))
    return inputDF.select(newSelectExpr:_*)
  }

  def renameInputsInfer(inputSentenceCol:String)(inputDF:DataFrame):DataFrame ={
    return inputDF.withColumn(sentence_mi,col(inputSentenceCol))
  }

  def renameOutputs(inputSentenceCol:String)(inputDF:DataFrame):DataFrame ={
    return inputDF.withColumn(inputSentenceCol,col(sentence_mi))
  }

  def removeUnsupportedLang(languageCol:String)(inputDF:DataFrame):DataFrame  ={

    return inputDF.withColumn(languageCol,when(langcodeToIndexUDF(col(languageCol)) =!= unknown_lang_index,col(languageCol)).otherwise(lit(unknown_lang_code)))
  }

  def clean(inputCol: String, outputCol: String)(inputDF: DataFrame): DataFrame = {
    return inputDF.withColumn(outputCol, trim(upper(regexp_replace(col(inputCol), "\\d", ""))))
    }


  def preprocessTrain(sentenceCol:String,languageCodeCol:String)(inputDF:DataFrame):DataFrame = {
    return inputDF.transform(renameInputsTrain(sentenceCol,languageCodeCol)).transform(removeUnsupportedLang(sentence_mi)).withColumn(lang_index_mi,langcodeToIndexUDF(col(language_code_mi))).transform(clean(sentence_mi,sentence_mi)).transform(tokenizeDF(sentence_mi,tokens_mi,6,3))
  }

  def preprocessInfer(sentenceCol:String)(inputDF:DataFrame):DataFrame = {
    return inputDF.transform(renameInputsInfer(sentenceCol)).transform(clean(sentence_mi,sentence_mi)).transform(tokenizeDF(sentence_mi,tokens_mi,6,3))
  }
  def postprocessInfer(indexColName:String,outputCol:String)(inputDF:DataFrame):DataFrame ={
    return inputDF.withColumn(outputCol,indexToLangCodeUDF(col(indexColName)))
  }

  def languageDetectionPipeline(hashingFeatureDim:Int,hiddenLayerDims:Array[Int],numLanguages:Int):org.apache.spark.ml.Pipeline = {
    val hashingTF = new HashingTF().setInputCol(tokens_mi).setOutputCol(rawfeatures_mi).setNumFeatures(hashingFeatureDim)
    val normalizer = new Normalizer().setInputCol(rawfeatures_mi).setOutputCol(normfeatures_mi).setP(1.0)
    val layers = Array[Int](hashingFeatureDim) ++ (hiddenLayerDims) ++ Array[Int](numLanguages)
    val mlpTrainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(16).setStepSize(0.05).setSeed(1234L).setMaxIter(2).setFeaturesCol(normfeatures_mi).setLabelCol(lang_index_mi).setPredictionCol(langprediction_mi)
    //val dt = new DecisionTreeClassifier().setLabelCol(lang_index_mi).setFeaturesCol(normfeatures_mi).setPredictionCol(langprediction_mi)
    val pipeline = new Pipeline().setStages(Array(hashingTF,normalizer,mlpTrainer))
    return pipeline
   }

   def train(trainingData:DataFrame,sourceCol:String, targetCol:String,hashingFeatureDim:Int,hiddenLayerDims:Array[Int]):PipelineModel = {
        val preprocessedDF = trainingData.transform(preprocessTrain(sourceCol,targetCol))
        //val numLanguages = preprocessedDF.select(language_code_mi).distinct.count
        val trainingPL = languageDetectionPipeline(hashingFeatureDim,hiddenLayerDims,allLangs.size)
        val trainedmodel = trainingPL.fit(preprocessedDF.select(tokens_mi,lang_index_mi))
        return trainedmodel
   }

   def predict(trainedPL:PipelineModel,sentenceCol:String,outputCol:String)(inputDF:DataFrame):DataFrame = { 
       return trainedPL.transform(inputDF.transform(preprocessInfer(sentenceCol))).transform(postprocessInfer(langprediction_mi,outputCol))
   }
   def evaluate(trainedPL:PipelineModel,testData:DataFrame,sentenceCol:String,expectlabelcol:String):Double = {
     val predictionDF = testData.withColumn(lang_index_mi,langcodeToIndexUDF(col(expectlabelcol))).transform(LanguageDetection.predict(trainedPL,sentenceCol,"EVALUATION_OUTPUT"))
     val evaluator = new MulticlassClassificationEvaluator().setLabelCol(lang_index_mi).setPredictionCol(langprediction_mi).setMetricName("accuracy")
     return evaluator.evaluate(predictionDF)
   }

   /*  
   ** usage : trainData.transform(LanguageDetection.preprocessMLP(SourceColumn,hashingFeatureDim))
   ** hashingFeatureDim : Need to match with MLP model used training 
   */
   private def preprocessMLP(sourceCol:String,hashingFeatureDim:Int)(inputDF:DataFrame):DataFrame = {
    val hashingTF = new HashingTF().setInputCol(tokens_mi).setOutputCol(rawfeatures_mi).setNumFeatures(hashingFeatureDim)
    val normalizer = new Normalizer().setInputCol(rawfeatures_mi).setOutputCol(normfeatures_mi).setP(1.0)
    return normalizer.transform(hashingTF.transform(inputDF.transform(LanguageDetection.tokenizeDF(sourceCol,tokens_mi,6,3))))
   }
   /*  
   ** usage : trainedModel = LanguageDetection.trainMLP()
   ** hashingFeatureDim : Need to match with MLP model used training 
   */

   def trainMLP(trainingData:DataFrame,sourceCol:String, targetCol:String,hashingFeatureDim:Int,hiddenLayerDims:Array[Int]):MultilayerPerceptronClassificationModel = {
    var preprocessedDF = trainingData.transform(preprocessMLP(sourceCol,hashingFeatureDim)).withColumn(LanguageDetection.lang_index_mi,LanguageDetection.langcodeToIndexUDF(col(targetCol)))
    val mlp =  new MultilayerPerceptronClassifier().setLayers(Array(hashingFeatureDim) ++ hiddenLayerDims ++ Array(allLangs.size)).setBlockSize(16).setStepSize(0.05).setSeed(1234L).setMaxIter(1000).setFeaturesCol(normfeatures_mi).setLabelCol(lang_index_mi).setPredictionCol(langprediction_mi)
    val model = mlp.fit(preprocessedDF)
    return model
   }
   /*  
   ** usage : InputData.transform(LanguageDetection.predictMLP(trainedModel,SourceColumn,TargetColumn))
   ** Output dataset will have some additional columns like feature column, token column etc. 
   ** They should be removed with select expression on the output
   ** constraints: InputDF should have sourceCol.
   */
   def predictMLP(model:MultilayerPerceptronClassificationModel,sourceCol:String,outputCol:String)(inputDF:DataFrame):DataFrame = {
     val precitionColumn = model.getPredictionCol
     return model.transform(inputDF.transform(preprocessMLP(sourceCol,model.numFeatures))).withColumn(outputCol,LanguageDetection.indexToLangCodeUDF(col(precitionColumn)))
   }

   /*  
   ** usage : LanguageDetection.evaluateMLP(trainedModel,testData,sentenceCol,expectlabelcol))
   ** returns a double( accurace on testData)
   ** constraints :  testData should have 'sentenceCol' & 'expectlabelcol' column 
   */ 

   def evaluateMLP(model:MultilayerPerceptronClassificationModel,testData:DataFrame,sentenceCol:String,expectlabelcol:String):Double = {
    val predictionDF = model.transform(testData.transform(preprocessMLP(sentenceCol,model.numFeatures))).withColumn(lang_index_mi,LanguageDetection.langcodeToIndexUDF(col(expectlabelcol)))
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol(lang_index_mi).setPredictionCol(model.getPredictionCol).setMetricName("accuracy")
    return evaluator.evaluate(predictionDF)
   }


   /*
   ** usage: load the saved model 
   ** val savedModel = 
   ** val newModel = retrainMLP()
   */

   def retrainMLP(existingModel:MultilayerPerceptronClassificationModel,newData:DataFrame,sourceCol:String, targetCol:String):MultilayerPerceptronClassificationModel = {
    var preprocessedDF = newData.transform(preprocessMLP(sourceCol,existingModel.numFeatures)).withColumn(LanguageDetection.lang_index_mi,LanguageDetection.langcodeToIndexUDF(col(targetCol)))
    val newMLP_Estimator = new MultilayerPerceptronClassifier().setLayers(existingModel.getLayers).setInitialWeights(existingModel.getInitialWeights).setBlockSize(existingModel.getBlockSize).setSeed(existingModel.getSeed).setMaxIter(existingModel.getMaxIter).setFeaturesCol(existingModel.getFeaturesCol).setLabelCol(existingModel.getLabelCol).setPredictionCol(existingModel.getPredictionCol)
    val retrainedModel = newMLP_Estimator.fit(preprocessedDF)
    return retrainedModel
   }

   /* 
   ** Model IO
   ** the following function is just a wrapper
   */
   def save(trainedModel:MultilayerPerceptronClassificationModel, modelPath:String):Unit = {
    trainedModel.write.overwrite().save(modelPath)
   }
   def load(modelPath:String):MultilayerPerceptronClassificationModel ={
    return MultilayerPerceptronClassificationModel.load(modelPath)
   }

}
 