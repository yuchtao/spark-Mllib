package cn.tipdm.spark

import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object EnsemblesGBTsRegression {

  def main(args: Array[String]): Unit = {
     var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    //.setSparkHome("/opt/spark-1.3.1-bin-hadoop2.6")

    val sc = new SparkContext(conf)
    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(sc, "E:\\Ñ¸À×ÏÂÔØ\\spark-1.5.2-bin-hadoop2.6\\spark-1.5.2-bin-hadoop2.6\\data\\mllib\\sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a GradientBoostedTrees model.
    //  The defaultParams for Regression use SquaredError by default.
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.maxDepth = 5
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    labelsAndPredictions.collect.foreach(println)
    val testMSE = labelsAndPredictions.map { case (v, p) => math.pow((v - p), 2) }.mean()
    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression GBT model:\n" + model.toDebugString)

  }

}