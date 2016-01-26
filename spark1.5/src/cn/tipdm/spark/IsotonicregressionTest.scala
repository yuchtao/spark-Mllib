package cn.tipdm.spark

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.IsotonicRegression
import org.apache.spark.mllib.regression.IsotonicRegressionModel
import org.apache.spark.mllib.regression.{IsotonicRegression, IsotonicRegressionModel}

object IsotonicregressionTest {
  def main(args: Array[String]) {
    var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    //.setSparkHome("/opt/spark-1.3.1-bin-hadoop2.6")

    val sc = new SparkContext(conf)
    // Load and parse the data file.
    val data = sc.textFile("E:\\Ñ¸À×ÏÂÔØ\\spark-1.5.2-bin-hadoop2.6\\spark-1.5.2-bin-hadoop2.6\\data\\mllib\\sample_isotonic_regression_data.txt")
    // Create label, feature, weight tuples from input data with weight set to default value 1.0.
    val parsedData = data.map { line =>
      val parts = line.split(',').map(_.toDouble)
      (parts(0), parts(1), 1.0)
    }
    data.collect.foreach(println)
    println("11111111111111111111")
    // Split data into training (60%) and test (40%) sets.
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)
    test.collect.foreach(println)
    // Create isotonic regression model from training data.
    // Isotonic parameter defaults to true so it is only shown for demonstration
    val model = new IsotonicRegression().setIsotonic(true).run(training)

    // Create tuples of predicted and real labels.
    val predictionAndLabel = test.map { point =>
      val predictedLabel = model.predict(point._2)
      (predictedLabel, point._1)
    }
    println("=========================")
    predictionAndLabel.collect.foreach(println)
    // Calculate mean squared error between predicted and real labels.
    val meanSquaredError = predictionAndLabel.map { case (p, l) => math.pow((p - l), 2) }.mean()
    println("Mean Squared Error = " + meanSquaredError)

  }
}