package cn.tipdm.spark

import org.apache.spark.mllib.regression.IsotonicRegression
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object RegressionTest {

  def main(args: Array[String]): Unit = {
    var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    //.setSparkHome("/opt/spark-1.3.1-bin-hadoop2.6")

    val sc = new SparkContext(conf)
    val data = sc.textFile("hdfs://hdp1.tipdm.com:8020/tmp/sparktest/data/mllib/sample_isotonic_regression_data.txt")

    // Create label, feature, weight tuples from input data with weight set to default value 1.0.
    val parsedData = data.map { line =>
      val parts = line.split(',').map(_.toDouble)
      (parts(0), parts(1), 1.0)
    }

    // Split data into training (60%) and test (40%) sets.
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    // Create isotonic regression model from training data.
    // Isotonic parameter defaults to true so it is only shown for demonstration
    val model = new IsotonicRegression().setIsotonic(true).run(training)

    // Create tuples of predicted and real labels.
    val predictionAndLabel = test.map { point =>
      val predictedLabel = model.predict(point._2)
      (predictedLabel, point._1)
    }

    // Calculate mean squared error between predicted and real labels.
    val meanSquaredError = predictionAndLabel.map { case (p, l) => math.pow((p - l), 2) }.mean()
    println("Mean Squared Error = " + meanSquaredError)

  }

}