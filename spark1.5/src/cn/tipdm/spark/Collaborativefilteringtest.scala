package cn.tipdm.spark

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.log4j.Logger
import org.apache.log4j.Level

object Collaborativefilteringtest {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    //.setSparkHome("/opt/spark-1.3.1-bin-hadoop2.6")

    val sc = new SparkContext(conf)
    // Load and parse the data file.
    val data = sc.textFile("E:\\迅雷下载\\spark-1.5.2-bin-hadoop2.6\\spark-1.5.2-bin-hadoop2.6\\data\\mllib\\als\\test.data")
    // Create label, feature, weight tuples from input data with weight set to default value 1.0.
    val ratings = data.map(_.split(',') match {
      case Array(user, item, rate) =>
        Rating(user.toInt, item.toInt, rate.toDouble)
    })
    
    val splits = ratings.randomSplit(Array(0.8, 0.2), seed = 111l)
    val test = splits(1)
    //第一个数据
    println("ratings.first"+ratings.first)
    println("用户数量："+ratings.map(_.user).count)
    println("产品数量："+ratings.map(_.product).count)
    // Build the recommendation model using ALS
    val rank = 10
    val numIterations = 10
    val model = ALS.train(ratings, rank, numIterations, 0.01)
    
    model.userFeatures.collect().foreach(println)
    // Evaluate the model on rating data
    val usersProducts = test.map {
      case Rating(user, product, rate) =>
        (user, product)
    }
    println("usersProducts====================")
    usersProducts.collect().foreach(println)
    
    val predictions =
      model.predict(usersProducts).map {
        case Rating(user, product, rate) =>
          ((user, product), rate)
      }
    println("predictions====================")
    predictions.collect().foreach(println)
    val ratesAndPreds = ratings.map {
      case Rating(user, product, rate) =>
        ((user, product), rate)
    }.join(predictions)
    
    println("ratesAndPreds====================")
    ratesAndPreds.collect().foreach(println)
    val MSE = ratesAndPreds.map {
      case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
    }.mean()
    println("Mean Squared Error = " + MSE)

  }

}