package cn.tipdm.spark

import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
object Gaussianmixture {

  def main(args: Array[String]): Unit = {
    var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    //.setSparkHome("/opt/spark-1.3.1-bin-hadoop2.6")

    val sc = new SparkContext(conf)
    // Load and parse the data
    val data = sc.textFile("E:\\Ñ¸À×ÏÂÔØ\\spark-1.5.2-bin-hadoop2.6\\spark-1.5.2-bin-hadoop2.6\\data\\mllib\\gmm_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()

    // Cluster the data into two classes using GaussianMixture
    val gmm = new GaussianMixture().setK(3).run(parsedData)

    // output parameters of max-likelihood model
    for (i <- 0 until gmm.k) {
      println(i+"------------"+"weight=%f\nmu=%s\nsigma=\n%s\n" format
        (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
    }

  }

}