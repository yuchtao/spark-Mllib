package cn.tipdm.spark
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.Logger
import org.apache.log4j.Level

object KmeansTest {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    //.setSparkHome("/opt/spark-1.3.1-bin-hadoop2.6")

    val sc = new SparkContext(conf)
    // Load and parse the data
    val data = sc.textFile("E:\\Ñ¸À×ÏÂÔØ\\spark-1.5.2-bin-hadoop2.6\\spark-1.5.2-bin-hadoop2.6\\data\\mllib\\kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 3
    val numIterations = 30
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    clusters.clusterCenters.iterator.foreach(println)
    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)
  }

}