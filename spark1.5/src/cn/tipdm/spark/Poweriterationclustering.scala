package cn.tipdm.spark

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.PowerIterationClustering
import org.apache.spark.mllib.clustering.PowerIterationClusteringModel

object Poweriterationclustering {

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
    val data = sc.textFile("E:\\Ñ¸À×ÏÂÔØ\\spark-1.5.2-bin-hadoop2.6\\spark-1.5.2-bin-hadoop2.6\\data\\mllib\\pic_data.txt")
    val similarities = data.map { line =>
      val parts = line.split(' ')
      (parts(0).toLong, parts(1).toLong, parts(2).toDouble)
    }

    // Cluster the data into two classes using PowerIterationClustering
    val pic = new PowerIterationClustering()
      .setK(2)
      .setMaxIterations(10)
    val model = pic.run(similarities)
    model.assignments.collect.foreach(println)
    model.assignments.foreach { a =>
      println(s"${a.id} -> ${a.cluster}")
    }

  }

}