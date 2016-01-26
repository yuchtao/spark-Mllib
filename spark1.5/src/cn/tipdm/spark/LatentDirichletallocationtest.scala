package cn.tipdm.spark

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.Vectors

object LatentDirichletallocationtest {

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
    val data = sc.textFile("E:\\Ñ¸À×ÏÂÔØ\\spark-1.5.2-bin-hadoop2.6\\spark-1.5.2-bin-hadoop2.6\\data\\mllib\\sample_lda_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble)))
    parsedData.collect.foreach(println)
    println("===================")
    // Index documents with unique IDs
    val corpus = parsedData.zipWithIndex.map(_.swap).cache()
    corpus.collect.foreach(println)
    // Cluster the documents into three topics using LDA
    val ldaModel = new LDA().setK(3).run(corpus)
    println("ldaModel.k==="+ldaModel.k)
    println("ldaModel.topicConcentration==="+ldaModel.topicConcentration)
    println("ldaModel.vocabSize==="+ldaModel.vocabSize)
    // Output topics. Each is a distribution over words (matching word count vectors)
    println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + " words):")
    val topics = ldaModel.topicsMatrix
   
    for (topic <- Range(0, 3)) {
      print("Topic " + topic + ":")
      for (word <- Range(0, ldaModel.vocabSize)) { print(" " + topics(word, topic)); }
      println()
    }
  }

}