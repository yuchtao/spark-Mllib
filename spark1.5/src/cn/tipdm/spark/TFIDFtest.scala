package cn.tipdm.spark

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import akka.dispatch.Foreach
import org.apache.spark.mllib.feature.IDF

object TFIDFtest {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)
    val data = sc.textFile("E:\\迅雷下载\\spark-1.5.2-bin-hadoop2.6\\spark-1.5.2-bin-hadoop2.6\\data\\mllib\\tfidf.txt")
    val documents: RDD[Seq[String]] = data.map(_.split(" ").toSeq)
    documents.collect.foreach(println)
    println("==================")
    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(documents)
    tf.foreach(println)
    println("==================")

    tf.cache()
    val idf = new IDF().fit(tf)
    val test = idf.idf.toDense.toArray
    /*for(i <- 0 until test.length){
      println("第"+i+"个数组是"+test(i))
    }*/
    val tfidf: RDD[Vector] = idf.transform(tf)
    tfidf.collect.foreach(println)

    
  }

}