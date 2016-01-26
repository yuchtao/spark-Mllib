package cn.tipdm.spark

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.fpm.AssociationRules
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset

object AssociationRulesTest {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)
    val freqItemsets = sc.parallelize(Seq(
      new FreqItemset(Array("a"), 15L),
      new FreqItemset(Array("b"), 35L),
      new FreqItemset(Array("a", "b"), 12L)));
    
    val ar = new AssociationRules()
      .setMinConfidence(0.2)
    val results = ar.run(freqItemsets)

    results.collect().foreach { rule =>
      println("[" + rule.antecedent.mkString(",")
        + "=>"
        + rule.consequent.mkString(",") + "]," + rule.confidence)
    }

  }
}