package cn.tipdm.spark

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.ElementwiseProduct
import org.apache.spark.mllib.linalg.Vectors

object ElementwiseProductTest {

  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)
    // Create some vector data; also works for sparse vectors
    val data = sc.parallelize(Array(Vectors.dense(1.0, 2.0, 3.0), Vectors.dense(4.0, 5.0, 6.0)))

    val transformingVector = Vectors.dense(0.0, 1.0, 2.0)
    val transformer = new ElementwiseProduct(transformingVector)
    // Batch transform and per-row transform give the same results:
    val transformedData = transformer.transform(data)
    val transformedData2 = data.map(x => transformer.transform(x))
    data.collect.foreach(println)
    
    /*transformedData.collect.foreach(println)
    
    transformedData2.collect.foreach(println)*/
  }

}