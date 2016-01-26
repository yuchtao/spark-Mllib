package cn.tipdm.spark
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.SparkContext
import org.apache.log4j.Logger
import org.apache.spark.mllib.linalg.Matrix
import org.apache.log4j.Level
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.regression.LabeledPoint

object Principalcomponentanalysis {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    //.setSparkHome("/opt/spark-1.3.1-bin-hadoop2.6")

    val sc = new SparkContext(conf)
   /* val vectorss: RDD[Vector] = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.dense(4.0, 5.0, 6.0),
      Vectors.dense(7.0, 8.0, 9.0)))
    val mat: RowMatrix = new RowMatrix(vectorss)
    

    // Compute the top 10 principal components.
    val pc: Matrix = mat.computePrincipalComponents(2) // Principal components are stored in a local dense matrix.

    // Project the rows to the linear space spanned by the top 10 principal components.
    val projected: RowMatrix = mat.multiply(pc)
    mat.rows.collect.foreach(println)
    println("==========")
    println(pc)
    println("==========")
    projected.rows.collect.foreach(println)*/
    
    /*val v1: RDD[Vector] = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0),
      Vectors.dense(1.0, 2.0)))
    val v2: RDD[Vector] = sc.parallelize(Seq(aaaa
      Vectors.dense(3.0, 4.0),
      Vectors.dense(3.0, 4.0)))*/
    val pos1 = LabeledPoint(1.0, Vectors.dense(1.0, 2.0, 3.0))
    val pos2 = LabeledPoint(2.0, Vectors.dense(4.0, 5.0, 6.0))
    val pos3 = LabeledPoint(3.0, Vectors.dense(7.0, 8.0, 9.0))
    val lps = sc.parallelize(List(pos1,pos2,pos3))
    lps.collect.foreach(println)
    println("===================")
    // Index documents with unique IDs
    val pca = new PCA(1).fit(lps.map(_.features))
    val projected = lps.map(p => p.copy(features = pca.transform(p.features)))
    println(pca)
    
    projected.collect.foreach(println)
    println(pca)
  }

}