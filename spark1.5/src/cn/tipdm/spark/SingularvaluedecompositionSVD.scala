package cn.tipdm.spark

import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.SparkContext
import org.apache.log4j.Logger
import org.apache.spark.mllib.linalg.Matrix
import org.apache.log4j.Level
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import breeze.linalg.svd
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector

object SingularvaluedecompositionSVD {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    //.setSparkHome("/opt/spark-1.3.1-bin-hadoop2.6")

    val sc = new SparkContext(conf)
    /*val vectorss: RDD[Vector] = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.dense(4.0, 5.0, 6.0),
      Vectors.dense(7.0, 8.0, 9.0)))*/
      val vectorss: RDD[Vector] = sc.parallelize(Seq(
      Vectors.dense(2.0, 2.0),
      Vectors.dense(2.0, 3.0)))
    val mat: RowMatrix = new RowMatrix(vectorss)

    // Compute the top 20 singular values and corresponding singular vectors.
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(2, computeU = true)
    val U: RowMatrix = svd.U // The U factor is a RowMatrix.
    val s: Vector = svd.s // The singular values are stored in a local dense vector.
    val V: Matrix = svd.V // The V factor is a local dense matrix.
    U.rows.collect.foreach(println)
    println("U====="+U)
    println("s====="+s)
    println("V====="+V)
  }

}