package cn.tipdm.spark

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.StandardScalerModel


object StandardScalerTest {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc, "E:\\Ñ¸À×ÏÂÔØ\\spark-1.5.2-bin-hadoop2.6\\spark-1.5.2-bin-hadoop2.6\\data\\mllib\\sample_libsvm_data.txt")

    val scaler1 = new StandardScaler().fit(data.map(x => x.features))
    val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(data.map(x => x.features))
    // scaler3 is an identical model to scaler2, and will produce identical transformations
    val scaler3 = new StandardScalerModel(scaler2.std, scaler2.mean)

    // data1 will be unit variance.
    val data1 = data.map(x => (x.label, scaler1.transform(x.features)))

    // Without converting the features into dense vectors, transformation with zero mean will raise
    // exception on sparse vector.
    // data2 will be unit variance and zero mean.
    val data2 = data.map(x => (x.label, scaler2.transform(Vectors.dense(x.features.toArray))))
  }

}