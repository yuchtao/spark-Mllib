package cn.tipdm.spark

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.ChiSqSelector

object Featureselectiontest {

  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    var conf = new SparkConf()
    conf.setMaster("local[3]")
      .setAppName("analysis")
      .set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)
    // Load some data in libsvm format
    val data = MLUtils.loadLibSVMFile(sc, "E:\\Ñ¸À×ÏÂÔØ\\spark-1.5.2-bin-hadoop2.6\\spark-1.5.2-bin-hadoop2.6\\data\\mllib\\sample_libsvm_data.txt")
    // Discretize data in 16 equal bins since ChiSqSelector requires categorical features
    // Even though features are doubles, the ChiSqSelector treats each unique value as a category
    val discretizedData = data.map { lp =>
      LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.map { x => (x / 16).floor }))
    }
    // Create ChiSqSelector that will select top 50 of 692 features
    val selector = new ChiSqSelector(50)
    // Create ChiSqSelector model (selecting features)
    val transformer = selector.fit(discretizedData)
    // Filter the top 50 features from each feature vector
    val filteredData = discretizedData.map { lp =>
      LabeledPoint(lp.label, transformer.transform(lp.features))
    }

  }

}