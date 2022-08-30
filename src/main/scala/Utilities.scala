import org.apache.spark.sql.{DataFrame, SparkSession}

import java.nio.file.Paths
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

object Utilities {
  def getSpark(appName:String, partitionCount:Int = 1,
               master:String = "local",
               verbose:Boolean = true): SparkSession = {
    if (verbose) println(s"$appName with Scala version: ${util.Properties.versionNumberString}")
    val sparkSession = SparkSession.builder().appName(appName).master(master).getOrCreate()
    sparkSession.conf.set("spark.sql.shuffle.partitions", partitionCount)
    if (verbose) println(s"Session started on Spark version ${sparkSession.version} with ${partitionCount} partitions")
    sparkSession
  }

  def readDataWithView(spark:SparkSession,
                       filePath:String,
                       source:String = "csv",
                       viewName:String = "dfTable",
                       header:Boolean = true,
                       inferSchema:Boolean= true,
                       printSchema:Boolean = true,
                       cacheOn: Boolean = true
                      ) :DataFrame = {

    val df = spark.read.format(source)
      .option("header", header.toString) //Spark wants string here since option is generic
      .option("inferSchema", inferSchema.toString) //we let Spark determine schema
      .load(filePath)
    //so if you pass only whitespace or nothing to view we will not create it
    //so if viewName is NOT blank
    if (viewName.nonEmpty) {
      df.createOrReplaceTempView(viewName)
      println(s"Created Temporary View for SQL queries called: $viewName")
    }
    if (printSchema) df.printSchema()
    if (cacheOn) df.cache()
    df
  }

  def SaveDataframeToParquet(df: DataFrame, filename: String, filepath: String = "src/resources/parquet"): Unit = {
    val fullpath = Paths.get(filepath, filename + "-" + GetCurrentTimestamp() + ".parquet").toString

    df
      .write
      .mode("overwrite")
      .option("header", true)
      .parquet(fullpath)
  }

  def SaveDataframeToCsv(df: DataFrame, filename: String, filepath: String = "src/resources/csv"): Unit = {
    val fullpath = Paths.get(filepath, filename + "-" + GetCurrentTimestamp() + ".csv").toString

    df
      .write
      .mode("overwrite")
      .option("header", true)
      .csv(fullpath)
  }

  def GetCurrentTimestamp(format: String = "yyyy-MM-dd_HH_mm"): String = {
    DateTimeFormatter.ofPattern(format).format(LocalDateTime.now())
  }
}
