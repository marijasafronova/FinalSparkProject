import Utilities.{SaveDataframeToCsv, SaveDataframeToParquet}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object Final {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR) //app will show errors, hiding other messages (like INFO etc.)

    val spark = Utilities.getSpark("Final")

    val filePath = if (args.isEmpty) "src/resources/stock_prices_.csv" else args(0)

    val stockPrices = Utilities.readDataWithView(spark, filePath).na.drop()

    // Getting date only in our format (yyyy-MM-dd), else null
    // Then removing any null -> incorrect date format in csv
    val dfDailyAvg = stockPrices
      .selectExpr(
        "date(to_date(date, 'yyyy-MM-dd')) as format_date",
        "(close - open)/open * 100 as daily_return")
      .groupBy(col("format_date"))
      .agg(round(avg("daily_return"), 2).as("daily_avg"))
      .na
      .drop()

    dfDailyAvg.show(10)

    SaveDataframeToParquet(dfDailyAvg, "daily_avg")
    SaveDataframeToCsv(dfDailyAvg, "daily_avg")

    // Casting value to decimal in order to avoid scientific notation (numberEpower)
    val dfMostTraded = stockPrices
      .selectExpr(
        "ticker",
        "cast((close * volume) as decimal(20,2)) as most_frequent" ///if deleting 20,2 another number??
      )
      .groupBy(col("ticker"))
      .agg(round(avg("most_frequent"), 2).as("most_frequent_avg"))
      .orderBy(desc("most_frequent_avg"))
      .limit(1)
      .show()

    // ********************* BONUS ************************

    // Calculating Annualized Standard Deviation = Standard Deviation of Daily Returns * sqrt(252)
    // 252 represents the typical number of trading days in a year.
    val dfMostVolatile = stockPrices
      .selectExpr(
        "ticker",
        "(close - open)/open * 100 as daily_return"
      )
      .groupBy("ticker")
      .agg(stddev("daily_return").as("volatility"))
      .withColumn("volatility", round(col("volatility") * sqrt(lit(252)), 2))
      .orderBy(desc("volatility"))
      .limit(1)
      .show()
    // ***************************************************

    //******************* BIG BONUS **********************
    // Creating dataframe for model
    val df = spark.read
      .format("csv")
      .option("header", true)
      .option("inferSchema", true)
      .load(filePath)
      .where("ticker == 'AAPL'")
      .toDF()

    // Creating dataframes for training and testing using ranks
    // ranks -> dividing dataframe rows into 100 percent
    val rankedDf = df
      .withColumn("rank", percent_rank().over(Window.partitionBy("ticker").orderBy("date")))
    val train = rankedDf.where("rank <= 0.8").drop("rank")
    val test = rankedDf.where("rank > 0.8").drop("rank")

    // Creating basic variables
    val rForm = new RFormula()
    val lr = new LinearRegression()
      .setLabelCol("close")
      .setFeaturesCol("features")

    // Creating pipelines
    val stages = Array(rForm, lr)
    val pipeline = new Pipeline().setStages(stages)

    // Training and evaluation
    val params = new ParamGridBuilder()
      .addGrid(rForm.formula, Array(
        "close ~ open + volume",
        "close ~ open + high + low",
        "close ~ open + high + low + volume"))
      .addGrid(lr.regParam, Array(0, 0.01, 0.1, 1.0))
      .build()

    // Specifying evaluation process
    val evaluator = new RegressionEvaluator()
      .setPredictionCol("prediction")
      .setLabelCol("label")

    // Fitting hyperparameters on a validation set in order to preventing overfitting
    val tvs = new TrainValidationSplit()
      .setTrainRatio(0.80)
      .setEstimatorParamMaps(params)
      .setEstimator(pipeline)
      .setEvaluator(evaluator)

    // Fitting the model
    val model = tvs.fit(train)
    val predictionModel = model.transform(test)

    predictionModel
      .select("date", "close", "prediction")
          .withColumn("prediction", expr("round(prediction, 2)"))
          .show(50)

    model.write.overwrite().save("src/resources/models/linear_regression")
    // ***************************************************
  }
}
