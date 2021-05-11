
import org.apache.spark._
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor, GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._

import scala.io._

object Anjali {

  def main(args: Array[String]) {

    // Time starting -
    val startTimeMillis = System.currentTimeMillis()

    val dataPath = "C:\\Users\\Anjali\\Downloads\\try2\\Anjali.csv"
    //    val dataPath = "C:\\Users\\Anjali\\Desktop\\pycsv\\flights.csv"
    val dataPath1 = "C:\\Users\\Anjali\\Downloads\\644175800_T_ONTIME_REPORTING\\644175800_T_ONTIME_REPORTING.csv"
    val dataPath2 = "C:\\Users\\Anjali\\Downloads\\644150511_T_ONTIME_REPORTING\\644150511_T_ONTIME_REPORTING.csv"
    val dataPath3 = "C:\\Users\\Anjali\\Desktop\\pycsv\\644161247_T_ONTIME_REPORTING.csv"
    val dataPath4 = "C:\\Users\\Anjali\\Desktop\\pycsv\\644147495_T_ONTIME_REPORTING.csv"
    val dataPath5 = "C:\\Users\\Anjali\\Desktop\\pycsv\\207233246_T_ONTIME_REPORTING.csv"
    val dataPath6 = "C:\\Users\\Anjali\\Desktop\\pycsv\\207233247_T_ONTIME_REPORTING.csv"
    val dataPath7 = "C:\\Users\\Anjali\\Desktop\\pycsv\\207233248_T_ONTIME_REPORTING.csv"
    val dataPath8 = "C:\\Users\\Anjali\\Desktop\\pycsv\\207233249_T_ONTIME_REPORTING.csv"
    val dataPath9 = "C:\\Users\\Anjali\\Desktop\\pycsv\\420961779_T_ONTIME_REPORTING.csv"

    var mlTechnique: Int = 0

    // untill and unless one of the following techniques are chosen correctly these options will continue coming
    while(mlTechnique != 1 && mlTechnique != 2 && mlTechnique != 3){
      print("\n")
      print("Which machine learning technique do you want to use? \n")
      print("[1] Linear Regression \n")
      print("[2] Random Forest Trees \n")
      mlTechnique = scala.io.StdIn.readInt()
    }

    print("\n")
    // print("Use categorical features? (yes/no) \n")
    val useCategorical = true //scala.io.StdIn.readBoolean()

    println("checkpoint-1 - core here")
    val conf = new SparkConf().setAppName("predictor").setMaster("local[*]") //--------------------change here
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val sqlContext = new SQLContext(sc)
    // using dataframes , not dataset

    println("checkpoint-2")

    //for Anjali.csv -- also works with the combined datasets
    val rawData = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(dataPath)   //.load(dataPath,dataPath1,dataPath2,dataPath3,dataPath4,dataPath5,dataPath6,dataPath7,dataPath8,dataPath9)   //-------- change here for combined & single file loading
      .withColumn("DelayOutputVar", col("ARR_DELAY").cast("double"))
      .withColumn("DepDelayDouble", col("DEP_DELAY").cast("double"))
      .withColumn("TaxiOutDouble", col("TAXI_OUT").cast("double"))
      .cache()

    //for flights.csv
    //    val rawData = sqlContext.read.format("com.databricks.spark.csv")
    //      .option("header", "true")
    //      .option("inferSchema", "true")
    //      .load(dataPath)
    //      .withColumn("DelayOutputVar", col("ARRIVAL_DELAY").cast("double"))
    //      .withColumn("DepDelayDouble", col("DEPARTURE_DELAY").cast("double"))
    //      .withColumn("TaxiOutDouble", col("TAXI_OUT").cast("double"))
    //      .cache()



    // printing data contained by rawData -
    //    println(" printing the rawData contents be like - \n")
    //    rawData.show()


    //println("\n"+ getNumPartitions(rawData))
    println("\n size - "+ rawData.rdd.partitions.size )
    //println(" partition individual size \n")
    //println("Partitions structure: {}".format(rawData.glom().collect()))


    println("checkpoint-3 , followed by printing rawData - double casted 3 columns \n")
    println(rawData)

    // dropping non-useful columns from the rawData. ----- flights.csv - 520mb
    //    val data2 = rawData
    //      .drop("WHEELS_ON")//
    //      .drop("CANCELLED") //
    //      .drop("AIR_SYSTEM_DELAY")//
    //      .drop("SECURITY_DELAY") //
    //      .drop("AIRLINE_DELAY") //
    //      .drop("TAXI_IN") //
    //      .drop("DIVERTED") //
    //      .drop("WEATHER_DELAY") //
    //      .drop("LATE_AIRCRAFT_DELAY") //
    //      .drop("DEPARTURE_DELAY") // Casted to double in a new variable -DepDelayDouble
    //      .drop("TAXI_OUT") // Casted to double in a new variable -TaxiOutDouble
    //      .drop("DISTANCE") //
    //      .drop("FLIGHT_NUMBER")//
    //      .drop("YEAR") //
    //      .drop("MONTH")//
    //      .drop("DAY") //
    //      .drop("DAY_OF_WEEK") //
    //      .drop("TAIL_NUMBER") //
    //      .drop("AIRLINE")//
    //      .drop("SCHEDULED_DEPARTURE")//
    //      .drop("DEPARTURE_TIME")//
    //      .drop("WHEELS_OFF")//
    //      .drop("SCHEDULED_TIME")//
    //      .drop("ELAPSED_TIME")//
    //      .drop("AIR_TIME")//
    //      .drop("SCHEDULED_ARRIVAL")//
    //      .drop("ARRIVAL_TIME")//
    //      .drop("ARRIVAL_DELAY")//
    //      .drop("CANCELLATION_REASON")//




    // ------------- Anjali.csv - 57mb
    val data2 = rawData
      .drop("ACTUAL_ELAPSED_TIME")
      .drop("ARR_TIME")
      .drop("AIR_TIME")
      .drop("TAXI_IN")
      .drop("DIVERTED")
      .drop("CARRIER_DELAY")
      .drop("WEATHER_DELAY")
      .drop("NAS_DELAY")
      .drop("SECURITY_DELAY")
      .drop("LATE_AIRCRAFT_DELAY")
      .drop("DEP_DELAY") // Casted to double in a new variable -DepDelayDouble
      .drop("TAXI_OUT") // Casted to double in a new variable -TaxiOutDouble
      .drop("UniqueCarrier")
      .drop("CANCELLATION_CODE")
      .drop("DEP_TIME")
      .drop("CRS_ARR_TIME")
      .drop("CRS_ELAPSED_TIME")
      .drop("DISTANCE")
      .drop("FlightNum")
      .drop("CRS_DEP_TIME")
      .drop("YEAR")
      .drop("MONTH")
      .drop("DAY_OF_MONTH")
      .drop("DAY_OF_WEEK")
      .drop("TAIL_NUM")

    println("checkpoint-4, printing data after dropping \n")
    println(data2)

    val data = data2.filter("DelayOutputVar is not null")

    val assembler = if(useCategorical){
      new VectorAssembler()
        .setInputCols(Array("ORIGINVec", "DESTVec", "DepDelayDouble", "TaxiOutDouble"))
        .setOutputCol("features")
      //        .setHandleInvalid("skip")
    }else{
      new VectorAssembler()
        .setInputCols(Array("DepDelayDouble", "TaxiOutDouble"))
        .setOutputCol("features")
      //        .setHandleInvalid("skip")

    }

    val categoricalVariables = if(useCategorical){
      Array("ORIGIN", "DEST")   //----------------------------------------- made changes in csv a/c
    }else{
      null
    }

    val categoricalIndexers = if(useCategorical){
      categoricalVariables.map(i => new StringIndexer().setInputCol(i).setOutputCol(i+"Index").setHandleInvalid("skip"))
    }else{
      null
    }
    val categoricalEncoders = if(useCategorical){
      categoricalVariables.map(e => new OneHotEncoder().setInputCol(e + "Index").setOutputCol(e + "Vec").setDropLast(false))
    }else{
      null
    }



    mlTechnique match {
      case 1 =>
        val lr = new LinearRegression()
          .setLabelCol("DelayOutputVar")
          .setFeaturesCol("features")
        val paramGrid = new ParamGridBuilder()
          .addGrid(lr.regParam, Array(0.1, 0.01))
          .addGrid(lr.fitIntercept)
          .addGrid(lr.elasticNetParam, Array(0.0, 1.0))
          .build()

        val steps:Array[org.apache.spark.ml.PipelineStage] = if(useCategorical){
          categoricalIndexers ++ categoricalEncoders ++ Array(assembler, lr)
        }else{
          Array(assembler, lr)
        }

        val pipeline = new Pipeline().setStages(steps)

        val tvs = new TrainValidationSplit()
          .setEstimator(pipeline)
          .setEvaluator(new RegressionEvaluator().setLabelCol("DelayOutputVar"))
          .setEstimatorParamMaps(paramGrid)
          .setTrainRatio(0.7)

        val Array(training, test) = data.randomSplit(Array(0.70, 0.30), seed = 12345)

        val model = tvs.fit(training)

        val holdout = model.transform(test).select("prediction", "DelayOutputVar")

        println("checkpoint-5 - printing the predicted and actual delay for 1st 50 testing datasets \n")
        println(holdout.select("prediction","DelayOutputVar").show(50))
        val rm = new RegressionMetrics(holdout.rdd.map(x =>
          (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))


        println("RMSE: " + Math.sqrt(rm.meanSquaredError))

      case 2 =>
        val rf = new RandomForestRegressor()
          .setNumTrees(10)
          .setMaxDepth(10)
          .setLabelCol("DelayOutputVar")
          .setFeaturesCol("features")


        val steps:Array[org.apache.spark.ml.PipelineStage] = if(useCategorical){
          categoricalIndexers ++ categoricalEncoders ++ Array(assembler, rf)
        }else{
          Array(assembler, rf)
        }

        val pipeline = new Pipeline().setStages(steps)

        val Array(training, test) = data.randomSplit(Array(0.70, 0.30), seed = 12345)

        val model = pipeline.fit(training)

        val holdout = model.transform(test).select("prediction", "DelayOutputVar")

        val rm = new RegressionMetrics(holdout.rdd.map(x =>
          (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

        println("RMSE: " + Math.sqrt(rm.meanSquaredError))


    }

    val endTimeMillis = System.currentTimeMillis()
    val durationSeconds = (endTimeMillis - startTimeMillis) / 1000
    println("\n Total Time taken by the program (secs) -" + durationSeconds)
    sc.stop() //--------------- commented it for port to be active - no effect
  }

}
