ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.8"

lazy val root = (project in file("."))
  .settings(
    name := "FinalSparkProject"
  )

libraryDependencies += "org.apache.logging.log4j" % "log4j-core" % "2.17.2"

libraryDependencies += "org.apache.logging.log4j" % "log4j-api" % "2.17.2"

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.2.2"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.2.2"

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "3.2.2"

libraryDependencies += "org.xerial" % "sqlite-jdbc" % "3.39.2.0"
