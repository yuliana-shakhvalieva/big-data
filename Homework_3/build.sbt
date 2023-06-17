name := "manual-lr-sparkml"

version := "0.1.0-SNAPSHOT"

scalaVersion := "2.13.11"

val sparkVersion = "3.3.2"
val breezeVersion = "1.3"
val scalaTestVersion = "3.2.2"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion withSources(),
  "org.apache.spark" %% "spark-mllib" % sparkVersion withSources(),
  "org.scalanlp" %% "breeze" % breezeVersion withSources(),
  "org.scalanlp" %% "breeze-natives" % breezeVersion withSources()
)

libraryDependencies += ("org.scalatest" %% "scalatest" % scalaTestVersion % "test" withSources())
