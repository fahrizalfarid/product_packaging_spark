## Run

### python
```bash
python main.py
```

### spark-submit
```bash
$ ./bin/spark-submit ~/main.py
```


```python
import os
os.environ["JAVA_HOME"] = "/home/opt/java8"
os.environ["SPARK_HOME"] = "/home/opt/spark-3.3.1-bin-hadoop3"

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowthModel


class prediction:
    def __init__(self, modelpath: str):
        self.spark = SparkSession.builder\
            .master("local[*]")\
            .appName("product_packaging_spark")\
            .config("spark.ui.port", "4050")\
            .config("spark.driver.memory", "1g")\
            .getOrCreate()
        self.model = FPGrowthModel.load(modelpath)
        self.model.setItemsCol("nama_barang_array")

    def loadCsv(self, filepath: str):
        self.df = (
            self.spark.read.format("csv")
            .option("header", "true")
            .option("sep", "\t")
            .load(filepath)
        )

        self.df = self.df.withColumnRenamed(existing="Kode Transaksi", new="kode_transaksi")\
            .withColumnRenamed(existing="Nama Barang", new="nama_barang")
        self.df.createOrReplaceTempView("penjualan")
        self.df = self.spark.sql("""
                    select kode_transaksi, 
                        COLLECT_LIST(nama_barang) as nama_barang_array
                        from penjualan group by kode_transaksi
                    """)

    def doPrediction(self, minConfidence: float) -> pyspark.sql.dataframe.DataFrame:
        self.model.setMinConfidence(minConfidence)

        result = self.model.transform(self.df)
        result.createOrReplaceTempView("result")
        result = self.spark.sql(
            """select * from result 
                where prediction[0] is not null or 
                prediction[1] is not null""")
        
        print(result.show(5))
        return result

    def exportResult(self, df: pyspark.sql.dataframe.DataFrame):
        df.coalesce(1)\
            .write.mode("overwrite")\
            .json("./result_json")

        df.coalesce(1)\
            .write.mode("overwrite")\
            .parquet("./result_parquet")
        
        df.toPandas()\
            .to_json(orient="records", path_or_buf="./result_easy_json.json")
    
    def stopSpark(self):
        self.spark.sparkContext.stop()
        self.spark.stop()


if __name__ == "__main__":
    p = prediction("./model_s.2")
    p.loadCsv("./data_transaksi2.txt")
    p.exportResult(
        p.doPrediction(0.8)
    )
    p.stopSpark()
```

### Output
```bash
+--------------+--------------------+--------------+
|kode_transaksi|   nama_barang_array|    prediction|
+--------------+--------------------+--------------+
|          #131|[Celana Jeans Sob...|[Shampo Biasa]|
|          #203|[Celana Jeans Sob...|[Shampo Biasa]|
|          #210|[Celana Jeans Sob...|[Shampo Biasa]|
|          #225|[Celana Jeans Sob...|[Shampo Biasa]|
|          #231|[Celana Jeans Sob...|[Shampo Biasa]|
|          #233|[Celana Jeans Sob...|[Shampo Biasa]|
|          #265|[Celana Jeans Sob...|[Shampo Biasa]|
|          #273|[Celana Jeans Sob...|[Shampo Biasa]|
|          #330|[Celana Jeans Sob...|[Shampo Biasa]|
|           #38|[Celana Jeans Sob...|[Shampo Biasa]|
|          #394|[Celana Jeans Sob...|[Shampo Biasa]|
|          #421|[Celana Jeans Sob...|[Shampo Biasa]|
|          #424|[Celana Jeans Sob...|[Shampo Biasa]|
|          #445|[Celana Jeans Sob...|[Shampo Biasa]|
|          #502|[Celana Jeans Sob...|[Shampo Biasa]|
|          #523|[Celana Jeans Sob...|[Shampo Biasa]|
|          #525|[Celana Jeans Sob...|[Shampo Biasa]|
|          #599|[Celana Jeans Sob...|[Shampo Biasa]|
|          #604|[Celana Jeans Sob...|[Shampo Biasa]|
|          #643|[Celana Jeans Sob...|[Shampo Biasa]|
+--------------+--------------------+--------------+
only showing top 20 rows
```
