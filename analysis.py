#! /usr/bin/env python2 
# -*- coding: utf-8 -*-


######################### 地震数据分析 —— 基于pyspark #########################

####################### 数据预处理 Data preprocessing #######################

## 读取并查看数据
import pyspark
import pyspark.sql.types as typ

# 设定文件路径
FilePath = "earthquake/earthquake.csv"

# 读取数据
data = spark.read.csv(FilePath, header='true', inferSchema='true', sep=',')

# 将数据上传到cache中 
data.cache()

# 查看总样本量，并判断是否有重复的样本
print 'Count of rows: {0}'.format(data.count())
print 'Count of distinct rows: {0}'.format(data.distinct().count())

# 无重复样本，但数据量过大，故进行抽样
# 随机抽取1%的数据，无放回抽样，设定随机数种子42
data_sample = data.sample(False, 0.01, 42) 
print 'Count of rows: {0}'.format(data_sample.count())

# 得到6053行的样本，查看每行缺失值
data_sample.rdd.map(
    lambda row: (row['id'], sum([c == None for c in row]))
).collect()

# 发现只有id为601277的样本存在缺失值，查看该样本
data_sample.where('id == 601277').show()

# 该样本除了district_id之外均为空值，可以直接删去，剩余6052行
data_sample = data_sample.dropna()
print 'Count of rows: {0}'.format(data_sample.count())

# 查看数据的schema并保存数据
data_sample.printSchema()


####################### 创建转换器 Create a transformer #######################

## 对'land_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'position', 'y'六个变量进行转换
import pyspark.ml.feature as ft

# 对'land_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'position'五个多分类变量
# 先使用StringIndexer转换数据类型，再使用OneHotEncoderEstimator进行One Hot编码
indexer1 = ft.StringIndexer(inputCol="land_condition", outputCol="land_condition_index")
data_sample = indexer1.fit(data_sample).transform(data_sample)
indexer2 = ft.StringIndexer(inputCol="foundation_type", outputCol="foundation_type_index")
data_sample = indexer2.fit(data_sample).transform(data_sample)
indexer3 = ft.StringIndexer(inputCol="roof_type", outputCol="roof_type_index")
data_sample = indexer3.fit(data_sample).transform(data_sample)
indexer4 = ft.StringIndexer(inputCol="ground_floor_type", outputCol="ground_floor_type_index")
data_sample = indexer4.fit(data_sample).transform(data_sample)
indexer5 = ft.StringIndexer(inputCol="position", outputCol="position_index")
data_sample = indexer5.fit(data_sample).transform(data_sample)

encoder = ft.OneHotEncoderEstimator( \
            inputCols=['land_condition_index', 'foundation_type_index', 'roof_type_index', 'ground_floor_type_index', 'position_index'], \
            outputCols=['land_condition_vec', 'foundation_type_vec', 'roof_type_vec', 'ground_floor_type_vec', 'position_vec'])
data_sample = encoder.fit(data_sample).transform(data_sample)

# 使用Binarizer将因变量y（4个分类）转换为二分类变量
# 其中0，1，2三类合并成一类记作0类，表示不需要重建，将因变量为3的类别记作1类，表示需要重建。
data_sample = data_sample.withColumn('y_double', data_sample['y'].cast(typ.DoubleType()))
binarizer = ft.Binarizer(threshold=2, inputCol="y_double", outputCol="label")
data_sample = binarizer.transform(data_sample)
data_sample.take(1)

# 使用VectorAssembler创建特征向量
featuresCreator = ft.VectorAssembler(
    inputCols=['floors_before', 'floors_after', 'age', 'area', 'height_before', 'height_after', \
               'land_condition_vec', 'foundation_type_vec', 'roof_type_vec', 'ground_floor_type_vec', 'position_vec'], 
    outputCol='features'
)

# 使用VectorIndexer自动识别分类变量，设定最大分类数为5
indexer = ft.VectorIndexer(inputCol="features", outputCol="indexed", maxCategories=5)

# 划分训练集和测试集
data_train, data_test = data_sample.randomSplit([0.8, 0.2], seed=42)


############################### 描述性统计分析 ###############################

data_sample.printSchema() # 查看数据Schema

data_sample.groupby('age').count().show() # 按建筑建成时间分组
data_sample.agg({'age': 'skewness'}).show()

numerical = ['floors_before', 'floors_after', 'height_before', 'height_after']
desc = data_sample.describe(numerical) # 查看地震前后的楼层数和高度变化
desc.show()


##########################  建立Logistic回归模型 ##########################

## 超参调优 Parameter hyper-tuning

### 创建评估器 Create an estimator

import pyspark.ml.classification as cl

logistic = cl.LogisticRegression(
    labelCol='label') # 对评估器的参数还需进一步进行超参调优，故先不设定超参数

### 网格搜索 Grid search

import pyspark.ml.tuning as tune

grid = tune.ParamGridBuilder() \
    .addGrid(logistic.maxIter,  
             [10, 50, 80]) \
    .addGrid(logistic.regParam, 
             [0.01, 0.001]) \
    .build()

### 创建管道 Create a pipeline

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[featuresCreator, indexer])
data_transformer = pipeline.fit(data_train)

## 模型拟合及性能评估 Fit the model & Model performance

# 使用BinaryClassificationEvaluator评估模型性能
import pyspark.ml.evaluation as ev

evaluator = ev.BinaryClassificationEvaluator(
    rawPredictionCol='probability', 
    labelCol='label')

# 进行5折交叉验证
cv = tune.CrossValidator(
    estimator=logistic, 
    estimatorParamMaps=grid, 
    evaluator=evaluator,
    numFolds=5
)

# 拟合模型，并在测试集上进行预测
cvModel = cv.fit(
    data_transformer \
        .transform(data_train)
)

prediction = cvModel.transform( \
    data_transformer \
        .transform(data_test))

results = prediction.select("id", "prediction", "probability", "label")

# 查看预测结果(前10行)
results.show(10)

lrTotalCorrect = results.rdd.map(lambda row : (1 if (row['prediction'] == row['label']) else 0)) \
                            .reduce(lambda x, y : x + y)
lrAccuracy = lrTotalCorrect / float(data_test.count())

# 显示最优模型的areaUnderROC、areaUnderPR和Accuracy
print "lrUnderROC: {0}" .format( \
        evaluator.evaluate(results, \
            {evaluator.metricName: 'areaUnderROC'}))  # 0.871900425015
print "lrUnderPR: {0}" .format( \
        evaluator.evaluate(results, \
            {evaluator.metricName: 'areaUnderPR'}))  # 0.930318105135
print "lrAccuracy: {0}" .format(lrAccuracy)  # 0.790186125212

# 显示最优参数组合
parameters = [
    (
        [
            {key.name: paramValue} 
            for key, paramValue 
            in zip(
                params.keys(), 
                params.values())
        ], metric
    ) 
    for params, metric 
    in zip(
        cvModel.getEstimatorParamMaps(), 
        cvModel.avgMetrics
    )
]

sorted(parameters, 
       key=lambda el: el[1], 
       reverse=True)[0]

# 最优参数组合为：([{'maxIter': 10}, {'regParam': 0.001}], 0.8735534629093946)

# 保存模型
from pyspark.ml import PipelineModel

lrmodelPath = 'earthquake/lrModel'
cvModel.write().overwrite().save(lrmodelPath)

'''
# 加载模型
loadedlrModel = PipelineModel.load(lrmodelPath)
prediction = loadedlrModel.transform( \
    data_transformer \
        .transform(data_test))

'''


######################## 建立RandomForestClassifier模型 ########################

### 创建评估器 Create an estimator
import pyspark.ml.classification as cl

RFclassifier = cl.RandomForestClassifier(
    labelCol='label')

### 网格搜索 Grid search

import pyspark.ml.tuning as tune

grid = tune.ParamGridBuilder() \
    .addGrid(RFclassifier.numTrees,  
             [20, 200]) \
    .addGrid(RFclassifier.maxDepth, 
             [5, 10]) \
    .build()

### 创建管道 Create a pipeline

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[featuresCreator])
data_transformer = pipeline.fit(data_train)


## 模型拟合及性能评估 Fit the model & Model performance

# 使用BinaryClassificationEvaluator评估模型性能
import pyspark.ml.evaluation as ev

evaluator = ev.BinaryClassificationEvaluator(
    rawPredictionCol='probability', 
    labelCol='label')

# 进行5折交叉验证
cv = tune.CrossValidator(
    estimator=RFclassifier, 
    estimatorParamMaps=grid, 
    evaluator=evaluator,
    numFolds=5
)

# 拟合模型，并在测试集上进行预测
cvModel = cv.fit(
    data_transformer \
        .transform(data_train)
)

prediction = cvModel.transform( \
    data_transformer \
        .transform(data_test))

results = prediction.select("id", "prediction", "probability", "label")

# 查看预测结果(前10行)
results.show(10)  

RFTotalCorrect = results.rdd.map(lambda row : (1 if (row['prediction'] == row['label']) else 0)) \
                            .reduce(lambda x, y : x + y)
RFAccuracy = RFTotalCorrect / float(data_test.count())

# 显示最优模型的areaUnderROC、areaUnderPR和Accuracy
print "RFUnderROC: {0}" .format( \
        evaluator.evaluate(results, \
            {evaluator.metricName: 'areaUnderROC'}))  # 逻辑回归：0.871900425015 ——> 随机森林：0.888740133576
print "RFUnderPR: {0}" .format( \
        evaluator.evaluate(results, \
            {evaluator.metricName: 'areaUnderPR'}))  # 逻辑回归：0.930318105135 ——> 随机森林：0.940623362605
print "RFAccuracy: {0}" .format(lrAccuracy)  # 逻辑回归：0.790186125212 ——> 随机森林：0.75296108291

# 显示最优参数组合
parameters = [
    (
        [
            {key.name: paramValue} 
            for key, paramValue 
            in zip(
                params.keys(), 
                params.values())
        ], metric
    ) 
    for params, metric 
    in zip(
        cvModel.getEstimatorParamMaps(), 
        cvModel.avgMetrics
    )
]

sorted(parameters, 
       key=lambda el: el[1], 
       reverse=True)[0]

# 最优参数组合为：([{'numTrees': 200}, {'maxDepth': 10}], 0.8807307840184255)

# 保存模型
from pyspark.ml import PipelineModel

RFmodelPath = 'earthquake/RFModel'
cvModel.write().overwrite().save(RFmodelPath)

'''
# 加载模型
loadedRFModel = PipelineModel.load(RFmodelPath)
prediction = loadedRFModel.transform( \
    data_transformer \
        .transform(data_test))

'''


###################### 使用TensorFlow构建一个简单的神经网络 ######################

import tensorflow as tf
from sparkflow.graph_utils import build_graph
from sparkflow.tensorflow_async import SparkAsyncDL
from pyspark.ml.pipeline import Pipeline

# 构建一个简单的神经网络
def small_model():
    x = tf.placeholder(tf.float32, shape=[None, 21], name='x') # 输入数据并设置占位符
    y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
    layer1 = tf.layers.dense(x, 10, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, 5, activation=tf.nn.relu)
    out = tf.layers.dense(layer2, 1, activation=tf.nn.sigmoid, name='out')
    loss = tf.losses.mean_squared_error(y, out) # 前面代表真实标签，后面代表神经网络输出结果
    return loss

mg = build_graph(small_model) # 构建计算图

spark_model = SparkAsyncDL(
    inputCol='features', # 输入列
    labelCol='label', # 输出列
    tensorflowGraph=mg,
    tfInput='x:0',
    tfLabel='y:0',
    tfOutput='out/Sigmoid:0',
    tfLearningRate=0.001, # 学习率设为0.001
    iters=50, # 训练50轮
    predictionCol='probability',
    miniBatchSize = 200, # 每批200个数据
    verbose=1,
    tfOptimizer='adam' # 优化器选择Adam
)

# 拟合模型
pipeline = Pipeline(stages=[featuresCreator])
data_transformer = pipeline.fit(data_train)

ANN_model = spark_model.fit(
        data_transformer \
        .transform(data_train)
)

# 模型预测
prediction = ANN_model.transform( \
    data_transformer \
        .transform(data_test)
)
results = prediction.select('id', 'label', 'probability')

from pyspark.sql.functions import col, when
prediction = when(
    col('probability') > 0.5, 1.0).otherwise(0.0)
results = results.withColumn('prediction', prediction)

# 查看预测结果(前10行)
results.show(10)                  

# 计算预测的精确度
TotalCorrect = results.rdd.map(lambda row : (1 if (row['prediction'] == row['label']) else 0)) \
                            .reduce(lambda x, y : x + y)
Accuracy = TotalCorrect / float(data_test.count())
print "ANN_Accuracy: {0}" .format(Accuracy)  # 0.785956006768

# 保存模型
from pyspark.ml import PipelineModel

ANNmodelPath = 'earthquake/ANNModel'
ANN_model.write().overwrite().saveANNmodelPath