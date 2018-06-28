特征工程涉及的主要流程：
1. 基于原始特征进行一些衍生特征的提取 featureExtraction下的相关代码
2. 先进行woe encoding编码，代码为featureEncoding下面的woe_encoding
3. 在woe 编码后的基础上，对特征进行单变量和多变量分析,featureSelection下相关性分析代码
4. 在单变量和多变量分析之后，基于上一步过滤后剩余的特征，进行模型效果显著性分析，特征重要性排序等等操作，在featureSelection目录下
5. 最终得到的有效特征 用于后续的模型训练
6. predict_data_generator: 在生成预测模型文件后，predict_data_generator用来针对原始预测数据进行一系列的特征工程转换，
    生成featuresInModel文件中包含的模型特征的实际数据，并用于模型预测

几个plk文件所对应的特征数据情况描述：
bin_dict: 保存每个衍生特征所对于的分箱点
featuresInModel: 用于模型训练的最终WOE编码后的特征名称，每个特征名称以WOE结尾
numericalFeatures: 连续性的衍生特征名称 不以WOE结尾
categoricalFeatures: 类别性的衍生特征名称 不以WOE结尾