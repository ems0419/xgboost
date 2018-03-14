# 参数说明
http://xgboost.readthedocs.io/en/latest/parameter.html?highlight=seed

此页面是以R为例，Python一样。本内容为个人的理解

参数主要有三大类：通用参数、基模型参数（tree booster、Linear Booster）、任务参数。

### 通用参数：
 
 #### -booster：设置集学习器。常用的就是gbtree（默认值）
 
 #### -silent：设置在学习（running）的过程中是否需要输出信息，需要的话，设置1，不需要的时候silent=0(default)
 
 #### -nthread：用多少个线程，默认情况下是全用
 
 #### -num_pbuffer：很少用到，有需要的时候再去研究这个参数是干嘛的。用户不需要设置
 
 #### -num_feature：很少用到，有需要的时候再去研究这个参数是干嘛的。用户不需要设置
 
### 基模型参数：当基模型为树模型时，设置tree booster参数

 #### -eta：
 
  即learning rate,默认值是0.3，范围是[0,1]，通常会通过参数字典进行调参。不同的learning rate收敛的速度是有差别的。
 
 ####  -gamma：
 
  对树的叶子节点做进一步分割的时候，设置的损失减少的最小值。如果这个值越大，这个算法会越保守（即不会激进的做下一步的分裂）。
 
  这个指标在一定程度上可以控制过拟合，因为它在一定程度上会控制这个算法有多保守。
 
  默认值是0，范围是[0,∞]
         
 ####  -max_depth：

  树的深度，控制过拟合

  默认值是6，范围是[0，∞]
             
 ####  -min_child_weight：

  最小的孩子的权重。默认值是1，范围是[0，∞]
 
 #### -max_delta_step：

  用的不多，有需要的时候再去研究这个参数是干嘛的。
 
 ####  -subsample：

  进行训练的数据比例，通过有放回地选取部分的样本，进行训练。在RandomForest中也涉及到此参数。

  行采样。当取0.5时，意味着XGBoost随机选取一半的数据来构建树，然后形成组合。

  默认值是1，范围是[0,1]

  tips:串行的模型容易过拟合。
             
 ####  -colsample_bytree：

  列采样，在构建树的过程中，每棵树划分的特征比重。

  默认值是1，范围是[0,1]
                    
 ####   -colsample_bylevel:

  列采样，在构建树的过程中，在每一层划分的特征比重。

  默认值是1，范围是[0,1]
                    
####  -lambda：

正则化项1的参数，表示正则化的强度，此值越大，这个算法会越保守
  
####  -alpha：

正则化项2的参数，表示正则化的强度，此值越大，这个算法会越保守
  
 ####  -scale_pos_weigth：
 
 在样本不均衡时会用到，是控制样本权重的参数。取值通常是负样本总数/正样本总数。
 
    #数据预处理，设置函数，来计算scale_pos_weight的值
    
    def preproc(dtrain, dtest, param):
        labels = dtrain.get_label()
        ratio = float(np.sum(labels==0))/np.sum(labels==1)    #负样本总数/正样本总数
        param['scale_pos_weigth'] = ratio                     #将计算出来的ratio放到param字典中
        return (dtrain, dtest, param)

    xgb.cv(param, dtrain, num_round, nfold=5, metrics={'auc'}, seed=3, fpreproc=preproc)
 
 ####  -其他参数：用的不多，需要的时候详细看
         
 
 ### 基模型参数：当基模型为linear时，设置linear booster参数
  
 ####  -lambda：
 ####  -alpha：
 ####  -lambda_bias：
 
 ### 任务参数
 
 #### -objective
 
 可以看做是损失函数、代价函数、目标函数的设定
 
 #### -base_score：通常不会去调整它
 
 #### eval_metric:
 
 评估标准
 
 #### -其他参数：用的不多，需要的时候详细看
 
 
 


# 调参指南（官方）
http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html#handle-imbalanced-dataset

调参过程，其实就是欠拟合和过拟合的程度的平衡。采用boosting模式时，很少会出现欠拟合（underfitting），除非是参数特别特别少。所以大部分情况下，更有可能会进入过拟合（overfitting）陷阱。
 
## 控制过拟合

主要有两种方式：

### 直接控制模型的复杂度
#### -max_depth
#### -min_child_weight
#### -gamma
### 从样本/数据的角度来增加鲁棒性
#### -subsample
#### -colsample
#### -减小eta，此时需要同时增加num_round


## 控制不均衡数据

#### 当只关心预测的排序（AUC）时，

 - 平衡正负样本权重，通过scale_pos_weight

 - 用AUC进行评价

#### 关心预测的正确的概率时，

 - 不能平衡正负样本比例

 - 通过max_delta_step来帮助收敛
 
 
# 自定义目标函数
    #自定义目标函数(log似然),需要提供一阶和二阶导数
    
    def logregobj(pred,dtrain):
		labels = dtrain.get_label()
		pred =1.0 / (1+np.exp(-pred))
		grad = pred - labels
		hess = pred *( 1 - pred)
		return grad, hell
	
	def evalerror(pred, dtrain):
    labels = dtrain.get_label()
    return 'error', float(sum(labels != (pred>0.0)))/len(labels)

	param = {'max_depth':2, 'eta':1, 'silent':1}

	#自定义目标函数训练
	model = xgb.train(param, dtrain, num_round, watch_list, logregobj, evalerror)
 
 
 
