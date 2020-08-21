# hll_xgb

测试重点：
1. 特征ks，psi能否运行
2. checkpoint只是参数设置就可以了，如果checkpoint_init=false，那么就pyspark就可以接着上次的继续训练。如果是第一次训练，设置init=true
3. 特征处理模块是否有正确处理特征



