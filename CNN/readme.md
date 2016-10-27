
# Batch Normalization

## reference
1. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
2. [implementing-batch-normalization-in-tensorflow](http://r2rt.com/implementing-batch-normalization-in-tensorflow.html)
3. [official batch normalization](https://github.com/tensorflow/tensorflow/blob/b826b79718e3e93148c3545e7aa3f90891744cc0/tensorflow/contrib/layers/python/layers/layers.py#L100)

## formula

       BN(x) = γ(x-μ)/σ +β
       
            x = [batch,height,width,depth] or [batch,input_size]
            μ = μ_B , batch mean
            σ = √(σ_B?+ε) , batch variance
            γ , scale
            β , offset
            
## example

use in CNN
```python

    cnn_output=[batch_size,height,width,depth]
    norm_output=batch_norm(cnn_output)

    cnn_output=[batch_size,total_num_filters]
    norm_output=batch_norm(tf.expand_dims(tf.expand_dims(cnn_output, 1), 1)
    cnn_output=tf.squeeze(norm_output)
  
```

offical batch normalization
```python

    # ordinary batch norm , input = [batch,size]
    input =  official_batch_norm_layer(input,size,is_training,False,scope="ordinary_batch_norm")   
    # cnn_output = [batch,height,width,channels]
    cnn_output = official_batch_norm_layer(cnn_output,channels,is_training,True,scope="cnn_batch_norm")
    # Bilstm_output = [batch,num_hidden*2]
    Bilstm_output = official_batch_norm_layer(Bilstm_output,num_hidden*2,is_training,False,scope="bilstm_batch_norm")
    
```

## note

1. apply batch normalization to the activation σ(Wx+b) would result in σ(BN(Wx+b)) , BN is the batch normalizing transform.
2. train and test, the (mean,variance) used by BN is different.
3. offical batch normalization is more effective than others.
4. I modified the official batch normalization to allow reception of variable batch size .
 
