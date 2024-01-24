from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, LSTM, Flatten, MultiHeadAttention, MaxPooling1D
from tensorflow.keras.layers import PReLU, SpatialDropout1D, Bidirectional, Multiply
from tensorflow.keras.layers import ZeroPadding1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import Model

input_shape_conv = (101, 4)
input_layer = Input(shape=input_shape_conv, name='Input_Layer')
padding_layer = ZeroPadding1D(padding=3, input_shape=(101,4),name="zero_padding_layer")(input_layer)
convolution_layer_1 = Conv1D(filters = 256,kernel_size=8,padding="valid",name='Conv_0')(padding_layer)
convolution_layer_1 = PReLU(name='PReLU_0')(convolution_layer_1)
convolution_layer_1 = SpatialDropout1D(0.01, name='SpatialDropout_0')(convolution_layer_1)
convolution_layer_1 = MaxPooling1D(pool_size=1, name='MaxPooling_0')(convolution_layer_1)
convolution_layer_1 = BatchNormalization(name='BatchNormalization_0')(convolution_layer_1)

Convolutional_Block_0 = Sequential([Model(inputs=input_layer,outputs=convolution_layer_1)],name='Convolutional_Block_0')

convolution_layer_2 = Conv1D(filters=128, kernel_size=4, padding="same", name='Conv_1')(convolution_layer_1)
convolution_layer_2 = PReLU(name='PReLU_1')(convolution_layer_2)
convolution_layer_2 = SpatialDropout1D(0.01, name='SpatialDropout_1')(convolution_layer_2)
convolution_layer_2 = MaxPooling1D(pool_size=1, name='MaxPooling_1')(convolution_layer_2)
convolution_layer_2 = BatchNormalization(name='BatchNormalization_1')(convolution_layer_2)

Convolutional_Block_1 = Sequential([Model(inputs=convolution_layer_1,outputs=convolution_layer_2)],name='Convolutional_Block_1')

convolution_layer_3 = Conv1D(filters=64, kernel_size=2, padding="same", name='Conv_2')(convolution_layer_2)
convolution_layer_3 = PReLU(name='PReLU_2')(convolution_layer_3)
convolution_layer_3 = SpatialDropout1D(0.01, name='SpatialDropout_2')(convolution_layer_3)
convolution_layer_3 = MaxPooling1D(pool_size=2, name='MaxPooling_2')(convolution_layer_3)
convolution_layer_3 = BatchNormalization(name='BatchNormalization_2')(convolution_layer_3)

Convolutional_Block_2 = Sequential([Model(inputs=convolution_layer_2,outputs=convolution_layer_3)],name='Convolutional_Block_2')

convolution_layer_4 = Conv1D(filters=64, kernel_size=2, padding="same", name='Conv_3')(convolution_layer_3)
convolution_layer_4 = PReLU(name='PReLU_3')(convolution_layer_4)
convolution_layer_4 = SpatialDropout1D(0.01, name='SpatialDropout_3')(convolution_layer_4)
convolution_layer_4 = MaxPooling1D(pool_size=2, name='MaxPooling_3')(convolution_layer_4)
convolution_layer_4 = BatchNormalization(name='BatchNormalization_3')(convolution_layer_4)

Convolutional_Block_3 = Sequential([Model(inputs=convolution_layer_3,outputs=convolution_layer_4)],name='Convolutional_Block_3')

Convolutional_Block = Sequential([Convolutional_Block_0,Convolutional_Block_1,Convolutional_Block_2,Convolutional_Block_3],name='Convolutional_Block')

Query = Conv1D(filters=64, padding="same", kernel_size=8, name=f'Query')(convolution_layer_4)

heads = 8
self_attention_layer,attention_scores = MultiHeadAttention(num_heads=heads,key_dim=64,name=f'MultiHeadAttention')(query=Query ,value=convolution_layer_4,return_attention_scores=True)

self_attention_layer = Multiply()([self_attention_layer,convolution_layer_4])

Attention_Block = Sequential([Model(inputs=convolution_layer_4,outputs=self_attention_layer)],name='Attention_Layer')
Attention_scores = Sequential([Model(inputs=input_layer,outputs=attention_scores)],name='Attention_scores')

bilstm_layer = Bidirectional(LSTM(64, return_sequences=True), merge_mode="sum", name='Bidirectional_LSTM')(self_attention_layer)
lstm_layer = LSTM(64, dropout=0.1, name='LSTM')(bilstm_layer)

Recurrent_Block = Sequential([Model(inputs=convolution_layer_4,outputs=lstm_layer)],name='Recurrent_Block')
Ensemble = Model(inputs=convolution_layer_4,outputs=lstm_layer)
Encoder = Sequential([Convolutional_Block,Ensemble],name='CEBLANE')
output_layer = Dense(1, activation="sigmoid", name='finalOutput')(Flatten(name='flattenOutput')(lstm_layer))

Output_block = Sequential([Model(inputs=lstm_layer,outputs=output_layer)],name='Output_block')
CBLANE = Sequential([Encoder,Output_block],name='CBLANE')