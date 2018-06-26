

```python
import time
import string
import math
import itertools
import numpy as np
import tensorflow as tf
from collections import namedtuple
```

# Load and Encode
First we are going to open and load the text from anna.txt file.
Then we would like to convert it into integers for our network to use. 
Here I'm creating a couple dictionaries to convert the characters to and from integers. 
Encoding the characters as integers makes it easier to use as input in the network.


```python
with open ('anna.txt','r') as file:
    text_data = file.read()
# remove the duplicates and sort the characters in a list
vocabulary_set = sorted(set(text_data))
char_to_int = {char:i for i,char in enumerate(vocabulary_set)}
int_to_char = dict(enumerate(vocabulary_set))
encoded = np.array([char_to_int[char] for char in text_data], dtype=np.int32)
```


```python
print('size of our text data:',len(text_data))
print('the first 50 characters of the text data:')
print('------------------------- characters from the book -------------------------------------------')
text_data[:50]

```

    size of our text data: 1985223
    the first 50 characters of the text data:
    ------------------------- characters from the book -------------------------------------------





    'Chapter 1\n\n\nHappy families are all alike; every un'




```python
print('-------------------------- Same Encoded 50 Character -----------------------------------------')
encoded[:50]
```

    -------------------------- Same Encoded 50 Character -----------------------------------------





    array([31, 64, 57, 72, 76, 61, 74,  1, 16,  0,  0,  0, 36, 57, 72, 72, 81,
            1, 62, 57, 69, 65, 68, 65, 61, 75,  1, 57, 74, 61,  1, 57, 68, 68,
            1, 57, 68, 65, 67, 61, 26,  1, 61, 78, 61, 74, 81,  1, 77, 70], dtype=int32)




```python
print('As you can see, character ',int_to_char[70], 'is decoded to ',char_to_int['n'])
```

    As you can see, character  n is decoded to  70



```python
print('Number of Classes in our data set:', len(vocabulary_set))
print('the first 50 characters of the sorted vocabulary set:')
print(set(itertools.islice(vocabulary_set, 50)))
```

    Number of Classes in our data set: 83
    the first 50 characters of the sorted vocabulary set:
    {':', 'G', 'K', '(', ')', '8', 'T', 'Q', ',', '.', '2', ';', ' ', 'N', '9', '!', '%', 'A', '-', '?', '&', '@', 'R', 'S', '6', 'C', '/', '$', 'L', '7', 'P', '"', '\n', 'U', 'O', 'I', 'M', 'D', 'E', "'", '1', 'F', '*', 'H', '0', '3', '5', 'B', 'J', '4'}



```python
all_set = set(string.printable)
# print(all_set)
print('there are',len(all_set)-len(vocabulary_set),'unused characters in this text book which they are:')
print(all_set - set(text_data))
```

    there are 17 unused characters in this text book which they are:
    {'#', '\\', '{', '\x0c', '}', '[', '\x0b', '~', '+', ']', '^', '<', '>', '=', '\t', '|', '\r'}


# Training Mini-Batches

A mini batch generator is defined to yeild x and y in mini batches. 
We would like to cut the length of our data sequence to have K total batches of size N. The number of characters per batch is M. 
Sine we want to predict the next character in the sequence, y is simply the same as x, but shifted by one in our trainig set.


```python
# N is number of sequences = batch_size
# M = num_steps
# M * N = number of characters per batch
# K is total number of batches
def batch_generator(array, batch_size, num_steps):
    # number of charachters per batch
    n_char_per_batch = batch_size * num_steps
    num_batches = len(array) // n_char_per_batch
    array = array[:n_char_per_batch * num_batches]
    array = np.reshape(array, (batch_size,-1))
    # split array into batch_size of sequences
    for n in range(num_steps):
        x = array[:,n:n+num_steps]
        y_temp = array[:,n+1:n+num_steps+1]
        y = np.zeros(x.shape, dtype=x.dtype)
        y[:,:y_temp.shape[1]] = y_temp
#         y[:, :-1], y[: ,-1] = x[:, 1:] , x[:, 0]
        yield x,y
```


```python
batches = batch_generator(encoded, 10, 50)
x,y = next(batches)
print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])
print('x\n', x.shape)
print('\ny\n', y.shape)
```

    x
     [[31 64 57 72 76 61 74  1 16  0]
     [ 1 57 69  1 70 71 76  1 63 71]
     [78 65 70 13  0  0  3 53 61 75]
     [70  1 60 77 74 65 70 63  1 64]
     [ 1 65 76  1 65 75 11  1 75 65]
     [ 1 37 76  1 79 57 75  0 71 70]
     [64 61 70  1 59 71 69 61  1 62]
     [26  1 58 77 76  1 70 71 79  1]
     [76  1 65 75 70  7 76 13  1 48]
     [ 1 75 57 65 60  1 76 71  1 64]]
    
    y
     [[64 57 72 76 61 74  1 16  0  0]
     [57 69  1 70 71 76  1 63 71 65]
     [65 70 13  0  0  3 53 61 75 11]
     [ 1 60 77 74 65 70 63  1 64 65]
     [65 76  1 65 75 11  1 75 65 74]
     [37 76  1 79 57 75  0 71 70 68]
     [61 70  1 59 71 69 61  1 62 71]
     [ 1 58 77 76  1 70 71 79  1 75]
     [ 1 65 75 70  7 76 13  1 48 64]
     [75 57 65 60  1 76 71  1 64 61]]
    x
     (10, 50)
    
    y
     (10, 50)


# Building Model

### Input layer
We need placeholders for inputs, targets and dropout layer keep_probability.

### LSTM Cells

Here we build one cell of LSTM and stack them up into as many as needed in one layer.
We can have multiple hidden layers of LSTM cells.


```python
# Build LSTM cell
# num_layers : number of hidden layers (verical number of LSTM cells)
# lstm_size : number of LAST cells horizontally in each hidden layer. This should be equal to number of steps that
#we mini batch by.

def LSTM_Cells(lstm_size, num_layers, batch_size, keep_prob):
    # one lstm cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # one cell wrapped with dropout layer
    dropped = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    # stack of lstm cells in the hidden layer
    cell = tf.contrib.rnn.MultiRNNCell([dropped]*num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)
    return cell, initial_state
```

### Output Layer


```python
# placeholders for x,y and keep_prob of dropout layers
def input_generator(batch_size, num_steps):
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps],name='x')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps],name='y')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    return inputs, targets, keep_prob
```

The output of RNN cells(hidden Layers) will be fully connected to output layer through softmax to produce predictions. So the size of this layer should be the same as size of our data set characters which is 83.
So if we have N sequences of inputs, each with M steps, when they pass through L number of lstm cells in our hidden layer, the output will be size N . M . L. This is a 3D tensor object that we need to reshape in to a 2D tensor of shape (N . M) . L.
Size of output of softmax layer is the same as size of logits or number of classes.


```python
def output_layer(lstm_output, soft_in_size, soft_out_size):
    # lstm_output : MN*L (2D tensor)
    # soft_in_size : LSTM cells size (L)
    # soft_out_size : number of classes
    
    output_sequence = tf.concat(lstm_output, axis=1)
    softmax_input = tf.reshape(output_sequence, [-1, soft_in_size])
    
    # to not let the tensor get softmax weights confused with lstm weights:
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((soft_in_size, soft_out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(soft_out_size))
    
    logits = tf.matmul(softmax_input, softmax_w) + softmax_b
    softmax_out = tf.nn.softmax(logits, name='prediction')
    return softmax_out, logits
```

### Training Loss 
Targets are required to be one_hot_encoded before loss calculation. 
Cross-Entropy is used to calculate the loss. 



```python
def loss_function(logits, targets, num_classes ):
    # targets(y) : labels -> they need to be reshaped to logits shape and also one-hot encoded
    one_hot_labels = tf.one_hot(targets, num_classes)
    labels_reshaped = tf.reshape(one_hot_labels, logits.get_shape())
    
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_reshaped)
    loss = tf.reduce_mean(loss)
    return loss
```

### Optimizer and Gradient Exploding Fix
The optimizer will take in the loss and learning rate and use a threshold to clip the gradients, if they grow bigger than the threshold. This will avoid the problem of gradient exploding.
Adamoptimizer has been used, which optionally can perform "learning decay" if required. 


```python
def optimizer_unit(loss, learning_rate, clip_grad):
    
    tvar = tf.trainable_variables()
    # tf.gradient calculates the symbolic gradients of loss with respect to weights at each time step
    gradients, _ = tf.clip_by_global_norm(tf.gradients(loss, tvar), clip_grad)
    train_op = tf.train.AdamOptimizer(learning_rate, name='Adam')
    optimizer = train_op.apply_gradients(zip(gradients,tvar))
        
    return optimizer
```

### RNN Network
Following is defined RNN class that initializes the one-hot-encoded input, lstm cells, output layer. It needs to use the last/final state of LSTM for the mini-batch, so the next batch continous the state from the previous batch.
Then it will calculate the Loss and do the optimization.
Out RNN network needs number of classes, batch size, number of steps per batch, lstm cell size, number of hidden layer, gradient threshold and learning rate as input arguments.
*tf.nn.dynamic_rnn* will do the job of running the data through lstm cells for us.


```python
class RNN_class:
    
    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 num_hidden_layers=2, lstm_size=128, learning_rate=0.001,
                 grad_clip=5, sampling=False):
        
        if sampling:
             batch_size ,num_steps = 1 , 1
        else: 
             batch_size, num_steps = batch_size, num_steps
        
        # reset the graph
        tf.reset_default_graph()
        
        # RNN data flow        
        #######input#######
        self.inputs, self.targets, self.keep_prob = input_generator(batch_size, num_steps)
        
        #######LSTM cells######
        LSTM_cells, self.initial_state = LSTM_Cells(lstm_size, num_hidden_layers, batch_size, self.keep_prob)
        
        # run inputs through the LSTM cells        
        # one-hot encoded x input
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        LSTM_output, last_state = tf.nn.dynamic_rnn(LSTM_cells, x_one_hot, initial_state=self.initial_state)
        # last state of previous output will be the fist state of next one
        self.final_state = last_state
        
        #######output######
        # softmax , predictions and logits
        self.predictions, self.logits = output_layer(LSTM_output, lstm_size, num_classes)
        
        # Loss and Optimize
        self.loss = loss_function(self.logits, self.targets, num_classes)
        self.optimizer = optimizer_unit(self.loss, learning_rate, grad_clip)
```

### Hyperparameters of Netwrok
* `batch_size` - Number of sequences running through the network in one pass.
* `num_steps` - Number of characters in the sequence the network is trained on. Larger is better typically, the network will learn more long range dependencies. But it takes longer to train. 100 is typically a good number here.
* `lstm_size` - The number of units in the hidden layers.
* `num_layers` - Number of hidden LSTM layers to use
* `learning_rate` - Learning rate for training
* `keep_prob` - The dropout keep probability when training. If you're network is overfitting, try decreasing this.


```python
batch_size = 100
num_steps = 100
lstm_size = 512
num_hidden_layers = 2
learning_rate = 0.001
keep_prob = 0.5
```

### Training Network
To train the network we creat a model and pass it inputs and targets and run the optimizer.
every often checkpoints are save with the following formats:
i{iteration number}_l{# hidden layer units}.ckpt

Steps taken to training the network are as followings:
    
  * initialize the epoch size, saving and printing frequencies  
  * creat a saver instance  
  * start the tf.session  
  * globally initialize variablesin the session  
  * load the checkpoint and resume training(optional)  
  * for each epoch:
    1. initialize the state of the model and loss
    2. Go through each batch with 
          * A. preparing the feed(model inputs, model labels, model keep_prob and model initial_state)
          * B. Calculate the loss and new state
          * C. print time of training , epoch, step and loss
          * D. Save the batch checkpoint
    3. Save the epoch checkpoints



```python
epochs = 20
saving_freq = 200
printing_freq = 50

RNN_model = RNN_class(len(vocabulary_set), batch_size=batch_size, num_steps=num_steps,
                 num_hidden_layers=num_hidden_layers, lstm_size=lstm_size, learning_rate=learning_rate,
                 grad_clip=5, sampling=False)

saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
#     writer = tf.summary.FileWriter("./log/...", sess.graph)
    sess.run(tf.global_variables_initializer())
#     saver.restore(sess, 'checkpoints/______.ckpt')
    batch_counter = 0
    for epoch in range(epochs):

        loss = 0
        state = sess.run(RNN_model.initial_state)
        for x,y in batch_generator(encoded, batch_size, num_steps):
            start_time = time.time()
            batch_counter += 1
            feed = {RNN_model.inputs:x, RNN_model.targets:y, RNN_model.keep_prob:keep_prob, RNN_model.initial_state:state}
            batch_loss, state, _ = sess.run([RNN_model.loss, RNN_model.final_state, RNN_model.optimizer],feed_dict=feed)

            if batch_counter % printing_freq == 0:
                end_time = time.time()
                print("{:.4f} sec/batch".format(end_time - start_time),
                      "Epoch:{}/{}".format(epoch+1, epochs),
                      "batch number: {}".format(batch_counter),
                      "training loss: {:.4f}".format(batch_loss))

            if batch_counter % saving_freq == 0:
                saver.save(sess, "checkpoints/iter_No{}_Layer_NO{}.ckpt".format(batch_counter, lstm_size))

        saver.save(sess, "checkpoints/Epoch{}_Layer_NO{}.ckpt".format(epoch, lstm_size))
#     writer.close()
```

    0.3059 sec/batch Epoch:1/20 batch number: 50 training loss: 3.1442
    0.3050 sec/batch Epoch:1/20 batch number: 100 training loss: 3.0602
    0.3052 sec/batch Epoch:2/20 batch number: 150 training loss: 2.8047
    0.3064 sec/batch Epoch:2/20 batch number: 200 training loss: 2.4395
    0.3058 sec/batch Epoch:3/20 batch number: 250 training loss: 2.2552
    0.3053 sec/batch Epoch:3/20 batch number: 300 training loss: 2.0733
    0.3054 sec/batch Epoch:4/20 batch number: 350 training loss: 1.8890
    0.3062 sec/batch Epoch:4/20 batch number: 400 training loss: 1.6908
    0.3071 sec/batch Epoch:5/20 batch number: 450 training loss: 1.3387
    0.3069 sec/batch Epoch:5/20 batch number: 500 training loss: 1.2225
    0.3066 sec/batch Epoch:6/20 batch number: 550 training loss: 0.8134
    0.3065 sec/batch Epoch:6/20 batch number: 600 training loss: 0.8896
    0.3058 sec/batch Epoch:7/20 batch number: 650 training loss: 0.5597
    0.3070 sec/batch Epoch:7/20 batch number: 700 training loss: 0.7140
    0.3077 sec/batch Epoch:8/20 batch number: 750 training loss: 0.4491
    0.3055 sec/batch Epoch:8/20 batch number: 800 training loss: 0.5887
    0.3058 sec/batch Epoch:9/20 batch number: 850 training loss: 0.3695
    0.3058 sec/batch Epoch:9/20 batch number: 900 training loss: 0.5021
    0.3100 sec/batch Epoch:10/20 batch number: 950 training loss: 0.3254
    0.3070 sec/batch Epoch:10/20 batch number: 1000 training loss: 0.4422
    0.3103 sec/batch Epoch:11/20 batch number: 1050 training loss: 0.2854
    0.3061 sec/batch Epoch:11/20 batch number: 1100 training loss: 0.3986
    0.3064 sec/batch Epoch:12/20 batch number: 1150 training loss: 0.2734
    0.3065 sec/batch Epoch:12/20 batch number: 1200 training loss: 0.3625
    0.3058 sec/batch Epoch:13/20 batch number: 1250 training loss: 0.2482
    0.3065 sec/batch Epoch:13/20 batch number: 1300 training loss: 0.3315
    0.3092 sec/batch Epoch:14/20 batch number: 1350 training loss: 0.2306
    0.3065 sec/batch Epoch:14/20 batch number: 1400 training loss: 0.2852
    0.3065 sec/batch Epoch:15/20 batch number: 1450 training loss: 0.2157
    0.3063 sec/batch Epoch:15/20 batch number: 1500 training loss: 0.2711
    0.3068 sec/batch Epoch:16/20 batch number: 1550 training loss: 0.1964
    0.3091 sec/batch Epoch:16/20 batch number: 1600 training loss: 0.2557
    0.3065 sec/batch Epoch:17/20 batch number: 1650 training loss: 0.1970
    0.3055 sec/batch Epoch:17/20 batch number: 1700 training loss: 0.2448
    0.3067 sec/batch Epoch:18/20 batch number: 1750 training loss: 0.1805
    0.3082 sec/batch Epoch:18/20 batch number: 1800 training loss: 0.2141
    0.3105 sec/batch Epoch:19/20 batch number: 1850 training loss: 0.1697
    0.3068 sec/batch Epoch:19/20 batch number: 1900 training loss: 0.2109
    0.3068 sec/batch Epoch:20/20 batch number: 1950 training loss: 0.1671
    0.3087 sec/batch Epoch:20/20 batch number: 2000 training loss: 0.2072


# Generating New Text 
Now from the trained network we can generate new text:
1. We pass a random string as an input to the generator to start with.
2. Then we creat our model
3. Create a Saver instance
4. Restore the session checkpoints
5. initialize the model state
6. for all the chars in the input string:
    * Encode the character
    * Run the model and append the predictions
7. for all the samples:
    * feed the predictions back to the netweork
    * Run the mode and append the predictions to the sample list



```python
def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c
```


```python
def text_generator(checkpoints, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [char for char in prime]
    RNN_model = RNN_class(len(vocabulary_set), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoints)
        init_state = sess.run(RNN_model.initial_state)
        for char in prime:
            x = np.zeros((1,1))   # x[[0]]
            x[0,0]= char_to_int[char]
            feed = {RNN_model.inputs:x, RNN_model.keep_prob:1.0, RNN_model.initial_state:init_state}
            predics, new_state = sess.run([RNN_model.predictions, RNN_model.final_state],feed_dict=feed)
        c = pick_top_n(predics, len(vocabulary_set))
        samples.append(int_to_char[c])
            
        for sample in range(n_samples):
            x[0,0] = c
            feed = {RNN_model.inputs:x, RNN_model.keep_prob:1.0,RNN_model.initial_state:new_state}
            predics, new_state = sess.run([RNN_model.predictions, RNN_model.final_state], feed_dict=feed)
            c = pick_top_n(predics, len(vocabulary_set))
            samples.append(int_to_char[c])
    return ''.join(samples)
```


```python
tf.train.get_checkpoint_state('checkpoints')
```




    model_checkpoint_path: "checkpoints/Epoch19_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch0_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/iter_No200_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch1_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch2_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/iter_No400_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch3_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch4_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/iter_No600_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch5_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch6_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/iter_No800_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch7_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch8_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/iter_No1000_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch9_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch10_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/iter_No1200_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch11_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch12_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/iter_No1400_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch13_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch14_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/iter_No1600_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch15_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch16_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/iter_No1800_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch17_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch18_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/iter_No2000_Layer_NO512.ckpt"
    all_model_checkpoint_paths: "checkpoints/Epoch19_Layer_NO512.ckpt"




```python
checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = text_generator(checkpoint, 2000, lstm_size, len(vocabulary_set), prime="Far")
print(samp)
```

    Fare had been his, and sut her lanct his
    marre, whith his faur curls, his blue
    eyes, and his plump, graceful little legs in tightly pulled-up
    stockings. Anna experienced almost physical pleaser. 
    The flonk of the wat see andion, her he
    sunderethat whe stam of his hight wase
    diany, wele gelling thet weald vodchad and sten and shat leald not herpelss, and his plems, gfances all geling wath th chas and would have said the rownor to have to ho with the district
    authorities. Where one would have to write out sheaves of papers, herellake of him. Seizing the first pretext, she got
    up, and with her light, resolute step went for her album. The stairs up
    to her room came out on the landing of the gring oot out
    he mand of the lade tianome to he reant ly mad your ander.
    
    "I son was ind abreat more now ald beched and love, with him, and the old wate the lough the light, stong what the last of wosd. En with a mease of his hand hew his
    he her her was precing of the barrit, and mose und with his unat the how out he d grant her and the pronow, stame ous
    of the warls of was on
    
    "What you de Hat, sard on'r not eve you now her fang oun your leve, whice you can't give
    her; and the other sacrifices everything for you and asks for nothing.
    What are you to do? How corled her and the uabloss thim ng heated
    prosersous, and his haad mon unhander, he wis selight, was the slow he was going out, her said to a _soreet he
    plostecter to the atcomnane to him, and hid of ronch out of the war, he was glaving away her secret, and
    that her face, burning with the flush of shame, had betrayed her
    already.
    
    "I tee have mere than whe
    did arvice and chisfert. She had been on the lookout for her,
    glancing at her watch every minute,, and, as so often happens, let slip
    just that minute when her visitor arrived, so that she would not have the strength to exchange it for the
    shameful position of a woman who has abandoned his touth and eancound of haviong sith, gut the sprite of that stering thite
    vouch ald as stor a 


The text generated by the RNN network doesn't make any sensce to us, but still it is pretty fascianting!
