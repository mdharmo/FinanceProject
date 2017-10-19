import numpy as np
import tensorflow as tf
import helpers
import time

def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths) # Gives us all false at initial
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None

    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)

def loop_fn_transition(time,previous_output,previous_state,previous_loop_state):

    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output,W),b)
        prediction = tf.argmax(output_logits,axis=1)
        next_input = tf.nn.embedding_lookup(embeddings,prediction)
        return next_input
    elements_finished = (time >= decoder_lengths)

    finished = tf.reduct_all(elements_finished)
    input = tf.cond(finished,lambda: pad_step_embedded,get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None


    return (elements_finished,
            input,
            state,
            output,
            loop_state)

def loop_fn(time, previous_output,previous_state,previous_loop_state):
    if previous_state is None:
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time,previous_output,previous_state,previous_loop_state)

def train(sess,EOS,decoder_inputs,encoder_inputs,decoder_targets,train_op,loss,decoder_prediction):

    max_batches = 3001
    batches_in_epoch = 1000
    loss_track = []

    for batch in range(max_batches):
        fd = next_feed(EOS,encoder_inputs,decoder_inputs,decoder_targets)
        _, train_loss = sess.run([train_op,loss],fd)
        loss_track.append(train_loss)

        if batch == 0 or batch%batches_in_epoch==0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss,fd)))
            predict_ = sess.run(decoder_prediction,fd)
            print(predict_)
            for i, (inp,pred) in enumerate(zip(fd[encoder_inputs].T,predict_.T)):
                print('   sample {}:'.format(i+1))
                print('     input     > {}'.format(inp))
                print('     predicted > {}'.format(pred))
                if i>= 2:
                    break
            print()

    return

def next_feed(EOS,PAD,encoder_inputs,encoder_input_lengths,decoder_targets):

    batch_size = 100
    batches = helpers.random_sequences(length_from=3,length_to=8,vocab_lower=2,vocab_upper=10,batch_size=batch_size)
    batch=next(batches)
    encoder_inputs_,encoder_input_lengths_ = helpers.batch(batch)

    decoder_targets_,_ = helpers.batch(
        [(sequence)+[EOS]+[PAD]*2 for sequence in batch]
    )

    decoder_inputs_,_ = helpers.batch([[EOS]+(sequence) for sequence in batch])
    return {encoder_inputs: encoder_inputs_,encoder_input_lengths: encoder_input_lengths_, decoder_targets: decoder_targets_}

def predict(data,x,y_true,session,y_pred_cls,y_pred,cost):

    # Number of images in the test-set.
    num_test = int(len(data.images))

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.


    # I'm just doing it one at a time.
    feed_dict = {x: data.images[:],
                     y_true: data.labels[:]}
    true_cost = session.run(cost,feed_dict=feed_dict)
    pred = session.run(y_pred,feed_dict=feed_dict)
    cls_pred = session.run(y_pred_cls,feed_dict=feed_dict)

    return cls_pred,num_test,pred,true_cost

def build_model(encoder_hidden_units,encoder_inputs_embedded,
                vocab_size,decoder_hidden_units,decoder_inputs_embedded,decoder_targets,encoder_inputs,encoder_inputs_length,
                EOS,PAD,embeddings):


    # Initialize the encoder cell as old-fashioned rnn cell
    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

    # We need to get the encoder outputs and especially final state for the decoder
    encoder_outputs,encoder_final_state = tf.nn.dynamic_rnn(encoder_cell,encoder_inputs_embedded,
                                                            dtype=tf.float32,time_major=True)




    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

    encoder_max_time,batch_size = tf.unstack(tf.shape(encoder_inputs))
    decoder_lengths = encoder_inputs_length+3

    W = tf.Variable(tf.random_uniform([decoder_hidden_units,vocab_size],-1,1),dtype=tf.float32)
    b = tf.Variable(tf.zeros([vocab_size]),dtype=tf.float32)

    # The embedding stuff is just preparing tokes for the decoder.  We don't feed in all time at once,
    # so this step is necessary...

    eos_time_slice = tf.ones([batch_size],dtype=tf.int32,name='EOS')
    pad_time_slice = tf.zeros([batch_size],dtype=tf.int32,name='PAD')

    eos_step_embedded = tf.nn.embedding_lookup(embeddings,eos_time_slice)
    pad_step_embedded = tf.nn.embedding_lookup(embeddings,pad_time_slice)

    decoder_outputs,decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell,loop_fn)
    decoder_ouputs = decoder_outputs.stack()

    decoder_max_steps,decoder_batch_size,decoder_dim = tf.unstakc(tf.shape(decoder_ouputs))
    decoder_outputs_flat = tf.reshpae(decoder_outputs,(-1,decoder_dim))
    decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat,W),b)
    decoder_logits = tf.reshpae(decoder_logits_flat,(decoder_max_steps,decoder_batch_size,vocab_size))

    decoder_prediction=tf.argmax(decoder_logits,2)

    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets,depth=vocab_size,dtype=tf.float32),
                                                                           logits=decoder_logits)
    loss = tf.reduce_mean(stepwise_cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return decoder_logits,decoder_prediction,train_op,loss

def main():

    PAD = 0
    EOS = 1

    vocab_size = 10
    input_embedding_size = 20

    encoder_hidden_units = 20
    decoder_hidden_units = encoder_hidden_units*2

    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    encoder_inputs_length = tf.placeholder(shape=(None,),dtype=tf.int32,name='encoder_inputs_length')

    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

    decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

    embeddings = tf.Variable(tf.random_uniform([vocab_size,input_embedding_size],-1.,1.),dtype=tf.float32)

    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings,encoder_inputs)
    decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings,decoder_inputs)

    print('Building Model')
    decoder_logits,decoder_prediction,train_op,loss = build_model(encoder_hidden_units,encoder_inputs_embedded,
                vocab_size,decoder_hidden_units,decoder_inputs_embedded,decoder_targets,encoder_inputs,encoder_inputs_length,EOS,PAD,
                                                                  embeddings)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    current_f1 = 0.
    best_f1 = current_f1
    num_epochs = 0
    patience = 0
    print()
    print('Begin Training')
    print()

    train(sess,EOS,decoder_inputs,encoder_inputs,decoder_targets,train_op,loss,decoder_prediction)




    return

if __name__=='__main__':
    main()