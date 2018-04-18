import os
import logging
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer; keyboard = Tracer()

### dividing into list (batches) ###
def div_batch(data, batch_size):
    batches = []
    for n in range(0, data.shape[0], batch_size):
        if n+batch_size > data.shape[0] - 1:
            batches.append(data[n:data.shape[0]])
        else:
            batches.append(data[n:n+batch_size])
    return batches

#===============================================================================
# Training
#===============================================================================
def Train(model,indata, indata_tactile, targetdata, indatatest, indatatest_tactile, targetdatatest,
             train_params, mode=None, summaries=None):
    DECAY = train_params['decay']
    KEEP_PROB = train_params['keep_prob']
    EPOCH = train_params['epoch']
    PRINT_ITER = train_params['print_iter']
    SNAP_ITER = train_params['snap_iter']
    TEST_ITER = train_params['test_iter']
    SNAP_DIR = train_params['snap_dir']
    LOG_NAME = train_params['log_name']
    BATCH_SIZE = train_params['batch_size']
    BATCH_SIZE_TEST = train_params['batch_size_test']
    MODE = mode
    if SNAP_DIR[-1] != '/':
        SNAP_DIR += '/'

    if not os.path.exists(SNAP_DIR):
        print('[ERROR]', SNAP_DIR, ' does not exist.')
        return None # exit this function
    
### init logger ###
    stream_logger = logging.StreamHandler()
    stream_logger.setLevel(logging.INFO)
    stream_logger.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    file_logger = logging.FileHandler(filename=LOG_NAME, mode='w')
    file_logger.setLevel(logging.INFO)
    file_logger.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    logging.getLogger().addHandler(stream_logger)
    logging.getLogger().addHandler(file_logger)
    logging.getLogger().setLevel(logging.INFO)

    
### Dividing into batches ###
    num_batch = int(len(indata_tactile)/ BATCH_SIZE)
    num_batch_test = int(indatatest_tactile.shape[0] / BATCH_SIZE_TEST)
    in_batch = div_batch(indata, BATCH_SIZE)
    out_batch = div_batch(targetdata, BATCH_SIZE)
    in_batch_tactile = div_batch(indata_tactile, BATCH_SIZE)
    in_batch_test = div_batch(indatatest, BATCH_SIZE_TEST)
    out_batch_test = div_batch(targetdatatest, BATCH_SIZE_TEST)
    in_batch_test_tactile = div_batch(indatatest_tactile, BATCH_SIZE_TEST)

#######################
#### Training loop ####
#######################
    for epoch in range(0, EPOCH):
    # Trainining for all batch
        loss_all = 0.0
        for i in range(num_batch):
            # Make batches
            x = in_batch[i]
            y = out_batch[i]
            x_tac = in_batch_tactile[i]
            # Training
            feed_dict={model.input_placeholder: x, 
                       model.teacher_placeholder: y, 
                       model.input_placeholder_tac: x_tac,  
                       model.keep_prob_placeholder:KEEP_PROB,
                       model.is_training:True}
            loss_val, opt_val = model.train(feed_dict)
            summary = model.sumrun(feed_dict, summaries)
            loss_all += loss_val
        # Print train-loss
        if epoch % PRINT_ITER == 0:
            model.summary_writer.add_summary(summary, epoch)
            logging.info('['+str(epoch)+'] loss_train = '+str(loss_all/num_batch))
#             logging.info('['+str(epoch)+'] accuracy_train = '+str(model.acc(feed_dict)))
        # Print test-loss
        if epoch % TEST_ITER == 0:
            loss_all_test = 0.0
            for i in range(num_batch_test):
                # Make batches
                x = in_batch_test[i]
                y = out_batch_test[i]
                x_tac = in_batch_test_tactile[i]
                # Testing
                feed_dict={model.input_placeholder: x, 
                           model.teacher_placeholder: y, 
                           model.input_placeholder_tac: x_tac,  
                       model.keep_prob_placeholder:1.0,
                       model.is_training:False}
                loss_val_test = model.losspre(feed_dict)
                summary_test = model.sumrun(feed_dict, summaries)
                loss_all_test += loss_val_test
        # Print num_batch_test
            model.summary_writer.add_summary(summary_test, epoch)
            logging.info('['+str(epoch)+'] loss_test = '+str(loss_all_test/num_batch_test))
#             logging.info('['+str(epoch)+'] accuracy_test = %g' % model.acc(feed_dict))
        # Snap into file
        if epoch % SNAP_ITER == 0:
            model.save(SNAP_DIR+'step', epoch)
    loss_train = loss_all/num_batch
    loss_test = loss_all_test/num_batch_test

    return loss_train, loss_test

