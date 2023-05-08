class Config(object):
    env = 'default'
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'
    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    max_epoch = 50
    lr = 0.001  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    gamma = 2
    s = 30
    m = 0.5 #0.05
    cos_theta = 0.5
    euc_theta = 7
