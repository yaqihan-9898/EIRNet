import numpy as np
import lib.config.config as cfg

if cfg.FLAGS.dataset=='DOSR':
    uplevelmap={0: 0, 1: 1, 2: 3, 3: 3, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 2, 10: 1, 11: 2, 12: 3, 13: 1, 14: 3, 15: 4, 16: 1, 17: 1, 18: 1, 19: 3, 20: 4}
    # uplevelmap_plus={0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1}

elif cfg.FLAGS.dataset=='HRSC2016':
    uplevelmap = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 2, 14: 3, 15: 2,
                  16: 4, 17: 3, 18: 4, 19: 4, 20: 4, 21: 4, 22: 4, 23: 3, 24: 3, 25: 4, 26: 2}  # HRSC
    # uplevelmap_plus = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1,
    #                    15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1}
    # downlevelmap = {0: [0], 1: [1], 2: [2, 6, 7, 13, 15, 26], 3: [3, 8, 9, 10, 11, 12, 14, 17, 23, 24],
    #                 4: [4, 16, 18, 19, 20, 21, 22, 25], 5: [5]}
else:
    raise NotImplementedError('We only provide best class weight for DOSR and HRSC2016.\nplease check your dataset name in config.\n'
                              'If you want to train other dataset, please change the L2 upper-level classes by yourself.')
def uplevel_utils(x):
    z = np.random.randint(2, size=(cfg.FLAGS.batch_size),dtype='int32')
    for i in range(len(x)):
        z[i] = uplevelmap[x[i]]
    return z

# def uplevel_utils_plus(x):
#     z = np.random.randint(2, size=(cfg.FLAGS.batch_size),dtype='int32')
#     for i in range(len(x)):
#         z[i] = uplevelmap_plus[x[i]]
#     return z

# def convert_uplevel(x):
#     '''
#     :param x: batch_size * low_cls_num
#     :return: batch_size * uplevel_num
#     '''
#     z = np.zeros([cfg.FLAGS.batch_size,cfg.FLAGS.uplevel_len], dtype='float32')
#     for i in range(len(x)):
#         for j in range(cfg.FLAGS.uplevel_len):
#             for it in downlevelmap:
#                 val = downlevelmap[it]
#                 z[i][it] = sum(x[i][kk] for kk in val)
#     return z










