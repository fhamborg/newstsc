# keys in the dict must match parameter names accepted by train.py. values must match accepted values for such
# parameters in train.py
combinations = {
    'model_name': [
        # SPC
        'spc_bert', 'spc_distilbert', 'spc_roberta',
        # AEN
        'aen_bert', 'aen_distilbert', 'aen_roberta', 'aen_glove',  # 'aen_distilroberta'
    ],
    'optimizer': ['adam'],
    'initializer': ['xavier_uniform_'],
    # TODO check this and other parameters, compare with available options in train.py
    'learning_rate': ['1e-5', '2e-5', '3e-5', '4e-5', '5e-5'],
    'batch_size': ['16', '32', '48'],
    'lossweighting': ['True', 'False'],
    'devmode': ['True'],
    'num_epoch': ['2', '3', '4'],
    'lsr': ['True', 'False'],
    'use_tp_placeholders': ['True', 'False'],
    'spc_lm_representation_distilbert': [  # 'sum_last', 'sum_last_four', 'sum_last_two', 'sum_all',
        'mean_last', 'mean_last_four', 'mean_last_two', 'mean_all'],
    'spc_lm_representation': ['pooler_output',
                              # 'sum_last', 'sum_last_four', 'sum_last_two', 'sum_all',
                              'mean_last', 'mean_last_four'],  # 'mean_last_two', 'mean_all'],
    'spc_input_order': ['target_text', 'text_target'],
    'aen_lm_representation': ['last',
                              # 'sum_last_four', 'sum_last_two', 'sum_all',
                              'mean_last_four'],  # 'mean_last_two', 'mean_all'],
    # 'use_early_stopping': ['True'],  # due to condition below, will only be used if num_epoch==10
    'finetune_glove': ['True', 'False'],
    'eval_only_after_last_epoch': ['True'],
}
