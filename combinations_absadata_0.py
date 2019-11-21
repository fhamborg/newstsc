combinations_absadata_0 = {
    'model_name': [
        # SPC
        # 'spc_bert',  # 'spc_distilbert', 'spc_roberta',
        # AEN
        #'aen_bert',  # 'aen_distilbert', 'aen_roberta', 'aen_glove',  # 'aen_distilroberta'
        # LCF
        'lcf_bert',
    ],
    'optimizer': ['adam'],
    'initializer': ['xavier_uniform_'],
    # TODO check this and other parameters, compare with available options in train.py
    'learning_rate': ['2e-5', '3e-5', '5e-5'],
    'batch_size': ['16', '32'],
    'balancing': ['None', 'lossweighting', 'oversampling'],
    'devmode': ['False'],
    'num_epoch': ['3', '4'],
    'lsr': ['True', 'False'],
    'use_tp_placeholders': ['False', 'True'],
    'spc_lm_representation_distilbert': [  # 'sum_last', 'sum_last_four', 'sum_last_two', 'sum_all',
        'mean_last'],  # , 'mean_last_four', 'mean_last_two', 'mean_all'],
    'spc_lm_representation': ['pooler_output'],
    # 'sum_last', 'sum_last_four', 'sum_last_two', 'sum_all',
    # 'mean_last'],  # 'mean_last_four'],  # 'mean_last_two', 'mean_all'],
    'spc_input_order': ['text_target'],  # 'target_text',
    'aen_lm_representation': ['last'],
    # 'sum_last_four', 'sum_last_two', 'sum_all',
    # ],  # 'mean_last_four'],  # 'mean_last_two', 'mean_all'],
    # 'use_early_stopping': ['True'],  # due to condition below, will only be used if num_epoch==10
    'finetune_glove': ['False', 'True'],
    'eval_only_after_last_epoch': ['True'],
    'local_context_focus': ['cdm', 'cdw'],  # ['cdw', 'cdm']
    'SRD': ['3'],
    'pretrained_model_name': ['bert_news_ccnc_10mio_3ep'],
    # 'pretrained_model_name': ['bert_news_ccnc_10mio_3ep', 'bert_news_ccnc_10mio_2ep', 'bert_news_ccnc_10mio_1ep',
    #                          'default', 'bert_news_ccnc_10mio_3ep', 'laptops_and_restaurants_2mio_ep15',
    #                          'laptops_1mio_ep30', 'restaurants_10mio_ep3'],
}
