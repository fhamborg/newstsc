combinations_g_0 = {
    'model_name': [
        # SPC
        # 'spc_bert', 'spc_distilbert', 'spc_roberta',
        # AEN
        # 'aen_bert', 'aen_distilbert', 'aen_roberta', 'aen_glove', 'aen_distilroberta'
        # LCF
        'lcf_bert',
    ],
    'optimizer': ['adam'],
    'initializer': ['xavier_uniform_'],
    'learning_rate': ['2e-5', '3e-5', '5e-5'],
    'batch_size': ['1'],  # ['16', '32'], # on SCC, 16 is fine for normal operation, 1 is fine for global models
    'balancing': ['lossweighting'],  # ['None', 'lossweighting', 'oversampling'],
    'devmode': ['False'],
    'num_epoch': ['3', '4'],
    'lsr': ['True', 'False'],
    'use_tp_placeholders': ['False'],  # ['False', 'True'],
    'spc_lm_representation_distilbert': ['mean_last'],
    # ['sum_last', 'sum_last_four', 'sum_last_two', 'sum_all', 'mean_last', 'mean_last_four', 'mean_last_two', 'mean_all'],
    'spc_lm_representation': ['pooler_output'],
    # ['pooler_output', 'sum_last', 'sum_last_four', 'sum_last_two', 'sum_all', 'mean_last', 'mean_last_four', 'mean_last_two', 'mean_all'],
    'spc_input_order': ['text_target'],  # 'target_text',
    'aen_lm_representation': ['last'],
    # ['last', 'sum_last_four', 'sum_last_two', 'sum_all', 'mean_last_four'],  # 'mean_last_two', 'mean_all'],
    'finetune_glove': ['False', 'True'],
    'eval_only_after_last_epoch': ['True'],
    'local_context_focus': ['cdm'],  # ['cdw', 'cdm']
    'SRD': ['3'],
    'pretrained_model_name': ['bert_news_ccnc_10mio_3ep'],
    # ['default', 'bert_news_ccnc_10mio_3ep', 'laptops_and_restaurants_2mio_ep15', 'laptops_1mio_ep30', 'restaurants_10mio_ep3'],
    'state_dict': ['None'],
    # ['None', 'lcf_bert_acl14twitter_val_recall_avg_0.7349_epoch3', 'lcf_bert_semeval14laptops_val_recall_avg_0.7853_epoch3', 'lcf_bert_semeval14restaurants_val_recall_avg_0.7672_epoch2', 'lcf_bert_newstsc_val_recall_avg_0.5954_epoch3'],
    'use_global_context': ['True'],  # ['True', 'False']
    'global_context_seqs_per_doc': ['20'],  # ['20']
}
