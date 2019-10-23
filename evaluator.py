from statistics import mean

from sklearn import metrics

from fxlogger import get_logger


class Evaluator:
    def __init__(self, sorted_expected_label_values, polarity_associations, snem_name):
        self.logger = get_logger()
        self.polarity_associations = polarity_associations
        self.pos_label_value = polarity_associations['positive']
        self.neg_label_value = polarity_associations['negative']
        self.sorted_expected_label_values = sorted_expected_label_values
        self.pos_label_index = self.sorted_expected_label_values.index(self.pos_label_value)
        self.neg_label_index = self.sorted_expected_label_values.index(self.neg_label_value)
        self.snem_name = snem_name

    def calc_statistics(self, y_true, y_pred):
        f1_macro = metrics.f1_score(y_true, y_pred, labels=self.sorted_expected_label_values, average='macro')
        f1_of_classes = metrics.f1_score(y_true, y_pred, labels=self.sorted_expected_label_values, average=None)
        f1_posneg = (f1_of_classes[self.pos_label_index] + f1_of_classes[self.neg_label_index]) / 2.0
        multilabel_confusion_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred,
                                                                          labels=self.sorted_expected_label_values)
        recalls_of_classes = metrics.recall_score(y_true, y_pred, labels=self.sorted_expected_label_values,
                                                  average=None)
        recall_avg = mean(recalls_of_classes)
        recall_macro = metrics.recall_score(y_true, y_pred, labels=self.sorted_expected_label_values, average='macro')
        precision_macro = metrics.precision_score(y_true, y_pred, labels=self.sorted_expected_label_values,
                                                  average='macro')
        accuracy = metrics.accuracy_score(y_true, y_pred)

        return {'f1_macro': f1_macro, 'multilabel_confusion_matrix': multilabel_confusion_matrix,
                'recalls_of_classes': recalls_of_classes, 'recall_avg': recall_avg, 'recall_macro': recall_macro,
                'precision_macro': precision_macro, 'accuracy': accuracy, 'f1_posneg': f1_posneg}

    def log_statistics(self, stats):
        self.logger.info("{}: {}".format(self.snem_name, stats[self.snem_name]))
        self.logger.info('> recall_avg: {:.4f}, f1_posneg: {:.4f}, acc: {:.4f}, f1_macro: {:.4f}'.format(
            stats['recall_avg'], stats['f1_posneg'], stats['accuracy'], stats['f1_macro'],
        ))