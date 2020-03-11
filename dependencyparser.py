import matplotlib.pyplot as plt
import networkx as nx
import neuralcoref
import numpy as np
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token

from fxlogger import get_logger


class DependencyParser:
    def __get_spacy(self):
        modelname = "en_core_web_lg"
        self.logger.info(f"creating spacy({modelname})...")
        nlp = spacy.load(modelname)

        # add neuralcoref to pipeline
        neuralcoref.add_to_pipe(nlp)

        return nlp

    @staticmethod
    def _token2docdependentid(token: Token):
        return f"{token.text}-{token.idx}"

    @staticmethod
    def __create_association():
        dep_tags = [
            "ROOT",
            "acl",
            "acomp",
            "advcl",
            "advmod",
            "agent",
            "amod",
            "appos",
            "attr",
            "aux",
            "auxpass",
            "case",
            "cc",
            "ccomp",
            "compound",
            "conj",
            "csubj",
            "csubjpass",
            "dative",
            "dep",
            "det",
            "dobj",
            "expl",
            "intj",
            "mark",
            "meta",
            "neg",
            "nmod",
            "npadvmod",
            "nsubj",
            "nsubjpass",
            "nummod",
            "oprd",
            "parataxis",
            "pcomp",
            "pobj",
            "poss",
            "preconj",
            "predet",
            "prep",
            "prt",
            "punct",
            "quantmod",
            "relcl",
            "xcomp",
        ]

        tag2ind = {}
        ind2tag = {}
        for i, dep_tag in enumerate(dep_tags):
            ind2tag[i] = dep_tag
            tag2ind[dep_tag] = i
        return tag2ind, ind2tag

    def __init__(self):
        self.logger = get_logger()
        self.nlp = self.__get_spacy()
        self.tag2ind, self.ind2tag = self.__create_association()
        self.num_tags = len(list(self.tag2ind.keys()))
        self.MAX_HOP_DISTANCE = 10

    ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST = "target_dependent_depdist_weights"
    ID_TARGET_DEPENDENT_WEIGHTS_COREF = "target_dependent_corefdist_weights"
    ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST_COREF = (
        "target_dependent_depdist_corefdist_weights"
    )

    @staticmethod
    def __get_num_dephops_from_token_to_token(graph: nx.Graph, a: Token, b: Token):
        try:
            return nx.shortest_path_length(
                graph,
                source=DependencyParser._token2docdependentid(a),
                target=DependencyParser._token2docdependentid(b),
            )
        except nx.exception.NetworkXNoPath as nopathex:
            return -1

    def __get_num_dephops_from_token_to_span(
        self, graph: nx.Graph, token: Token, span: Span, mode: str = "min"
    ):
        if mode != "min":
            raise ValueError()

        min_dist = 100
        for _t in span:
            dist = self.__get_num_dephops_from_token_to_token(graph, token, _t)
            min_dist = min(min_dist, dist)

        return min_dist

    def __get_hopdists_to_span(self, doc: Doc, span: Span, graph: nx.Graph):
        self.logger.debug(f"target span: {span}")
        depdists = {}
        for token in doc:
            depdists[
                self._token2docdependentid(token)
            ] = self.__get_num_dephops_from_token_to_span(graph, token, span)

        sorted_depdists = {}
        for tup in sorted(depdists.items(), key=lambda pair: pair[1], reverse=False):
            sorted_depdists[tup[0]] = tup[1]

        return sorted_depdists

    def __doc2graph(self, doc: Doc):
        # Load spacy's dependency tree into a networkx graph, from https://stackoverflow.com/a/41817795
        edges = []
        for token in doc:
            # FYI https://spacy.io/docs/api/token
            for child in token.children:
                self.logger.debug(
                    f"{token} ({self._token2docdependentid(token)}) - {child} ({self._token2docdependentid(child)})"
                )
                edges.append(
                    (
                        f"{self._token2docdependentid(token)}",
                        f"{self._token2docdependentid(child)}",
                    )
                )

        graph = nx.Graph(edges)
        return graph

    def __depdists2normalized(self, depdists: dict):
        maxv = -1
        for k, v in depdists.items():
            maxv = max(maxv, v)

        norm_depdists = {}
        for k, v in depdists.items():
            norm_depdists[k] = float(v) / maxv

        return norm_depdists

    def __dephop_distance_to_weight(self, dist: int):
        """
        Turns hop distances (from a token to the target) into weights. Returns 0 for tokens that do
        not have a (direct or indirect) a relation to the target, e.g., because they are in a different
        sentence than the target is.
        :param dist:
        :return:
        """
        if dist < 0:
            return 0

        weight = max(self.MAX_HOP_DISTANCE - dist, 0)
        normalized_weight = weight / self.MAX_HOP_DISTANCE
        return normalized_weight

    def __calcuate_targetdependent_depdistbased_weight_vector(
        self, doc: Doc, num_tokens: int, dependency_hop_distances: dict
    ):
        token_weights = np.zeros((num_tokens))
        for ind, token in enumerate(doc):
            token_depid = self._token2docdependentid(token)
            token_dephop_distance_to_target_span = dependency_hop_distances[token_depid]
            normalized_dependency_weight_of_token = self.__dephop_distance_to_weight(
                token_dephop_distance_to_target_span
            )
            token_weights[ind] = normalized_dependency_weight_of_token
            self.logger.debug(f"{token}: {normalized_dependency_weight_of_token}")

        return token_weights

    def __is_overlapping_spans(self, a: Span, b: Span):
        return a.end >= b.start or b.end >= a.start

    def __calculate_targetdependent_corefbased_weight_vector(
        self,
        doc: Doc,
        num_tokens: int,
        target_span: Span,
        doc_depandcoref_graph: nx.Graph,
    ):
        token_weights_from_coref_to_target = np.zeros((num_tokens))

        for dependent_ind, dependent_token in enumerate(doc):
            for crc in dependent_token._.coref_clusters:
                crc_main = crc.main
                crc_mentions = crc.mentions

                for crc_mention in crc_mentions:
                    score = 0
                    if self.__is_overlapping_spans(target_span, crc_mention):
                        score = 0.5
                    if dependent_token in crc_main:
                        score = 1

                    token_weights_from_coref_to_target[dependent_ind] = score

                    if score > 0:
                        # add to the depandcoref graph
                        for crc_mention_token in crc_mention:
                            doc_depandcoref_graph.add_edge(
                                DependencyParser._token2docdependentid(dependent_token),
                                DependencyParser._token2docdependentid(
                                    crc_mention_token
                                ),
                            )

                self.logger.debug(f"{dependent_token}: {crc}")

        return token_weights_from_coref_to_target

    def process_sentence(
        self,
        sentence: str,
        target_local_from: int,
        target_local_to: int,
        properties: list,
    ):
        doc = self.nlp(sentence)
        doc_dep_graph = self.__doc2graph(doc)
        doc_depandcoref_graph = doc_dep_graph.copy()

        target_span = doc.char_span(target_local_from, target_local_to)
        dependency_hop_distances_to_target = self.__get_hopdists_to_span(
            doc, target_span, doc_dep_graph
        )

        num_tokens = len(doc)

        target_dependent_depdistbased_weight_vector = None
        if DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST in properties:
            target_dependent_depdistbased_weight_vector = self.__calcuate_targetdependent_depdistbased_weight_vector(
                doc, num_tokens, dependency_hop_distances_to_target
            )

        target_dependent_corefbased_weight_vector = None
        if DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_COREF in properties:
            target_dependent_corefbased_weight_vector = self.__calculate_targetdependent_corefbased_weight_vector(
                doc, num_tokens, target_span, doc_depandcoref_graph
            )

        target_dependent_depdistcorefbased_weight_vector = None
        if DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST_COREF in properties:
            if target_dependent_depdistbased_weight_vector is None:
                target_dependent_depdistbased_weight_vector = self.__calcuate_targetdependent_depdistbased_weight_vector(
                    doc, num_tokens, dependency_hop_distances_to_target
                )
            if target_dependent_corefbased_weight_vector is None:
                target_dependent_corefbased_weight_vector = self.__calculate_targetdependent_corefbased_weight_vector(
                    doc, num_tokens, target_span, doc_depandcoref_graph
                )

            depcoref_hop_distances_to_target = self.__get_hopdists_to_span(
                doc, target_span, doc_depandcoref_graph
            )
            target_dependent_depdistcorefbased_weight_vector = self.__calcuate_targetdependent_depdistbased_weight_vector(
                doc, num_tokens, depcoref_hop_distances_to_target
            )

            for t in zip(
                doc,
                target_dependent_depdistbased_weight_vector,
                target_dependent_corefbased_weight_vector,
                target_dependent_depdistcorefbased_weight_vector.tolist(),
            ):
                self.logger.info(t)

        return (
            target_dependent_depdistbased_weight_vector,
            target_dependent_corefbased_weight_vector,
            target_dependent_depdistcorefbased_weight_vector,
        )


if __name__ == "__main__":
    dp = DependencyParser()
    dp.process_sentence(
        "John Smith followed DACA's rules, he succeeded in school, at work and in business, and he has contributed in building a better America.",
        0,
        4,
        [DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST_COREF],
    )

    dp.process_sentence(
        "Hi, my name is John Smith and I'm a student in computer and information science at the University of Konstanz in Germany. And his name is Peter.",
        15,
        25,
        [DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST_COREF],
    )
