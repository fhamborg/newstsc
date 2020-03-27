import shelve

import networkx as nx
import neuralcoref
import numpy as np
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token

from fxlogger import get_logger


class DependencyParser:
    def __init__(self, mode: str, max_seq_length: int, max_hop_distance: int = 10):
        self.logger = get_logger()
        self.nlp = self.__get_spacy()
        self.tag2ind, self.ind2tag = self.__create_association()
        self.num_tags = len(list(self.tag2ind.keys()))
        self.max_seq_length = max_seq_length
        self.max_hop_distance = max_hop_distance
        self.mode = mode
        cache_filepath = DependencyParser.__CACHE_FILEPATH_TEMPLATE.format(mode)
        self.cache = shelve.open(cache_filepath)
        self.logger.info(
            f"loaded cache with {len(self.cache)} entries from {cache_filepath}"
        )

    ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST = "target_dependent_depdist_weights"
    ID_TARGET_DEPENDENT_WEIGHTS_COREF = "target_dependent_corefdist_weights"
    ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST_COREF = (
        "target_dependent_depdist_corefdist_weights"
    )
    __CACHE_FILEPATH_TEMPLATE = "caches/depparser_{}.shelve"

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

    @staticmethod
    def __get_num_dephops_from_token_to_token(graph: nx.Graph, a: Token, b: Token):
        try:
            return nx.shortest_path_length(
                graph,
                source=DependencyParser._token2docdependentid(a),
                target=DependencyParser._token2docdependentid(b),
            )
        except nx.exception.NetworkXNoPath as nopathex:
            # will happen if there is no path between two nodes; it's fine
            return -1
        except nx.exception.NodeNotFound as notfoundex:
            # should never happen since we add every token as a node to the graph
            raise notfoundex

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
        """
        Load spacy's dependency tree into a networkx graph, from https://stackoverflow.com/a/41817795
        :param doc:
        :return:
        """
        graph = nx.Graph()

        for token in doc:
            # we add each token as a node and if existent any connection of that node to
            # other nodes
            graph.add_node(self._token2docdependentid(token))
            # FYI https://spacy.io/docs/api/token
            for child in token.children:
                self.logger.debug(
                    f"{token} ({self._token2docdependentid(token)}) - {child} ({self._token2docdependentid(child)})"
                )
                graph.add_edge(
                    self._token2docdependentid(token), self._token2docdependentid(child)
                )

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

        weight = max(self.max_hop_distance - dist, 0)
        normalized_weight = weight / self.max_hop_distance
        return normalized_weight

    def __calcuate_targetdependent_depdistbased_weight_vector(
        self, doc: Doc, dependency_hop_distances: dict
    ):
        token_weights = np.zeros((self.max_seq_length,), dtype=np.float32)

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
        self, doc: Doc, target_span: Span, doc_depandcoref_graph: nx.Graph,
    ):
        token_weights_from_coref_to_target = np.zeros(
            (self.max_seq_length,), dtype=np.float32
        )

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

    def __debug_print(
        self,
        doc: Doc,
        target_dependent_depdistbased_weight_vector,
        target_dependent_corefbased_weight_vector,
        target_dependent_depdistcorefbased_weight_vector,
    ):
        for t in zip(
            doc,
            target_dependent_depdistbased_weight_vector,
            target_dependent_corefbased_weight_vector,
            target_dependent_depdistcorefbased_weight_vector.tolist(),
        ):
            self.logger.info(t)

    def process_sentence(
        self, sentence: str, target_local_from: int, target_local_to: int,
    ):
        cache_id = f"{sentence}-{target_local_from}-{target_local_to}"
        if cache_id in self.cache:
            return self.cache[cache_id]

        doc = self.nlp(sentence)
        doc_dep_graph = self.__doc2graph(doc)
        doc_depandcoref_graph = doc_dep_graph.copy()

        target_span = doc.char_span(target_local_from, target_local_to)
        dependency_hop_distances_to_target = self.__get_hopdists_to_span(
            doc, target_span, doc_dep_graph
        )

        if (
            self.mode == DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST
            or self.mode == DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST_COREF
        ):
            target_dependent_depdistbased_weight_vector = self.__calcuate_targetdependent_depdistbased_weight_vector(
                doc, dependency_hop_distances_to_target
            )
        if (
            self.mode == DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_COREF
            or self.mode == DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST_COREF
        ):
            target_dependent_corefbased_weight_vector = self.__calculate_targetdependent_corefbased_weight_vector(
                doc, target_span, doc_depandcoref_graph
            )
        if self.mode == DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST_COREF:
            depcoref_hop_distances_to_target = self.__get_hopdists_to_span(
                doc, target_span, doc_depandcoref_graph
            )
            target_dependent_depdistcorefbased_weight_vector = self.__calcuate_targetdependent_depdistbased_weight_vector(
                doc, depcoref_hop_distances_to_target
            )

        result_vector = None
        if self.mode == DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST:
            result_vector = target_dependent_depdistbased_weight_vector
        elif self.mode == DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_COREF:
            result_vector = target_dependent_corefbased_weight_vector
        elif self.mode == DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST_COREF:
            result_vector = target_dependent_depdistcorefbased_weight_vector

        self.cache[cache_id] = result_vector
        self.cache.sync()
        return result_vector


if __name__ == "__main__":
    dp = DependencyParser(
        DependencyParser.ID_TARGET_DEPENDENT_WEIGHTS_DEPDIST_COREF, 150
    )
    dp.process_sentence(
        'Instead, the White House is hoping Putin would "respond positively" and "change their behavior."',
        35,
        40,
    )
    # dp.process_sentence(
    #     "John Smith followed DACA's rules, he succeeded in school, at work and in business, and he has contributed in building a better America.",
    #     0,
    #     4,
    # )
    # dp.process_sentence(
    #     "Hi, my name is John Smith and I'm a student in computer and information science at the University of Konstanz in Germany. And his name is Peter.",
    #     15,
    #     25,
    # )
