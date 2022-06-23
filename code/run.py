import os

import time
import copy
import shutil
import argparse
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras

from dataenv import DataEnv
from model import Model
import config
import datasets
import logging
import logging.handlers
import sys
import random

logger = None


def set_logger(logname):
    global logger
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def loss_fn(prediction, tails):
    """ predictions: (tf.Tensor) batch_size x n_nodes
        tails: (np.array) batch_size
    """
    pred_idx = tf.stack([tf.range(0, len(tails)), tails], axis=1)  # batch_size x 2
    pred_prob = tf.gather_nd(prediction, pred_idx)  # batch_size
    pred_loss = tf.reduce_mean(- tf.math.log(pred_prob + 1e-20))
    return pred_loss


def calc_metric(heads,
                relations,
                prediction, targets, filter_pool):
    hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = 0., 0., 0., 0., 0., 0., 0.

    n_preds = prediction.shape[0]
    for i in range(n_preds):
        head = heads[i]
        rel = relations[i]
        tar = targets[i]
        pred = prediction[i]
        fil = list(filter_pool[(head, rel)] - {tar})

        sorted_idx = np.argsort(-pred)
        mask = np.logical_not(np.isin(sorted_idx, fil))
        sorted_idx = sorted_idx[mask]

        rank = np.where(sorted_idx == tar)[0].item() + 1

        if rank <= 1:
            hit_1 += 1
        if rank <= 3:
            hit_3 += 1
        if rank <= 5:
            hit_5 += 1
        if rank <= 10:
            hit_10 += 1
        mr += rank
        mrr += 1. / rank
        max_r = max(max_r, rank)

    hit_1 /= n_preds
    hit_3 /= n_preds
    hit_5 /= n_preds
    hit_10 /= n_preds
    mr /= n_preds
    mrr /= n_preds

    return hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r


def calc_metric_v2(heads,
                   relations,
                   prediction, candidates):
    aps = []
    n_preds = prediction.shape[0]
    for i in range(n_preds):
        head = heads[i]
        pred = prediction[i]
        tail_dct = candidates[head]

        score_ans = [(pred[tail], ans) for tail, ans in tail_dct.items()]
        score_ans.sort(key=lambda x: x[0], reverse=True)

        ranks = []
        correct = 0
        for idx, item in enumerate(score_ans):
            if item[1] == '+':
                correct += 1
                ranks.append(correct / (1. + idx))
        if len(ranks) == 0:
            ranks.append(0)
        aps.append(np.mean(ranks))
    mean_ap = np.mean(aps)
    n_queries = len(aps)
    return mean_ap, n_queries


class Trainer(object):
    def __init__(self, model, data_env, hparams):
        self.model = model
        self.data_env = data_env
        self.hparams = hparams

        if hparams.clipnorm is None:
            self.optimizer = keras.optimizers.Adam(learning_rate=self.get_lr())
        else:
            self.optimizer = keras.optimizers.Adam(learning_rate=self.get_lr(), clipnorm=hparams.clipnorm)

        self.train_loss = None
        self.train_accuracy = None

    def get_lr(self, epoch=None):
        if isinstance(hparams.learning_rate, float):
            return hparams.learning_rate
        elif isinstance(hparams.learning_rate, (tuple, list)):
            if epoch is None:
                return hparams.learning_rate[-1]
            else:
                return hparams.learning_rate[epoch - 1]
        else:
            raise ValueError('Invalid `learning_rate`')

    def train_step(self, heads, tails,
                   rels,
                   epoch, batch_i, tc=None):
        self.optimizer.learning_rate = self.get_lr(epoch)
        n_splits = 1
        heads_li, tails_li, rels_li = [heads], [tails], [rels]
        # heads_li, tails_li = [heads], [tails]
        while True:
            try:
                train_loss = 0.
                accuracy = 0.
                prediction_li = []
                for k in range(n_splits):
                    # heads, tails = heads_li[k], tails_li[k]
                    heads, tails, rels = heads_li[k], tails_li[k], rels_li[k]
                    self.model.set_init(heads, rels, tails, batch_i, epoch)
                    # self.model.set_init(heads, tails, batch_i, epoch)
                    with tf.GradientTape() as tape:
                        hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb = \
                            self.model.initialize(tc=tc, training=True)

                        for step in range(1, self.hparams.con_steps + 1):
                            hidden_uncon, hidden_con, node_attention = \
                                self.model.flow(hidden_uncon, hidden_con, node_attention,
                                                query_head_emb, query_rel_emb, tc=tc, training=True)
                        prediction = node_attention

                        # pred_tails = tf.argmax(node_attention, axis=1)
                        # self.model.graph.add_inputs(np.stack([heads, rels, tails, pred_tails], axis=1), epoch, batch_i)

                        loss = loss_fn(prediction, tails)

                    if tc is not None:
                        t0 = time.time()
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    if tc is not None:
                        tc['grad']['comp'] += time.time() - t0

                    if tc is not None:
                        t0 = time.time()
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    if tc is not None:
                        tc['grad']['apply'] += time.time() - t0

                    train_loss += loss
                    accuracy += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1), tails), tf.float32))
                    prediction_li.append(prediction.numpy())

                train_loss /= n_splits
                accuracy /= n_splits

                decay = self.hparams.moving_mean_decay
                self.train_loss = train_loss if self.train_loss is None else self.train_loss * decay + train_loss * (
                        1 - decay)
                self.train_accuracy = accuracy if self.train_accuracy is None else self.train_accuracy * decay + accuracy * (
                        1 - decay)

                train_metric = None
                if self.hparams.print_train_metric:
                    train_metric = calc_metric(np.concatenate(heads_li, axis=0),
                                               np.concatenate(rels_li, axis=0),
                                               np.concatenate(prediction_li, axis=0),
                                               np.concatenate(tails_li, axis=0),
                                               self.data_env.filter_pool)

                return train_loss.numpy(), accuracy.numpy(), self.train_loss.numpy(), self.train_accuracy.numpy(), train_metric

            except tf.errors.ResourceExhaustedError:
                print('Meet `tf.errors.ResourceExhaustedError`')
                n_splits += 1
                print('split into %d batches' % n_splits)
                heads_li = np.array_split(heads, n_splits, axis=0)
                tails_li = np.array_split(tails, n_splits, axis=0)
                rels_li = np.array_split(rels, n_splits, axis=0)


class Evaluator(object):
    def __init__(self, model, data_env, hparams, mode='test', rel=None):
        self.model = model
        self.data_env = data_env
        self.hparams = hparams

        if mode == 'test':
            self.test_candidates = data_env.test_candidates if rel is None else data_env.test_candidates[rel]
        else:
            self.test_candidates = None

        self.heads = []
        self.relations = []
        self.prediction = []
        self.targets = []

        self.eval_loss = None
        self.eval_accuracy = None

        if hparams.test_analyze_attention:
            self.entropy_along_steps = [0.] * hparams.con_steps
            self.top1_occupy_along_steps = [0.] * hparams.con_steps
            self.top3_occupy_along_steps = [0.] * hparams.con_steps
            self.top5_occupy_along_steps = [0.] * hparams.con_steps
            self.count = 0

    def eval_step_recommend(self, dict_r, epoch, batch_i, rels,
                            disable_output_attention=True, disable_analyze_attention=True):
        prediction = 0.0
        recall = 0.0
        count = 0
        Hit = 0.0
        dict_key = list(dict_r.keys())
        random.shuffle(dict_key)
        for u in dict_key:
            pre = []
            count += 1
            hit_num = 0.0
            self.model.set_init(tf.constant([u for _ in range(5)]),
                                rels,
                                tails=dict_r[u], batch_i=batch_i, epoch=epoch)

            hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb = \
                self.model.initialize(training=False,
                                      output_attention=not disable_output_attention,
                                      analyze_attention=not disable_analyze_attention)
            for step in range(1, self.hparams.con_steps + 1):
                hidden_uncon, hidden_con, node_attention = \
                    self.model.flow(hidden_uncon, hidden_con, node_attention, query_head_emb,
                                    query_rel_emb,
                                    training=False,
                                    output_attention=not disable_output_attention,
                                    analyze_attention=not disable_analyze_attention)

            pred = tf.math.top_k(node_attention, 5).indices  # batch_size x n_nodes
            pred = list(pred.numpy())
            for i in range(5):
                for item in pred[i]:
                    if 34463 >= self.data_env.ds.id2entity[int(item)] >= 22363:
                        pre.append(int(item))
                        break
            pre = list(set(pre))
            if len(pre) > 1:
                pre = pre[:1]
            for item in range(0, len(pre)):
                if pre[item] in dict_r[u]:
                    hit_num += 1
            prediction_t = hit_num / 10
            if hit_num != 0:
                Hit += 1
            recall_t = hit_num / len(dict_r[u])
            prediction += prediction_t
            recall += recall_t
            print(count)
            if count % 128 == 0:
                logger.info('[count]{:d};[Prediction] {:6f}'.format(count, prediction / count))
                logger.info('[count]{:d};[Recall]{:6f}.'.format(count, recall / count))
                logger.info('[count]{:d};[Hit]{:6f}.'.format(count, Hit / count))
        prediction = prediction / len(dict_r.keys())
        recall = recall / len(dict_r.keys())
        Hit = Hit / len(dict_r.keys())
        logger.info('[A_Prediction] {:6f}'.format(prediction))
        logger.info('[A_Recall]{:6f}'.format(recall))
        logger.info('[A_Hit]{:6f}'.format(Hit))
    # def eval_step(self, heads, tails,
    #               rels,
    #               epoch, batch_i,
    #               disable_output_attention=True, disable_analyze_attention=True):
    #     n_splits = 1
    #     heads_li, tails_li, rels_li = [heads], [tails], [rels]
    #     #heads_li, tails_li = [heads], [tails]
    #     while True:
    #         try:
    #             eval_loss = 0.
    #             accuracy = 0.
    #             prediction_li = []
    #             for k in range(n_splits):
    #                 heads, tails, rels = heads_li[k], tails_li[k], rels_li[k]
    #                 #heads, tails = heads_li[k], tails_li[k]
    #                 self.model.set_init(heads,
    #                                     rels,
    #                                     tails, batch_i, epoch)
    #
    #                 # query_rel_emb = \
    #                 hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb = \
    #                     self.model.initialize(training=False,
    #                                           output_attention=not disable_output_attention,
    #                                           analyze_attention=not disable_analyze_attention)
    #
    #                 for step in range(1, self.hparams.con_steps + 1):
    #                     hidden_uncon, hidden_con, node_attention = \
    #                         self.model.flow(hidden_uncon, hidden_con, node_attention, query_head_emb,
    #                                         query_rel_emb,
    #                                         training=False,
    #                                         output_attention=not disable_output_attention,
    #                                         analyze_attention=not disable_analyze_attention)
    #                 prediction = node_attention  # batch_size x n_nodes
    #
    #                 self.heads.append(heads)
    #                 self.relations.append(rels)
    #                 self.prediction.append(prediction.numpy())
    #                 self.targets.append(tails)
    #
    #                 loss = loss_fn(prediction, tails)
    #
    #                 eval_loss += loss
    #                 accuracy += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1), tails), tf.float32))
    #
    #                 prediction_li.append(prediction.numpy())
    #
    #                 if not disable_output_attention:
    #                     self.model.save_attention_to_file(node_attention,
    #                                                       self.data_env.ds.id2entity,
    #                                                       self.data_env.ds.id2relation,
    #                                                       epoch, self.hparams.dir_name,
    #                                                       training=False)
    #
    #                 if not disable_analyze_attention:
    #                     self.entropy_along_steps = [a + b for a, b in zip(self.entropy_along_steps,
    #                                                                       self.model.entropy_along_steps)]
    #                     self.top1_occupy_along_steps = [a + b for a, b in zip(self.top1_occupy_along_steps,
    #                                                                           self.model.top1_occupy_along_steps)]
    #                     self.top3_occupy_along_steps = [a + b for a, b in zip(self.top3_occupy_along_steps,
    #                                                                           self.model.top3_occupy_along_steps)]
    #                     self.top5_occupy_along_steps = [a + b for a, b in zip(self.top5_occupy_along_steps,
    #                                                                           self.model.top5_occupy_along_steps)]
    #                     self.count += 1
    #
    #             eval_loss /= n_splits
    #             accuracy /= n_splits
    #
    #             decay = self.hparams.moving_mean_decay
    #             self.eval_loss = eval_loss if self.eval_loss is None else self.eval_loss * decay + eval_loss * (
    #                     1 - decay)
    #             #self.eval_accuracy = accuracy if self.eval_accuracy is None else self.eval_accuracy * decay + accuracy * (
    #             #        1 - decay)
    #             self.eval_accuracy = accuracy if self.eval_accuracy is None else self.eval_accuracy + accuracy
    #
    #             cur_accuracy = self.eval_accuracy/epoch
    #
    #             return eval_loss.numpy(), accuracy.numpy(), self.eval_loss.numpy(), cur_accuracy.numpy()
    #
    #         except (tf.errors.InternalError, tf.errors.ResourceExhaustedError, SystemError):
    #             print('Meet `tf.errors.InternalError` or `tf.errors.ResourceExhaustedError` or `SystemError`')
    #             n_splits += 1
    #             print('split into %d batches' % n_splits)
    #             heads_li = np.array_split(heads, n_splits, axis=0)
    #             tails_li = np.array_split(tails, n_splits, axis=0)
    #             rels_li = np.array_split(rels, n_splits, axis=0)

    def reset_metric(self):
        self.heads = []
        self.relations = []
        self.prediction = []
        self.targets = []

        self.eval_loss = None
        self.eval_accuracy = None

    def metric_result(self):
        heads = np.concatenate(self.heads, axis=0)
        relations = np.concatenate(self.relations, axis=0)
        prediction = np.concatenate(self.prediction, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        return calc_metric(heads,
                           relations,
                           prediction, targets, self.data_env.filter_pool)

    def metric_result_v2(self):
        if self.test_candidates is None:
            return None
        else:
            heads = np.concatenate(self.heads, axis=0)
            relations = np.concatenate(self.relations, axis=0)
            prediction = np.concatenate(self.prediction, axis=0)
            return calc_metric_v2(heads,
                                  relations
                                  , prediction, self.test_candidates)

    def metric_for_analysis(self):
        if self.hparams.test_analyze_attention:
            entropy_along_steps = [a / self.count for a in self.entropy_along_steps]
            top1_occupy_along_steps = [a / self.count for a in self.top1_occupy_along_steps]
            top3_occupy_along_steps = [a / self.count for a in self.top3_occupy_along_steps]
            top5_occupy_along_steps = [a / self.count for a in self.top5_occupy_along_steps]
            return entropy_along_steps, top1_occupy_along_steps, top3_occupy_along_steps, top5_occupy_along_steps
        else:
            return None


def reset_time_cost(hparams):
    if hparams.timer:
        return {'model': defaultdict(float), 'graph': defaultdict(float), 'grad': defaultdict(float)}
    else:
        return None


def str_time_cost(tc):
    if tc is not None:
        model_tc = ', '.join('m.{} {:3f}'.format(k, v) for k, v in tc['model'].items())
        graph_tc = ', '.join('g.{} {:3f}'.format(k, v) for k, v in tc['graph'].items())
        grad_tc = ', '.join('d.{} {:3f}'.format(k, v) for k, v in tc['grad'].items())
        return model_tc + ', ' + graph_tc + ', ' + grad_tc
    else:
        return ''


def run_eval(data_env, model, hparams, epoch, batch_i,
             enable_test_bs=False, disable_output_attention=True, disable_analyze_attention=True):
    if hparams.eval_valid:
        logger.info('[Valid_len] {:d}'.format(data_env.n_valid))
        valid_evaluator = Evaluator(model, data_env, hparams, mode='valid')
        valid_evaluator.reset_metric()
        data_env.get_setGraph()
        b_i = 1
        # n_b = data_env.n_valid / hparams.test_batch_size
        test_data = data_env.valid

        heads, tails, rels = test_data[:, 0], test_data[:, 1], test_data[:, 2]
        dict_r = dict()
        for item in range(0, len(heads)):
            if heads[item] not in dict_r:
                dict_r[heads[item]] = []
            dict_r[heads[item]].append(tails[item])
        valid_evaluator.eval_step_recommend(dict_r, epoch, b_i, rels, disable_output_attention=disable_output_attention,
                                            disable_analyze_attention=disable_analyze_attention)

        return

    if data_env.test_by_rel is not None:
        for rel in data_env.get_test_relations():
            logger.info('[test_len] {:d}'.format(data_env.n_test))
            test_evaluator = Evaluator(model, data_env, hparams, mode='test')
            test_evaluator.reset_metric()
            data_env.get_setGraph()
            b_i = 1
            # n_b = data_env.n_valid / hparams.test_batch_size
            test_data = data_env.test
            heads, tails, rels = test_data[:, 0], test_data[:, 1], test_data[:, 2]
            dict_r = dict()
            for item in range(0, len(heads)):
                if heads[item] not in dict_r:
                    dict_r[heads[item]] = []
                dict_r[heads[item]].append(tails[item])
            test_evaluator.eval_step_recommend(dict_r, epoch, b_i, rels,
                                                disable_output_attention=disable_output_attention,
                                                disable_analyze_attention=disable_analyze_attention)

    else:
        logger.info('[test_len] {:d}'.format(data_env.n_test))
        test_evaluator = Evaluator(model, data_env, hparams, mode='test')
        test_evaluator.reset_metric()
        data_env.get_setGraph()
        b_i = 1
        # n_b = data_env.n_valid / hparams.test_batch_size
        test_data = data_env.test
        heads, tails, rels = test_data[:, 0], test_data[:, 1], test_data[:, 2]
        dict_r = dict()
        for item in range(0, len(heads)):
            if heads[item] not in dict_r:
                dict_r[heads[item]] = []
            dict_r[heads[item]].append(tails[item])
        test_evaluator.eval_step_recommend(dict_r, epoch, b_i, rels,
                                           disable_output_attention=disable_output_attention,
                                           disable_analyze_attention=disable_analyze_attention)
        if not disable_analyze_attention:
            entropy, top1_occupy, top3_occupy, top5_occupy = test_evaluator.metric_for_analysis()
            print(
                '[ANALYSIS TEST] {:d}, {:d} | entropy: {} | top1_occupy: {} | top3_occupy: {} | top5_occupy: {} * * * * *'.format(
                    epoch, batch_i,
                    ', '.join(['{:.6f}'.format(e) for e in entropy]),
                    ', '.join(['{:.6f}'.format(o) for o in top1_occupy]),
                    ', '.join(['{:.6f}'.format(o) for o in top3_occupy]),
                    ', '.join(['{:.6f}'.format(o) for o in top5_occupy])))
            logger.info('[ANALYSIS TEST] {:d}, {:d} | entropy: {} | top1_occupy: {} | top3_occupy: {} | top5_occupy: '
                        '{} * * * * *'.format(epoch, batch_i,
                                              ', '.join(['{:.6f}'.format(e) for e in entropy]),
                                              ', '.join(['{:.6f}'.format(o) for o in top1_occupy]),
                                              ', '.join(['{:.6f}'.format(o) for o in top3_occupy]),
                                              ', '.join(['{:.6f}'.format(o) for o in top5_occupy])))


def run(dataset, hparams):
    data_env = DataEnv(dataset)
    model = Model(data_env.graph, hparams)

    trainer = Trainer(model, data_env, hparams)
    train_batcher = data_env.get_train_batcher(remove_all_head_tail_edges=hparams.remove_all_head_tail_edges)
    logger.info('[Train_Len] {:d}'.format(data_env.n_train))
    logger.info('[Text_Len] {:d}'.format(data_env.n_test))
    t0_tr = time.time()
    n_batches = int(np.ceil(data_env.n_train / hparams.batch_size))
    checkpoint = tf.train.Checkpoint(myModel=model)  # 键名保持为“myModel”
    # checkpoint.restore('../tmp/model.ckpt-9')
    # checkpoint.restore(tf.train.latest_checkpoint('../tmp/')).expect_partial()
    # checkpoint_latest = tf.train.latest_checkpoint('../tmp/')
    # step = checkpoint_latest.split('-')[-1]
    # step = tf.train.get_or_create_global_step()
    # print(int(step))
    run_eval(data_env, model, hparams, 0, 0,
             enable_test_bs=True,
             disable_output_attention=True,
             disable_analyze_attention=True)
    for epoch in range(1, hparams.max_epochs + 1):
        batch_i = 0
        # checkpoint = tf.train.Checkpoint(myModel=model)  # 键名保持为“myModel”
        tmp_100_path = '../tmp/model.ckpt-' + str(epoch)
        checkpoint.restore(tmp_100_path)
        for train_batch, batch_size in train_batcher(hparams.batch_size):
            time_cost = reset_time_cost(hparams)

            heads, tails, rels = train_batch[:, 0], train_batch[:, 1], train_batch[:, 2]
            # heads, tails = train_batch[:, 0], train_batch[:, 1]
            cur_train_loss, cur_accuracy, train_loss, accuracy, train_metric = trainer.train_step(
                heads, tails, rels, epoch, batch_i, tc=time_cost)

            t1_tr = time.time()
            dt_tr = t1_tr - t0_tr
            t0_tr = t1_tr
            if hparams.print_train and batch_i % hparams.print_train_freq == 0:
                if hparams.print_train_metric:
                    hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = train_metric
                    print('[TRAIN] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} {} | '
                          'hit_1: {:.4f} | hit_3: {:.4f} | hit_5: {:.4f} | hit_10: {:.4f} | '
                          'mr: {:.1f} | mmr: {:.4f} | max_r: {:.1f}'.format(
                        epoch, batch_i, train_loss, cur_train_loss, accuracy, cur_accuracy,
                        dt_tr, str_time_cost(time_cost), hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r))
                    logger.info('[TRAIN] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} {} | '
                                'hit_1: {:.4f} | hit_3: {:.4f} | hit_5: {:.4f} | hit_10: {:.4f} | '
                                'mr: {:.1f} | mmr: {:.4f} | max_r: {:.1f}'.format(
                        epoch, batch_i, train_loss, cur_train_loss, accuracy, cur_accuracy,
                        dt_tr, str_time_cost(time_cost), hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r))
                else:
                    print('[TRAIN] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} {}'.format(
                        epoch, batch_i, train_loss, cur_train_loss, accuracy, cur_accuracy, dt_tr,
                        str_time_cost(time_cost)))
                    logger.info(
                        '[TRAIN] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} {}'.format(
                            epoch, batch_i, train_loss, cur_train_loss, accuracy, cur_accuracy, dt_tr,
                            str_time_cost(time_cost)))
            # if batch_i in set([int(p) for p in hparams.eval_within_epoch]):
            batch_i += 1
        t0 = time.time()
        hparams.eval_valid = True
        print('[EVAL] enable test_batch_size')
        run_eval(data_env, model, hparams, epoch, batch_i,
                 enable_test_bs=True,
                 disable_output_attention=True,
                 disable_analyze_attention=True)
        print('[EVAL] {:d}, {:d} | time: {:.4f}'.format(epoch, batch_i, time.time() - t0))
        # checkpoint.save('../tmp/model.ckpt')
    t0 = time.time()
    hparams.eval_valid = False
    print('[EVAL] disable test_batch_size')
    run_eval(data_env, model, hparams, epoch=hparams.max_epochs, batch_i=0,
             enable_test_bs=True,
             disable_output_attention=True,
             disable_analyze_attention=True)
    print('[EVAL]| time: {:.4f}'.format(time.time() - t0))
    print('DONE')


if __name__ == '__main__':

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    # config_t = tf.compat.v1.ConfigProto()
    # config_t.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存
    # sess = tf.compat.v1.Session(config=config_t)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Beauty',
                        choices=['Beauty', 'FB237', 'FB237_v2', 'FB15K', 'WN18RR', 'WN18RR_v2', 'WN', 'YAGO310',
                                 'NELL995', 'CDsVinyl'])
    parser.add_argument('--n_dims_sm', type=int, default=None)
    parser.add_argument('--n_dims', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_edges_per_example', type=int, default=None)
    parser.add_argument('--max_edges_per_node', type=int, default=None)
    parser.add_argument('--max_attended_nodes', type=int, default=None)
    parser.add_argument('--max_seen_nodes', type=int, default=None)
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--test_max_edges_per_example', type=int, default=None)
    parser.add_argument('--test_max_edges_per_node', type=int, default=None)
    parser.add_argument('--test_max_attended_nodes', type=int, default=None)
    parser.add_argument('--test_max_seen_nodes', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--aggregate_op', default=None)
    parser.add_argument('--uncon_steps', type=int, default=None)
    parser.add_argument('--con_steps', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--clipnorm', type=float, default=None)
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=None)
    parser.add_argument('--timer', action='store_true', default=None)
    parser.add_argument('--print_train', action='store_true', default=None)
    parser.add_argument('--print_train_metric', action='store_true', default=None)
    parser.add_argument('--print_train_freq', type=int, default=None)
    parser.add_argument('--eval_within_epoch', default=None)
    parser.add_argument('--eval_valid', action='store_true', default=None)
    parser.add_argument('--moving_mean_decay', type=float, default=None)
    parser.add_argument('--test_output_attention', action='store_true', default=None)
    parser.add_argument('--test_analyze_attention', action='store_true', default=None)
    args = parser.parse_args()

    set_logger('train_log.txt')

    default_parser = config.get_default_config(args.dataset)
    hparams = copy.deepcopy(default_parser.parse_args())
    for arg in vars(args):
        attr = getattr(args, arg)
        if attr is not None:
            setattr(hparams, arg, attr)
    print(hparams)

    if hparams.dataset == 'NELL995':
        nell995_cls = getattr(datasets, hparams.dataset)
        for ds in nell995_cls.datasets():
            print('nell > ' + ds.name)
            if hparams.test_output_attention:
                dir_name = '../output/NELL995_subgraph/' + ds.name
                hparams.dir_name = dir_name
                if os.path.exists(dir_name):
                    shutil.rmtree(dir_name)
                os.makedirs(dir_name)
            run(ds, hparams)
    else:
        ds = getattr(datasets, hparams.dataset)()
        print(ds.name)
        if hparams.test_output_attention:
            dir_name = '../output/' + ds.name + '_subgraph'
            hparams.dir_name = dir_name
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            os.makedirs(dir_name)
        run(ds, hparams)
