import os
import glob
import numpy as np
import matplotlib

# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx
import argparse
import datasets
import matplotlib.patches as mpatches
from matplotlib.legend import Legend
import random

edge_color = ['red', 'orange', 'yellow', 'green', 'pink', 'blue', 'purple', 'black']
entity_color = ['red', 'orange', 'yellow', 'greenyellow', 'pink', 'cyan']


def style_col_node(styles, nodes_style):
    # styles_count = len(styles)
    # until = 1 / (styles_count - 1)
    col = [entity_color[styles.index(style)] for style in nodes_style]
    return col


def style_col_edge(styles, edge_style):
    styles_count = len(styles)
    edge_styles = []
    # until = 1 / (styles_count / 2 - 1)
    col = []
    styles = styles[:8]
    color = edge_color
    for style in edge_style:
        if style in styles:
            # col.append(styles.index(style) * until)
            col.append(color[styles.index(style)])
            edge_styles.append(style)
        else:
            index = styles.index(style.split('_', 1)[-1])
            col.append(color[index])
            edge_styles.append(style.split('_', 1)[-1])
    return col, edge_styles


def config_entity_style():
    entity_file = "../data/KBC/Beauty/kg_entities.txt"
    id2entity_style = {}  # {} = dict
    style = []
    with open(entity_file, "rb") as f:
        for line in f:
            # Format: [entity_global_id]\t[entity_type]_[entity_local_id]\t[entity_value]
            cells = line.decode().strip().split("\t")
            # global_id 是整個圖的id
            global_id = int(cells[0])
            entity_eid = cells[1].rsplit("_", maxsplit=1)
            # eid 是對應實體類的id
            # entity 是實體類型
            entity = entity_eid[0]
            if entity not in style:
                style.append(entity)
            # entity => entity_type
            id2entity_style[global_id] = entity
    return id2entity_style, style


def config_rel_style():
    entity_file = "../data/KBC/Beauty/kg_relations.txt"
    id2rel_style = {}  # {} = dict
    style = []
    with open(entity_file, "rb") as f:
        for line in f:
            # Format: [entity_global_id]\t[entity_type]_[entity_local_id]\t[entity_value]
            cells = line.decode().strip().split("\t")
            # global_id 是整個圖的id
            relation_rid = int(cells[0])
            relation = cells[1]
            if relation not in style:
                style.append(relation)
            # eid 是對應實體類的id
            # entity 是實體類型
            # entity => entity_type
            id2rel_style[relation_rid] = relation
    return id2rel_style, style


def draw_a_graph(filename, dataset, topk_all=None, topk_per_step=None, font_size=4, node_size=100, edge_width=0.5,
                 disable_draw=False, id2entity_style=None, id2rel_style=None, explain=False,
                 entity_style=None, rel_style=None):
    nodes_per_step = []
    rels_dct = {}
    edges_set = {}

    head, tail = os.path.basename(filename)[len('test_epoch1_'):].split('.')[0].split('-_')
    head = head.split('(')[0].split('_')[1]

    tail = tail.split('(')[0]
    with open(filename) as fin:
        mode = None
        for line in fin.readlines():
            line = line.strip()
            if line == 'nodes:':
                mode = 'nodes'
            elif line == 'edges:':
                mode = 'edges'
            else:
                if mode == 'nodes':
                    nodes = []
                    max_att = 0.
                    for sp in line.split('\t'):
                        sp2 = sp.split(':')
                        node_att = float(sp2[1])
                        max_att = max(max_att, node_att)
                        sp2 = sp2[0].split('(')
                        node_id = sp2[0]
                        node_name = sp2[1][:-1]
                        nodes.append((node_id, node_name, node_att))

                    if topk_per_step is not None:
                        sorted(nodes, key=lambda node: - node[2])
                        nodes = nodes[:topk_per_step]

                    nodes = {node_id: (node_name, node_att / max_att) for node_id, node_name, node_att in nodes}
                    nodes_per_step.append(nodes)

                elif mode == 'edges':
                    edges = []
                    for sp in line.split('\t'):
                        sp2 = sp.split('->')
                        node_id1 = sp2[0].split('(')[0]
                        node_id2 = sp2[2].split('(')[0]
                        rel_id = sp2[1].split('(')[0]
                        rel_name = sp2[1].split('(')[1][:-1]
                        rels_dct[rel_id] = rel_name
                        edges.append((node_id1, rel_id, node_id2))
                        if (node_id1, rel_id, node_id2) not in edges_set:
                            edges_set[(node_id1, rel_id, node_id2)] = len(edges_set)

    nodes = {}
    n_steps = len(nodes_per_step)

    for t in range(n_steps):
        for k, v in nodes_per_step[t].items():
            node_id = k
            node_name = v[0]
            node_att = v[1]
            att_h = node_att * np.power(0.9, t)
            att_t = node_att * np.power(0.9, n_steps - 1 - t)
            att = 0.5 - att_h / 2 if att_h > att_t else 0.5 + att_t / 2
            node_att = nodes[node_id][1] if node_id in nodes else 0.5
            nodes[node_id] = (node_name, att) if abs(att - 0.5) > abs(node_att - 0.5) else (node_name, node_att)

    left, right, bottom, top = 3.0, 8.0, 2.0, 8.0
    len_node = len(nodes)
    v_spacing = (top - bottom) / float(len_node / 2)
    h_spacing = (right - left) / 5

    nodes_all = []
    if str(tail) not in nodes:
        return 0
    else:
        if explain:
            i = [0, 0, 0, 0, 0, 0]
            nodes_count = []
            layer_top = (top + bottom) / 2.
            nodes_all.append((i[0], {'id': str(head), 'name': nodes[str(head)][0], 'att': nodes[str(head)][1],
                                     'pos': (left, layer_top - 2 * v_spacing)}))
            nodes_all.append((len_node - 1, {'id': str(tail), 'name': nodes[str(tail)][0], 'att': nodes[str(tail)][1],
                                             'pos': (left + 6 * h_spacing, layer_top - 2 * v_spacing)}))
            for edge in edges_set:
                node = edge[-1]
                if node not in nodes:
                    continue
                if int(node) == int(head) or int(node) == int(tail):
                    continue
                if int(edge[0]) == int(head) and node not in nodes_count and i[1] < len_node / 5:
                    i[0] += 1
                    i[1] += 1
                    nodes_count.append(node)
                    rel1 = random.uniform(0.9, 1.1)
                    rel2 = random.uniform(0.8, 1.0)
                    nodes_all.append((i[0], {'id': node, 'name': nodes[node][0], 'att': nodes[node][1],
                                             'pos': (
                                             left + rel1 * 1.5 * h_spacing, layer_top - rel2 * i[1] * v_spacing)}))
                elif i[2] < len_node / 5 and node not in nodes_count:
                    i[0] += 1
                    i[2] += 1
                    nodes_count.append(node)
                    rel1 = random.uniform(0.9, 1.1)
                    rel2 = random.uniform(0.8, 1.0)
                    nodes_all.append((i[0], {'id': node, 'name': nodes[node][0], 'att': nodes[node][1],
                                             'pos': (
                                             left + rel1 * 2.5 * h_spacing, layer_top - rel2 * i[2] * v_spacing)}))
                elif i[3] < len_node / 5 and node not in nodes_count:
                    i[0] += 1
                    i[3] += 1
                    nodes_count.append(node)
                    rel1 = random.uniform(0.9, 1.1)
                    rel2 = random.uniform(0.8, 1.0)
                    nodes_all.append((i[0], {'id': node, 'name': nodes[node][0], 'att': nodes[node][1],
                                             'pos': (
                                             left + rel1 * 3.5 * h_spacing, layer_top - rel2 * i[3] * v_spacing)}))
                elif i[4] < len_node / 5 and node not in nodes_count:
                    i[0] += 1
                    i[4] += 1
                    nodes_count.append(node)
                    rel1 = random.uniform(0.9, 1.1)
                    rel2 = random.uniform(0.8, 1.0)
                    nodes_all.append((i[0], {'id': node, 'name': nodes[node][0], 'att': nodes[node][1],
                                             'pos': (
                                             left + rel1 * 4.5 * h_spacing, layer_top - rel2 * i[4] * v_spacing)}))
                elif i[5] < len_node / 5 and node not in nodes_count:
                    i[0] += 1
                    i[5] += 1
                    nodes_count.append(node)
                    rel1 = random.uniform(0.9, 1.1)
                    rel2 = random.uniform(0.8, 1.0)
                    nodes_all.append((i[0], {'id': node, 'name': nodes[node][0], 'att': nodes[node][1],
                                             'pos': (
                                             left + rel1 * 5.5 * h_spacing, layer_top - rel2 * i[5] * v_spacing)}))

        else:
            nodes_all = [(i, {'id': k, 'name': v[0], 'att': v[1]}) for i, (k, v) in enumerate(nodes.items())]

        graph = nx.MultiGraph()
        if topk_all is not None:
            sorted(nodes_all, key=lambda node: - node[1]['att'])
            nodes_all = [(i, e[1]) for i, e in enumerate(nodes_all[:topk_all])]
            # for node in nodes_all:
            #     graph.add_node(node, pos=node[1]['pos'])

        id2i = {node[1]['id']: node[0] for node in nodes_all}
        i2id = {node[0]: node[1]['id'] for node in nodes_all}

        edges = list(edges_set.keys())
        sorted(edges, key=lambda e: edges_set[e])
        edges_all = [(id2i[n1], id2i[n2], {'rel_id': r, 'rel_name': rels_dct[r]})
                     for n1, r, n2 in edges if n1 in id2i and n2 in id2i and dataset.id2relation[int(r)] < 8]
        edges_count = [(edge[0], edge[1]) for edge in edges_all]
        for edge in edges:
            if edge[0] in id2i and edge[2] in id2i and (id2i[edge[0]], id2i[edge[2]]) not in edges_count and \
                    (id2i[edge[2]], id2i[edge[0]]) not in edges_count:
                edges_all.append((id2i[edge[0]], id2i[edge[2]], {'rel_id': edge[1], 'rel_name': rels_dct[edge[1]]}))

        if not disable_draw:
            graph.add_nodes_from(nodes_all)
            graph.add_edges_from(edges_all)
            if explain:
                pos = nx.get_node_attributes(graph, 'pos')
            else:
                pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')

            def get_label(node):
                node_ = node[1]
                node_label = id2entity_style[dataset.id2entity[int(node_['id'])]] + ':' + \
                             str(dataset.id2entity[int(node_['id'])])
                return node_label

            def get_node_params(nodes, size=100, inflate=0, min_max=0.1):
                node_atts = [n[1]['att'] for n in nodes]
                nodes_style = [id2entity_style[dataset.id2entity[int(n[1]['id'])]] for n in nodes]
                cols = style_col_node(styles=entity_style, nodes_style=nodes_style)
                c = np.array(cols)
                # vmin = c.min()
                # vmax = max(c.max(), min_max)
                sizes = [
                    size * 2.5 if nodes[i][1]['id'] == head or nodes[i][1]['id'] == tail else size * (1 + a * inflate)
                    for i, a in enumerate(node_atts)]
                return cols, sizes

            cols, sizes = get_node_params(nodes_all, size=node_size)
            nx.draw_networkx_nodes(graph, pos, node_color=cols, node_size=sizes, edgecolors='y', alpha=0.95,
                                   linewidths=0.5)

            # nx.draw_networkx_nodes(graph, pos, node_color=cols, vmin=vmin, vmax=vmax, cmap=attcmp, node_size=sizes,
            #                        edgecolors='y', alpha=0.95, linewidths=0.5)

            def get_edge_params(edges, nodes=None, width=1., inflate=2):
                node_atts = [n[1]['att'] for n in nodes]
                edge_atts = [0.5 - (abs(node_atts[e[0]] - 0.5) + abs(node_atts[e[1]] - 0.5)) / 2.
                             if (node_atts[e[0]] + node_atts[e[1]]) / 2. < 0.5 else
                             0.5 + (abs(node_atts[e[0]] - 0.5) + abs(node_atts[e[1]] - 0.5)) / 2.
                             for e in edges]
                edges_style = [id2rel_style[dataset.id2relation[int(e[2]['rel_id'])]] for e in edges]
                cols, style = style_col_edge(styles=rel_style, edge_style=edges_style)
                # c = np.array(cols)
                # vmin = c.min()
                # vmax = c.max()
                widths = [width * (1 + a * inflate) for a in edge_atts]
                return cols, widths, style

            # e_vmin, e_vmax,
            e_cols, widths, edges_style = get_edge_params(edges_all, nodes_all, width=edge_width)
            edge_list = [(edge[0], edge[1]) for edge in edges_all]
            # for index, edge in enumerate(edges_all):
            #     print(id2rel_style[dataset.id2relation[int(edge[2]['rel_id'])]], e_cols[index])
            nx.draw_networkx_edges(graph, pos, edgelist=edge_list, width=widths, edge_color=e_cols,
                                   # edge_vmin=e_vmin,edge_vmax=e_vmax,edge_cmap=attcmp,
                                   arrowstyle='->', alpha=0.7)
            # nx.draw_networkx_edges(graph, pos, edge_vmin=e_vmin, edge_vmax=e_vmax,
            #                        arrowstyle='->', alpha=0.7)

            if explain:
                nx.draw_networkx_labels(graph, pos, labels={nd[0]: get_label(nd) for nd in nodes_all},
                                        font_size=font_size, font_color='k')
            else:
                nx.draw_networkx_labels(graph, pos, labels={nd[0]: nd[1]['name'] for nd in nodes_all},
                                        font_size=font_size, font_color='k')
            # assert nodes_all[id2i[head]][1]['name'] == dataset.id2entity[int(head)]

            if tail not in id2i:
                plt.title('{} - {} (missed)'.format(head, tail))
            # if explain:
            #     nx.draw_networkx_edge_labels(graph, pos, edge_labels={
            #         (e[0], e[1]): edges_style[index] + ':' + e_cols[index]
            #         for index, e in enumerate(edge_list)}, font_size=1, font_color='k')

        edges = [(i2id[e[0]], e[2]['rel_id'], i2id[e[1]]) for e in edges_all]
        edge_labels = rel_style[:8]

        # until = 1 / (len(entity_style) - 1)
        # node_colors = [index for index in range(len(entity_style))]
        # entity_color = edge_color[:6]
        # 用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
        edges_patches = [mpatches.Patch(color=edge_color[i], label="{:s}".format(edge_labels[i])) for i in
                         range(len(edge_color))]
        entity_patches = [mpatches.Patch(color=entity_color[i], label="{:s}".format(entity_style[i])) for i in
                          range(len(entity_style))]
        ax = plt.gca()
        # ax.legend(handles=patches, ncol=1, loc='upper left')
        ax.legend(handles=edges_patches, ncol=1, loc='lower left')
        leg = Legend(ax, handles=entity_patches, ncol=1, loc='lower right', labels=entity_style)
        ax.add_artist(leg)

        # return head, relation, tail, edges
        return head, tail, edges


def draw(dataset, dirpath, new_dirpath, id2entity_style, id2rel_style, entity_style, rel_style):
    if not os.path.exists(new_dirpath):
        os.mkdir(new_dirpath)

    for filename in glob.glob(os.path.join(dirpath, '*.txt')):
            print(filename)
            plt.figure(figsize=(8, 10))
            plt.subplot(2, 1, 1)
            if draw_a_graph(filename, dataset, font_size=3, id2entity_style=id2entity_style, id2rel_style=id2rel_style,
                            explain=False, entity_style=entity_style, rel_style=rel_style) == 0:
                continue
            else:
                try:
                    draw_a_graph(filename, dataset, font_size=3, id2entity_style=id2entity_style,
                                 id2rel_style=id2rel_style,
                                 explain=False, entity_style=entity_style, rel_style=rel_style)
                    plt.subplot(2, 1, 2)
                    draw_a_graph(filename, dataset, topk_per_step=5, font_size=5, node_size=180, edge_width=1,
                                 id2entity_style=id2entity_style, id2rel_style=id2rel_style, explain=True,
                                 entity_style=entity_style, rel_style=rel_style)

                    plt.tight_layout()
                    plt.savefig(os.path.join(new_dirpath, os.path.basename(filename)[:-4] + '.pdf'), format='pdf')
                    plt.close()
                    # head, tail, edges = draw_a_graph(filename, dataset, topk_per_step=3, font_size=5, node_size=180,
                    #                                  edge_width=1, id2entity_style=id2entity_style,
                    #                                  id2rel_style=id2rel_style,
                    #                                  explain=True, entity_style=entity_style, rel_style=rel_style)
                    #
                    # with open(os.path.join(new_dirpath, os.path.basename(filename)), 'w') as fout:
                    #     fout.write('{}\t{}\n\n'.format(dataset.id2entity[int(head)],
                    #                                    # dataset.id2relation[int(rel)],
                    #                                    dataset.id2entity[int(tail)]))
                    #     for h, r, t in edges:
                    #         fout.write('{}\t{}\t{}\n'.format(dataset.id2entity[int(h)],
                    #                                          dataset.id2relation[int(r)],
                    #                                          dataset.id2entity[int(t)]))
                except IndexError as e:
                    print('Cause `IndexError` for file `{}`'.format(filename))
                    print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Beauty',
                        choices=['FB237', 'FB237_v2', 'FB15K', 'WN18RR', 'WN18RR_v2', 'WN', 'YAGO310', 'NELL995',
                                 'Beauty'])
    args = parser.parse_args()
    id2entity_style, entity_style = config_entity_style()
    id2rel_style, rel_style = config_rel_style()
    # ds = getattr(datasets, args.dataset)()
    # if args.dataset == 'NELL995':
    #     nell995_cls = getattr(datasets, args.dataset)
    #     for ds in nell995_cls.datasets():
    #         print('nell > ' + ds.name)
    #         dir_name = '../output/NELL995_subgraph/' + ds.name
    #         if not os.path.exists(dir_name):
    #             continue
    #         dir_name_2 = '../visual/NELL995_subgraph/' + ds.name
    #         os.makedirs(dir_name_2, exist_ok=True)
    #         draw(ds, dir_name, dir_name_2)
    ds = getattr(datasets, args.dataset)()
    print(ds.name)
    dir_name = '../output/' + ds.name + '_subgraph'
    dir_name_2 = '../visual/' + ds.name + '_subgraph'
    os.makedirs(dir_name_2, exist_ok=True)
    draw(ds, dir_name, dir_name_2, id2entity_style, id2rel_style, entity_style, rel_style)
